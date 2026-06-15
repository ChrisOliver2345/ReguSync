import sys
from torch import nn
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from flash import FlashTransformerEncoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from grad_reverse import grad_reverse
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial
from torch.distributions import Poisson, Bernoulli
import numpy as np

class Gene_Transformer(nn.Module):
    def __init__(self,
                 args,
                 ntoken: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 n_features_a: int,
                 n_features_b: int,
                 vocab: Any = None,
                 dropout: float = 0.5,
                 pad_token: str = "<pad>",
                 pad_value: int = 0,
                 n_input_bins: Optional[int] = None
                 ):
        super().__init__()
        self.args = args
        self.d_model = d_model
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = CategoryValueEncoder(n_input_bins, d_model, padding_idx=pad_value)
        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        if args.grn_la == True:
            self.pos_emb = nn.Embedding(args.max_seq_len, d_model)
            self.level_emb = nn.Embedding(args.max_la+1, d_model)

        self.basic_encoder_a = BasicEncoder(in_features=n_features_a, out_features=d_model, dropout=dropout)
        self.basic_encoder_b = BasicEncoder(in_features=n_features_b, out_features=d_model, dropout=dropout)

        encoder_layers = FlashTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
            norm_scheme='pre',
            causal=True
        )

        self.transformer_encoder_early = TransformerEncoder(encoder_layers, num_layers=2)
        self.transformer_encoder_late = TransformerEncoder(encoder_layers, num_layers=nlayers-2)

        encoder_layers_light = FlashTransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
            norm_scheme='pre',
            causal=False
        )

        self.generator_a = TransformerEncoder(encoder_layers_light, num_layers=1)
        self.generator_b = TransformerEncoder(encoder_layers_light, num_layers=1)

        self.decoder_a = ExprDecoder(seq_len=args.max_seq_len, output_dim=n_features_a, d_model=d_model)
        self.decoder_b = ExprDecoder(seq_len=args.max_seq_len, output_dim=n_features_b, d_model=d_model)

        # self.cls_decoder = ClsDecoder(d_model=d_model, n_cls=n_cls, nlayers=2)

        self.sim = Similarity(temp=0.5)  # TODO: auto set temp

        self.discriminator_a = Discriminator(d_model=d_model, seq_len=args.max_seq_len, hidden_dim=args.d_model, reverse_grad=True)
        self.discriminator_b = Discriminator(d_model=d_model, seq_len=args.max_seq_len, hidden_dim=args.d_model, reverse_grad=True)
        # 梯度反转使生成器优化方向与判别器相反，试图让判别器输出接近均匀分布（即无法区分真实和生成）

        self.cross_attn_a = CrossAttentionBlock(dim=d_model, num_heads=2, mlp_ratio=4., qkv_bias=True, drop=dropout,
                                                attn_drop=dropout, has_mlp=False)
        self.cross_attn_b = CrossAttentionBlock(dim=d_model, num_heads=2, mlp_ratio=4., qkv_bias=True, drop=dropout,
                                                attn_drop=dropout, has_mlp=False)

        self.gate_a = GateMLP(d_model=d_model)
        self.gate_b = GateMLP(d_model=d_model)

        self.alignment = AlignmentModule(dim_token=d_model, dim_graph=args.d_graph, dropout=dropout)

        self.cell_mlp = nn.Sequential(
            nn.Linear(args.max_seq_len, args.d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.d_model * 4, args.d_model)
        )

        if self.args.modal_a_loss == 'nb':
            self.theta_aa = nn.Parameter(torch.randn(n_features_a), requires_grad=True)
            self.theta_ba = nn.Parameter(torch.randn(n_features_a), requires_grad=True)

        if self.args.modal_b_loss == 'nb':
            self.theta_ab = nn.Parameter(torch.randn(n_features_b), requires_grad=True)
            self.theta_bb = nn.Parameter(torch.randn(n_features_b), requires_grad=True)

    def _get_cell_emb_from_layer(
            self, layer_output: Tensor,
    ) -> Tensor:

        cell_avg = torch.mean(layer_output, dim=2)
        cell_emb = self.cell_mlp(cell_avg)

        return cell_emb

    def _get_gene_emb_from_layer(
            self, layer_output: Tensor,
    ) -> Tensor:

        genes_embd = layer_output

        return genes_embd

    def forward(
            self,
            src: Tensor,
            values: Tensor,
            src_key_padding_mask: Tensor,
            basic_a: Tensor,
            basic_b: Tensor,
            values_grn: Tensor,
            do_train: bool,
            level: Tensor = None,
    ) -> Mapping[str, Tensor]:

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        total_embs = src + values
        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.args.grn_la == True:
            position_ids = torch.arange(total_embs.size(1), device=self.args.device)
            position_ids = position_ids.unsqueeze(0).expand(total_embs.size(0), -1)
            pos_embs = self.pos_emb(position_ids).to(total_embs.dtype)
            
            level_ids = level.unsqueeze(0).expand(total_embs.size(0), -1)
            level_embs = self.level_emb(level_ids).to(total_embs.dtype)

            total_embs = total_embs + pos_embs + level_embs

        early_output = self.transformer_encoder_early(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        fused_embds = self.alignment(early_output, values_grn)
        # fused_embds = early_output

        transformer_output = self.transformer_encoder_late(
            fused_embds, src_key_padding_mask=src_key_padding_mask
        )

        basic_embds_a = self.basic_encoder_a(basic_a)
        basic_embds_b = self.basic_encoder_b(basic_b)

        batch_size = transformer_output.size(0) // 2
        cells_embd = self._get_cell_emb_from_layer(transformer_output)
        cells_embd_a = cells_embd[:batch_size]
        cells_embd_b = cells_embd[batch_size:]

        transformer_output_a = transformer_output[:batch_size]
        transformer_output_b = transformer_output[batch_size:]

        concat_a = torch.cat([cells_embd_a.detach(), basic_embds_a], dim=-1)
        beta_a = self.gate_a(concat_a)
        cells_embd_a = cells_embd_a + beta_a * basic_embds_a 

        concat_b = torch.cat([cells_embd_b.detach(), basic_embds_b], dim=-1)
        beta_b = self.gate_b(concat_b)
        cells_embd_b = cells_embd_b + beta_b * basic_embds_b 

        genes_embd_a = self._get_gene_emb_from_layer(transformer_output_a)
        genes_embd_b = self._get_gene_emb_from_layer(transformer_output_b)

        pred_genes_embd_b = self.generator_a(genes_embd_a, src_key_padding_mask=src_key_padding_mask[:batch_size, :-1])
        pred_genes_embd_a = self.generator_b(genes_embd_b, src_key_padding_mask=src_key_padding_mask[batch_size:, :-1])

        concat_embd_a = torch.cat([genes_embd_a, pred_genes_embd_a], dim=0)
        concat_embd_b = torch.cat([genes_embd_b, pred_genes_embd_b], dim=0)

        disc_a = self.discriminator_a(concat_embd_a)
        disc_b = self.discriminator_b(concat_embd_b)

        embd_a_b = torch.cat([cells_embd_a.unsqueeze(1), genes_embd_b], dim=1)
        t_cells_embd_b = self.cross_attn_a(embd_a_b).squeeze(1)  # 把a翻译为b

        embd_b_a = torch.cat([cells_embd_b.unsqueeze(1), genes_embd_a], dim=1)
        t_cells_embd_a = self.cross_attn_b(embd_b_a).squeeze(1)  # 把b翻译为a

        output = {}
        recons_aa = self.decoder_a(cell_emb=cells_embd_a, gene_emb=genes_embd_a)
        recons_bb = self.decoder_b(cell_emb=cells_embd_b, gene_emb=genes_embd_b)

        if do_train:
            recons_ab = self.decoder_b(cell_emb=t_cells_embd_b, gene_emb=genes_embd_b)
            recons_ba = self.decoder_a(cell_emb=t_cells_embd_a, gene_emb=genes_embd_a)
        else:
            recons_ab = self.decoder_b(cell_emb=t_cells_embd_b, gene_emb=pred_genes_embd_b)
            recons_ba = self.decoder_a(cell_emb=t_cells_embd_a, gene_emb=pred_genes_embd_a)


        output["recons_a2a"] = recons_aa
        output["recons_a2b"] = recons_ab
        output["recons_b2b"] = recons_bb
        output["recons_b2a"] = recons_ba

        # output["cls_output"] = self.cls_decoder(cells_embd)  # (batch, n_cls)

        output["basic_embds_a"] = basic_embds_a
        output["basic_embds_b"] = basic_embds_b
        output["cells_embd_a"] = cells_embd_a
        output["cells_embd_b"] = cells_embd_b
        output["cells_t_embd_a"] = t_cells_embd_a
        output["cells_t_embd_b"] = t_cells_embd_b
        output["disc_a"] = disc_a
        output["disc_b"] = disc_b

        return output

    def compute_loss(self, output_dict, celltype_labels, true_value_a, true_value_b, locs_a, locs_b):
        
        # gamma = 1.0
        cell_embds_a = output_dict["cells_embd_a"]
        cell_embds_b = output_dict["cells_embd_b"]
        # base_embds_a = output_dict["basic_embds_a"]
        # base_embds_b = output_dict["basic_embds_b"]
        # loss_cos_a = F.cosine_similarity(cell_embds_a, base_embds_a, dim=-1).mean()
        # loss_mse_a = F.mse_loss(cell_embds_a, base_embds_a)
        # loss_reg_a = (1 - loss_cos_a) + gamma * loss_mse_a
        # loss_cos_b = F.cosine_similarity(cell_embds_b, base_embds_b, dim=-1).mean()
        # loss_mse_b = F.mse_loss(cell_embds_b, base_embds_b)
        # loss_reg_b = (1 - loss_cos_b) + gamma * loss_mse_b
        # loss_reg = (loss_reg_a + loss_reg_b) / 2
        loss_reg = torch.tensor(0.0, device=self.args.device)

        if self.args.spatial == True:
            spatial_regularization_strength = 0.05
            regularization_acceleration = False  

            coords = locs_a  # 假设 locs_a 和 locs_b 相同
            coords = coords.to(self.args.device)

            def compute_spatial_penalty(emb, coords, regularization_acceleration, edge_subset_sz=1000000):
                emb = emb.to(self.args.device)
                if regularization_acceleration or emb.shape[0] > 5000:
                    cell_random_subset_1 = torch.randint(0, emb.shape[0], (edge_subset_sz,)).to(emb.device)
                    cell_random_subset_2 = torch.randint(0, emb.shape[0], (edge_subset_sz,)).to(emb.device)
                    emb1 = torch.index_select(emb, 0, cell_random_subset_1)
                    emb2 = torch.index_select(emb, 0, cell_random_subset_2)
                    c1 = torch.index_select(coords, 0, cell_random_subset_1)
                    c2 = torch.index_select(coords, 0, cell_random_subset_2)
                    pdist = nn.PairwiseDistance(p=2)

                    emb_dists = pdist(emb1, emb2)
                    emb_dists = emb_dists / torch.max(emb_dists) if torch.max(emb_dists) > 0 else emb_dists

                    sp_dists = pdist(c1, c2)
                    sp_dists = sp_dists / torch.max(sp_dists) if torch.max(sp_dists) > 0 else sp_dists

                    n_items = emb_dists.size(0)
                else:
                    emb_dists = torch.cdist(emb, emb, p=2)
                    emb_dists = emb_dists / torch.max(emb_dists) if torch.max(emb_dists) > 0 else emb_dists

                    sp_dists = torch.cdist(coords, coords, p=2)
                    sp_dists = sp_dists / torch.max(sp_dists) if torch.max(sp_dists) > 0 else sp_dists

                    n_items = emb.size(0) * emb.size(0)

                penalty = torch.sum(torch.mul(1.0 - emb_dists, sp_dists)) / n_items
                return penalty

            # 分别计算每个模态的空间惩罚
            penalty_a = compute_spatial_penalty(cell_embds_a, coords, regularization_acceleration)
            penalty_b = compute_spatial_penalty(cell_embds_b, coords, regularization_acceleration)

            # 总空间损失
            loss_reg_spatial = spatial_regularization_strength * (penalty_a + penalty_b) / 2
        else:
            loss_reg_spatial = torch.tensor(0.0, device=self.args.device)

    
        # Assuming batch size can be inferred from cell_embds_a.size(0)
        bs = cell_embds_a.size(0)
        disc_labels = torch.cat([torch.zeros(bs, dtype=torch.long), torch.ones(bs, dtype=torch.long)], dim=0).to(self.args.device)
        ce_loss = nn.CrossEntropyLoss()
        loss_disc_a = ce_loss(output_dict["disc_a"], disc_labels)
        loss_disc_b = ce_loss(output_dict["disc_b"], disc_labels)
        loss_disc = (loss_disc_a + loss_disc_b) / 2

        # criterion_cls = nn.CrossEntropyLoss()
        # loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
        loss_cls = torch.tensor(0.0, device=self.args.device)

        # cell_t_embds_a = output_dict["cells_t_embd_a"]
        # cell_t_embds_b = output_dict["cells_t_embd_b"]
        # loss_translate_a = F.mse_loss(cell_embds_a, cell_t_embds_a)
        # loss_translate_b = F.mse_loss(cell_embds_b, cell_t_embds_b)
        # loss_translate = (loss_translate_a + loss_translate_b) / 2
        # loss_translate = torch.tensor(0.0, device=self.args.device)

        if self.args.modal_a == 'RNA':

            if self.args.modal_a_loss == 'nb':
                size_factor_a = true_value_a.sum(1).unsqueeze(1).to(self.args.device)
                theta_aa = torch.exp(self.theta_aa)  # 使用 self.theta_aa
                mu_aa = F.softmax(output_dict["recons_a2a"], dim=1) * size_factor_a
                px_aa = NegativeBinomial(mu=mu_aa, theta=theta_aa)
                loss_recons_aa = -px_aa.log_prob(true_value_a).sum(1).mean()

                theta_ba = torch.exp(self.theta_ba)  # 使用 self.theta_ba
                mu_ba = F.softmax(output_dict["recons_b2a"], dim=1) * size_factor_a
                px_ba = NegativeBinomial(mu=mu_ba, theta=theta_ba)
                loss_recons_ba = -px_ba.log_prob(true_value_a).sum(1).mean()

            elif self.args.modal_a_loss == 'mse':
                size_factor_a = true_value_a.sum(1).unsqueeze(1).to(self.args.device)
                mu_aa = F.softmax(output_dict["recons_a2a"], dim=1) * size_factor_a
                loss_recons_aa = F.mse_loss(mu_aa, true_value_a, reduction='mean')

                mu_ba = F.softmax(output_dict["recons_b2a"], dim=1) * size_factor_a
                loss_recons_ba = F.mse_loss(mu_ba, true_value_a, reduction='mean')
                      

        if self.args.modal_b == 'ATAC':
            
            if self.args.modal_b_loss == 'bernoulli':
                probs_ab = F.sigmoid(output_dict["recons_a2b"])
                px_ab = Bernoulli(probs=probs_ab)
                loss_recons_ab = -px_ab.log_prob(true_value_b).sum(1).mean()
                # 不要.sum(1).mean() 。先计算每个细胞的总损失,然后求细胞的平均

                probs_bb = F.sigmoid(output_dict["recons_b2b"])
                px_bb = Bernoulli(probs=probs_bb)
                loss_recons_bb = -px_bb.log_prob(true_value_b).sum(1).mean()
            
            elif self.args.modal_b_loss == 'bce':
                bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
                loss_recons_ab = bce_loss(output_dict["recons_a2b"], true_value_b)
                loss_recons_bb = bce_loss(output_dict["recons_b2b"], true_value_b)

            elif self.args.modal_b_loss == 'nb':
                size_factor_b = true_value_b.sum(1).unsqueeze(1).to(self.args.device)
                theta_bb = torch.exp(self.theta_bb)
                mu_bb = F.softmax(output_dict["recons_b2b"], dim=1) * size_factor_b
                px_bb = NegativeBinomial(mu=mu_bb, theta=theta_bb)
                loss_recons_bb = -px_bb.log_prob(true_value_b).sum(1).mean()

                theta_ab = torch.exp(self.theta_ab)  
                mu_ab = F.softmax(output_dict["recons_a2b"], dim=1) * size_factor_b
                px_ab = NegativeBinomial(mu=mu_ab, theta=theta_ab)
                loss_recons_ab = -px_ab.log_prob(true_value_b).sum(1).mean()            

        elif self.args.modal_b == 'ADT':
            if self.args.modal_b_loss == 'mse':
                mu_bb = F.softmax(output_dict["recons_b2b"], dim=1)
                loss_recons_bb = F.mse_loss(mu_bb, true_value_b, reduction='mean')
                mu_ab = F.softmax(output_dict["recons_a2b"], dim=1)
                loss_recons_ab = F.mse_loss(mu_ab, true_value_b, reduction='mean')

            elif self.args.modal_b_loss == 'nb':
                size_factor_b = true_value_b.sum(1).unsqueeze(1).to(self.args.device)
                theta_ab = torch.exp(self.theta_ab)  # 使用 self.theta_ab
                mu_ab = F.softmax(output_dict["recons_a2b"], dim=1) * size_factor_b
                px_ab = NegativeBinomial(mu=mu_ab, theta=theta_ab)
                loss_recons_ab = -px_ab.log_prob(true_value_b).sum(1).mean()

                theta_bb = torch.exp(self.theta_bb)  # 使用 self.theta_bb
                mu_bb = F.softmax(output_dict["recons_b2b"], dim=1) * size_factor_b
                px_bb = NegativeBinomial(mu=mu_bb, theta=theta_bb)
                loss_recons_bb = -px_bb.log_prob(true_value_b).sum(1).mean()
                
            else:
                raise ValueError(f"Invalid modal_b_loss: {self.args.modal_b_loss}")

        if self.args.modal_a_loss_lambda is not None or self.args.modal_b_loss_lambda is not None:
            loss_recons = (self.args.modal_a_loss_lambda*loss_recons_aa + self.args.modal_b_loss_lambda*loss_recons_ab + self.args.modal_b_loss_lambda*loss_recons_bb + self.args.modal_a_loss_lambda*loss_recons_ba) / 4.0
        else:
            loss_recons = (loss_recons_aa + loss_recons_ab + loss_recons_bb + loss_recons_ba) / 4.0
        # loss_recons = loss_recons_ab

        if self.args.spatial == True:
            loss_total = loss_disc + loss_recons + loss_reg_spatial
        else:
            loss_total = loss_disc + loss_recons

        return loss_total, loss_reg, loss_disc, loss_cls, loss_recons, loss_reg_spatial


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ExprDecoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        output_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, output_dim)
        )

    def forward(
            self,
            cell_emb: Tensor,
            gene_emb: Tensor
    ) -> Dict[str, Tensor]:

        # cell_emb_expanded = cell_emb.unsqueeze(-1)
        # recon_matrix = torch.bmm(gene_emb, cell_emb_expanded).squeeze(-1)
        recon_matrix = cell_emb
        recon_output = self.output_layer(recon_matrix)

        return recon_output


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
) -> Tensor:
    """
    Args:
        layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
        weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
            when :attr:`self.cell_emb_style` is "w-pool".

    Returns:
        :obj:`Tensor`: shape (batch, embsize)
    """
    if self.cell_emb_style == "cls":
        cell_emb = layer_output[:, 0, :]  # (batch, embsize)
    elif self.cell_emb_style == "avg-pool":
        cell_emb = torch.mean(layer_output, dim=1)
    elif self.cell_emb_style == "w-pool":
        if weights is None:
            raise ValueError("weights is required when cell_emb_style is w-pool")
        if weights.dim() != 2:
            raise ValueError("weights should be 2D")
        cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
        cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

    return cell_emb


class BasicEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Discriminator(nn.Module):

    def __init__(
            self,
            d_model: int,
            seq_len: int,
            hidden_dim: int = 256,
            reverse_grad: bool = False
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model * seq_len, hidden_dim),  # 展平输入
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.out_layer = nn.Linear(hidden_dim // 4, 2)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:

        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)

        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # 展平为 (batch_size, (seq_len-1)*d_model)
        x = self.model(x)

        return self.out_layer(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.attn(self.norm1(x))
        if self.has_mlp:
            x = x + self.mlp(self.norm2(x))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果未指定输出维度，默认等于输入维度
        hidden_features = hidden_features or in_features  # 如果未指定隐藏维度，默认等于输入维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GateMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, concat_emb):
        return torch.sigmoid(self.mlp(concat_emb))


class AlignmentModule(nn.Module):
    def __init__(self, dim_token, dim_graph, dropout=0.1):
        super(AlignmentModule, self).__init__()

        self.proj = nn.Linear(dim_graph, dim_token)
        self.ffn = nn.Sequential(
            nn.Linear(dim_token, dim_token * 2),
            nn.ReLU(),
            nn.Linear(dim_token * 2, dim_token)
        )
        self.norm = nn.LayerNorm(dim_token)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, token_embds, gene_embds):

        batch_size, num_tokens, _ = token_embds.size()
        num_genes, dim_graph = gene_embds.size()
        assert num_tokens == num_genes, "num_tokens must equal num_genes"

        gene_embds = self.proj(gene_embds)
        gene_embds = gene_embds.unsqueeze(0).expand(batch_size, -1, -1)

        attn_scores = torch.bmm(token_embds, gene_embds.transpose(1, 2))
        attn_scores = torch.softmax(attn_scores, dim=-1)

        g_hat = torch.bmm(attn_scores, token_embds)
        z_g = self.ffn(self.norm(g_hat))
        # z_g = self.norm(g_hat + self.ffn(g_hat))
        final_g = z_g + gene_embds

        combined = token_embds + final_g

        return combined
