import argparse
import sys
import torch
from pathlib import Path
from utils import set_seed, sc_logger
import numpy as np
import pandas as pd
import anndata as ad
import time
import re
import muon as mu
from utils import get_RP_score
import os
from pretrain_GAS import calculate_gas
import h5py
from collections import defaultdict
from pretrain_GNN import GAT_model
from pretrain_utils import norm_data
from torch.utils.data import Dataset
import scipy.sparse as sp
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scanpy as sc
from anndata import AnnData
from mousipy import translate
import pickle
import gc

from scipy.sparse import hstack, issparse


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--modal_a', type=str, default='RNA')
    parser.add_argument('--modal_b', type=str, default='ATAC')
    parser.add_argument('--modal_a_file', type=str)
    parser.add_argument('--modal_b_file', type=str)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--GRNs', type=str, default="STRINGdb")
    parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
    parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--species', type=str)
    parser.add_argument('--hvg_num', type=int, default=-1)

    args = parser.parse_args()

    return args


def run():
    args = prepare()

    dataset_name = args.dataset
    save_dir = Path(f"./SAVE/pretrain/dev_pretrain_{dataset_name}/{time.strftime('%Y-%m-%d_%H-%M-%S')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = sc_logger(save_path=save_dir / "run.log")
    logger.info(vars(args))

    set_seed(args.seed)

    if args.GRNs == "STRINGdb":
        edges, graph_genes = prepare_STRINGbd_graph(logger=logger, species=args.species)

    modal_a_adata, modal_b_adata = load_data(args)

    cache_dir = f"./TEMP/{args.dataset}"
    os.makedirs(cache_dir, exist_ok=True)

    if args.modal_b == 'ATAC':

        adata_a, adata_rp, adata_b = preprocess_data(modal_a_adata, modal_b_adata, args, logger)
        cache_file = os.path.join(cache_dir, f"pretrain_gas.h5")

        if os.path.exists(cache_file):
            logger.info(f"Loading cached gas matrix from {cache_file}")
            gas_df = pd.read_hdf(cache_file, key='gas')
        else:
            logger.info(f"Computing gas matrix for {args.dataset}")
            gas_df, _ = calculate_gas(adata_rp=adata_rp, adata_rna=adata_a, adata_atac=adata_b, method="wnn", return_weight=False)
            gas_df.to_hdf(cache_file, key='gas', mode='w')
            logger.info(f"Saved gas matrix to {cache_file}")

        logger.info(f"GAS genes: {gas_df.shape[0]}")
        logger.info(f"GAS cells: {gas_df.shape[1]}")

    elif args.modal_b == 'ADT':
        adata_a, adata_b, _ = preprocess_data(modal_a_adata, modal_b_adata, args, logger)

        sc.pp.normalize_total(adata_a, target_sum=1e4)
        sc.pp.log1p(adata_a)

        def clr_normalize_each_cell(adata, inplace=True):
            """Normalize count vector for each cell, i.e. for each row of .X"""

            def seurat_clr(x):
                # TODO: support sparseness
                s = np.sum(np.log1p(x[x > 0]))
                exp = np.exp(s / len(x))
                return np.log1p(x / exp)

            if not inplace:
                adata = adata.copy()

            adata.layers['X_raw'] = adata.X.copy() if sp.issparse(adata.X) else np.copy(adata.X)

            # apply to dense or sparse matrix, along axis. returns dense matrix
            clr_result = np.apply_along_axis(
                seurat_clr, 1, (adata.X.A if sp.issparse(adata.X) else adata.X)
            )
            # .A 是 scipy.sparse 矩阵的一个属性，用于将稀疏矩阵转换为密集矩阵

            adata.X = clr_result
            adata.layers['X_clr'] = clr_result

            return adata

        adata_b = clr_normalize_each_cell(adata_b)

        mdata = mu.MuData({'rna': adata_a, 'adt': adata_b})
        sc.pp.highly_variable_genes(mdata['rna'], n_top_genes=2000, flavor='seurat')
        sc.pp.scale(mdata['rna'])
        sc.tl.pca(mdata['rna'], n_comps=30)

        sc.pp.scale(mdata['adt'])
        if args.dataset in ['SPOTS_Cross_Tissue', 'Cross_Protocols_3']:
            sc.tl.pca(mdata['adt'], n_comps=5)
        else:
            sc.tl.pca(mdata['adt'], n_comps=10)

        print("正在计算邻居")
        sc.pp.neighbors(mdata['rna'], use_rep='X_pca')
        sc.pp.neighbors(mdata['adt'], use_rep='X_pca')

        print("正在计算 WNN")
        mu.pp.neighbors(mdata, key_added='wnn')
        print("WNN 计算完成")

        rna_weights = mdata.obs['rna:mod_weight'].values
        adt_weights = mdata.obs['adt:mod_weight'].values

        rna_data = adata_a.X.toarray() if sp.issparse(adata_a.X) else adata_a.X
        adt_data = adata_b.X if not sp.issparse(adata_b.X) else adata_b.X.toarray()  # CLR 后已经是 dense

        weighted_rna = rna_data * rna_weights[:, np.newaxis]
        weighted_adt = adt_data * adt_weights[:, np.newaxis]

        rna_features = adata_a.var_names
        adt_features = adata_b.var_names

        common_features = np.intersect1d(rna_features, adt_features)
        rna_only = np.setdiff1d(rna_features, common_features)
        adt_only = np.setdiff1d(adt_features, common_features)

        rna_idx = pd.Index(rna_features)
        adt_idx = pd.Index(adt_features)
        
        common_rna_cols = rna_idx.get_indexer(common_features)
        common_adt_cols = adt_idx.get_indexer(common_features)
        rna_only_cols = rna_idx.get_indexer(rna_only)
        adt_only_cols = adt_idx.get_indexer(adt_only)

        weighted_rna_only = weighted_rna[:, rna_only_cols] if len(rna_only) > 0 else np.empty((weighted_rna.shape[0], 0))
        weighted_adt_only = weighted_adt[:, adt_only_cols] if len(adt_only) > 0 else np.empty((weighted_adt.shape[0], 0))

        if len(common_features) > 0:
            weighted_rna_common = weighted_rna[:, common_rna_cols]
            weighted_adt_common = weighted_adt[:, common_adt_cols]
            df_common = pd.DataFrame(
                weighted_rna_common + weighted_adt_common,
                columns=common_features,
                index=adata_a.obs_names
            )
        else:
            df_common = pd.DataFrame()

        df_rna_only = pd.DataFrame(
        weighted_rna_only,
        columns=rna_only,
        index=adata_a.obs_names
        )
        df_adt_only = pd.DataFrame(
            weighted_adt_only,
            columns=adt_only,
            index=adata_a.obs_names
        )
        
        # 拼接融合 DataFrame (cells x total_features)
        df_combined = pd.concat([df_rna_only, df_common, df_adt_only], axis=1)
        
        # 创建 gas_df (features x cells)
        gas_df = df_combined.T

    gas_genes = set(gas_df.index)

    filtered_graph_genes = graph_genes.intersection(gas_genes)
    edges = [(gene1, gene2) for gene1, gene2 in edges if gene1 in filtered_graph_genes and gene2 in filtered_graph_genes]
    all_genes = gas_genes

    gene_to_idx = {gene: idx for idx, gene in enumerate(sorted(all_genes))}

    tf_data = []
    target_data = []
    for gene1, gene2 in edges:
        tf_data.append((gene1, gene_to_idx[gene1]))
        target_data.append((gene2, gene_to_idx[gene2]))

    tf_df = pd.DataFrame(tf_data, columns=['Gene', 'index'])
    target_df = pd.DataFrame(target_data, columns=['Gene', 'index'])
    tf_df = tf_df.drop_duplicates(subset=['Gene', 'index']).reset_index(drop=True)
    target_df = target_df.drop_duplicates(subset=['Gene', 'index']).reset_index(drop=True)

    logger.info(f"TF DataFrame shape: {tf_df.shape}")
    logger.info(f"Target DataFrame shape: {target_df.shape}")

    tf_targets = defaultdict(set)
    for tf, target in edges:
        tf_targets[tf].add(target)

    np.random.shuffle(edges)
    train_size = int(0.8 * len(edges))
    logger.info(f"Train size: {train_size}")
    train_edges = edges[:train_size]
    val_edges = edges[train_size:]

    def generate_df(edges_part, tf_targets, all_genes):
        # 先获取唯一 TF
        unique_tfs = set(tf for tf, _ in edges_part)

        pos_data = []
        neg_data = []
        for tf in unique_tfs:
            # 正样本：从该 TF 的 targets 中随机选一个
            tf_specific_targets = [target for t, target in edges_part if t == tf]
            if tf_specific_targets:
                target = np.random.choice(tf_specific_targets)
                pos_data.append((gene_to_idx[tf], gene_to_idx[target], 1))

            # 负样本：从非 targets 中随机选一个
            possible_neg = list(all_genes - tf_targets[tf])
            if possible_neg:
                gene_c = np.random.choice(possible_neg)
                neg_data.append((gene_to_idx[tf], gene_to_idx[gene_c], 0))

        # 合并正负
        all_data = pos_data + neg_data
        df = pd.DataFrame(all_data, columns=['TF', 'Target', 'Label'])
        df = df.sample(frac=1).reset_index(drop=True)  # 随机打乱
        return df
   
    logger.info("Generating train_data")
    train_data = generate_df(train_edges, tf_targets, all_genes)
   
    logger.info("Generating validation_data")
    validation_data = generate_df(val_edges, tf_targets, all_genes)

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {validation_data.shape}")

    splits = []
    for fold in range(1, args.folds + 1):
        pair_labels_file = os.path.join(cache_dir, f"Total_{args.folds}_fold_{fold}_test_cells.csv")
        pair_labels_df = pd.read_csv(pair_labels_file)
        test_cell_names = pair_labels_df['cell_name_a'].values
        # test_pair_labels = pair_labels_df['pair_label'].values

        all_cell_names = gas_df.columns
        if args.dataset == 'P0_BrainCortex_SNAREseq_GSE126074_sub':
            train_cell_names = test_cell_names
        else:
            train_cell_names = np.setdiff1d(all_cell_names, test_cell_names)

        gas_df_split = gas_df[train_cell_names]
        splits.append(gas_df_split)

    tf = tf_df['index'].values.astype(np.int64)
    target = target_df['index'].values.astype(np.int64)
    tf = torch.from_numpy(tf)
    tf = tf.to(args.device).long()

    train_data = train_data.values
    validation_data = validation_data.values

    for i, split in enumerate(splits):

        logger.info(f"Fold {i + 1}:")
        logger.info(f"  Total genes: {split.shape[0]}")
        logger.info(f"  Total cells: {split.shape[1]}")

        data_input = split
        loader = norm_data(data_input, normalize=False, pca_dims=50)
        feature = loader.process_data()

        feature = torch.from_numpy(feature)
        data_feature = feature.to(args.device).float()

        # train_data_pt = torch.from_numpy(train_data).long()
        # val_data_pt = torch.from_numpy(validation_data).long()
        # train_data_pt = train_data_pt.to(args.device)
        # validation_data_pt = val_data_pt.to(args.device)

        train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
        adj = train_load.Adj_Generate(tf, loop=True)
        adj = adj2saprse_tensor(adj)

        model = GAT_model(input_dim=feature.size()[1],
                          hidden1_dim=256,
                          hidden2_dim=args.output_dim,
                          hidden3_dim=32,
                          output_dim=16,
                          num_head1=3,
                          num_head2=3,
                          alpha=0.2,
                          device=args.device,
                          type='dot',
                          reduction='concate'
                          )

        adj = adj.to(args.device)
        model = model.to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

        for epoch in range(args.epochs):
            running_loss = 0.0

            for train_x, train_y in DataLoader(train_load, batch_size=64, shuffle=True):
                model.train()
                optimizer.zero_grad()

                if args.flag:
                    train_y = train_y.to(args.device)
                else:
                    train_y = train_y.to(args.device).view(-1, 1)

                # train_y = train_y.to(device).view(-1, 1)
                pred, embd = model(data_feature, adj, train_x)

                # pred = torch.sigmoid(pred)
                if args.flag:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)
                loss_BCE = F.binary_cross_entropy(pred, train_y)

                loss_BCE.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss_BCE.item()


            # model.eval()
            # score = model(data_feature, adj, validation_data)
            # if args.flag:
            #     score = torch.softmax(score, dim=1)
            # else:
            #     score = torch.sigmoid(score)

            print('Epoch:{}'.format(epoch + 1),
                  'train loss:{}'.format(running_loss))

            if epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    embd = model.encode(data_feature, adj)
                    embd_np = embd.cpu().detach().numpy()
                    embd_df = pd.DataFrame(embd_np, index=sorted(all_genes))
                    save_dir = os.path.join(cache_dir, "STRINGdb")
                    os.makedirs(save_dir, exist_ok=True)
                    embd_df.to_csv(os.path.join(save_dir, f"pretrain_total_{args.folds}_fold_{i + 1}_gene_embeddings.csv"), index=True)


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor


def prepare_STRINGbd_graph(logger, species):

    if species == 'human':
        # r"E:\TotalWorkspace\ShareData_omics\Dataset_STRING\Homo_sapiens\9606.protein.links.high_confidence_700.h5"
        gene_link_path = r"E:\TotalWorkspace\ShareData_omics\Dataset_STRING\Homo_sapiens\9606.protein.links.all_scores.h5"
    elif species == 'mouse':
        gene_link_path = r"E:\TotalWorkspace\ShareData_omics\Dataset_STRING\Mus_musculus\10090.protein.links.all_scores.h5"
    else:
        raise ValueError(f"Unsupported species: {species}")


    with h5py.File(gene_link_path, 'r') as h5f:
        gene_link = h5f['data'][:]
    gene_link_data = [(row['gene_name1'].decode('utf-8'),
                    row['gene_name2'].decode('utf-8'),
                    row['combined_score']) for row in gene_link if row['combined_score'] >= 700]

    genes = set()
    for gene1, gene2, _ in gene_link_data:
        genes.add(gene1)
        genes.add(gene2)
    num_genes = len(genes)
    num_edges = len(gene_link_data)
    logger.info(f"Total number of genes (nodes): {num_genes}")
    logger.info(f"Total number of edges: {num_edges}")

    if species == 'human':
        tf_path = r"E:\TotalWorkspace\ShareData_omics\Dataset_BEELINE\BEELINE-Networks\human-tfs.csv"
    elif species == 'mouse':
        tf_path = r"E:\TotalWorkspace\ShareData_omics\Dataset_BEELINE\BEELINE-Networks\mouse-tfs.csv"
    else:
        raise ValueError(f"Unsupported species: {species}")

    tf_pair = pd.read_csv(tf_path, index_col=False, header=0)

    if species == 'mouse':
        protains = tf_pair['TF'].tolist()
        print(f"读取到 {len(protains)} 个蛋白质名称: {protains[:5]}...") 
        n_genes = len(protains)
        n_dummy_cells = 1 
        X = np.zeros((n_dummy_cells, n_genes))  
        adata = AnnData(X=X, var=pd.DataFrame(index=protains))
        translated_adata = translate(adata, target='Symbol', stay_sparse=False, verbose=True)
        mouse_genes = translated_adata.var_names.tolist()
        print(f"翻译后小鼠基因总数: {len(mouse_genes)}")
        print(f"翻译后小鼠基因样本（前5个）: {mouse_genes[:5]}")
        print(f"翻译后小鼠基因样本（后5个）: {mouse_genes[-5:]}")

        pure_numeric_genes = [gene for gene in mouse_genes if str(gene).isdigit()]
        if pure_numeric_genes:
            print(f"发现纯数字基因名称: {pure_numeric_genes}")
        else:
            print("没有发现纯数字基因名称")

        gene_list = mouse_genes

    else:
        human_genes = tf_pair['TF'].tolist()
        gene_list = human_genes
    
    tf_list = set(gene_list)
    print(f'一共获取了{len(tf_list)}个TF')

    directed_edges = []
    for gene1, gene2, _ in gene_link_data:
        # 如果 gene1 是 TF，则添加 TF → gene2（TF 或非 TF）
        if gene1 in tf_list:
            directed_edges.append((gene1, gene2))
        # 如果 gene2 是 TF，则添加 TF → gene1（反向）
        if gene2 in tf_list and gene1 != gene2:  # 避免重复添加自环
            directed_edges.append((gene2, gene1))

    filtered_genes = set()
    for gene1, gene2 in directed_edges:
        filtered_genes.add(gene1)
        filtered_genes.add(gene2)
    num_filtered_genes = len(filtered_genes)
    num_filtered_edges = len(directed_edges)

    logger.info(f"Total number of genes (nodes) after filtering (TF-TF and TF-Gene): {num_filtered_genes}")
    logger.info(f"Total number of directed edges after filtering: {num_filtered_edges}")

    return directed_edges, filtered_genes


def preprocess_data(modal_a_adata, modal_b_adata, args, logger):
    n_cells = modal_a_adata.n_obs
    pair_labels = np.arange(n_cells)
    modal_a_adata.obs["pair_labels"] = pair_labels
    modal_b_adata.obs["pair_labels"] = pair_labels
   
    if args.modal_a == 'RNA':
        adata_a = modal_a_adata
        if args.hvg_num != -1:
            print('filtering and subsetting highly variable genes ...')
            sc.pp.calculate_qc_metrics(adata_a, percent_top=None, log1p=False, inplace=True)
            min_genes_per_cell = 200
            min_cells_per_gene = 5
            sc.pp.filter_cells(adata_a, min_genes=min_genes_per_cell)
            sc.pp.filter_genes(adata_a, min_cells=min_cells_per_gene)
            sc.pp.highly_variable_genes(adata_a, n_top_genes=args.hvg_num, flavor='seurat_v3', subset=True)

    if args.modal_b == 'ATAC':

        cache_dir = f"./TEMP/{args.dataset}"
        os.makedirs(cache_dir, exist_ok=True)
       
        cache_file = os.path.join(cache_dir, f"rp_score.pkl")

        if os.path.exists(cache_file):
            logger.info(f"Loading cached RP score matrix from {cache_file}")
            with open(cache_file, 'rb') as f:
                rp_sparse = pickle.load(f)
                index = pickle.load(f)
                columns = pickle.load(f)
        else:
            if not sp.issparse(modal_b_adata.X):
                sparse_x = sp.csr_matrix(modal_b_adata.X)
            else:
                sparse_x = modal_b_adata.X.copy()
            
            cell_index = modal_b_adata.obs.index
            peak_columns = modal_b_adata.var.index
            
            def is_underscore_format(peak):
                return re.match(r'^chr[\w\d]+_\d+_\d+$', peak) is not None

            if not all(is_underscore_format(col) for col in peak_columns):
                formatted_columns = [peak.replace(":", "_").replace("-", "_") for peak in peak_columns]
                peak_columns = pd.Index(formatted_columns)

            if args.species == 'mouse':
                organism = "GRCm38"
            elif args.species == 'human':
                organism = "GRCh38"
            else:
                raise ValueError(f"Unsupported species: {args.species}")

            logger.info(f"Computing RP score matrix for {args.dataset}")
            atac_data = {'data': sparse_x, 'index': cell_index, 'columns': peak_columns}

            rp_sparse, index, columns = get_RP_score(
                atac_data=atac_data,
                organism=organism,
                decaydistance=10000,
                model="Simple"
            )
            logger.info(f"Computing finished")

            with open(cache_file, 'wb') as f:
                pickle.dump(rp_sparse, f)
                pickle.dump(index, f)
                pickle.dump(columns, f)

            logger.info(f"Saved RP score matrix to {cache_file}")

            gc.collect()

        adata_b = ad.AnnData(
            X=rp_sparse, 
            obs=modal_b_adata.obs.loc[index], 
            var=pd.DataFrame(index=columns)
        )

        common_genes = np.intersect1d(adata_a.var_names, adata_b.var_names)
        adata_a = adata_a[:, common_genes].copy()
        adata_b = adata_b[:, common_genes].copy()

    elif args.modal_b == 'ADT':
        adata_b = modal_b_adata

    if args.hvg_num != -1:
        filtered_cells = adata_a.obs_names
        adata_b = adata_b[filtered_cells, :] 
        modal_b_adata = modal_b_adata[filtered_cells, :]

    gc.collect()

    return adata_a, adata_b, modal_b_adata


class scRNADataset(Dataset):
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:, :2]
        train_label = self.train_set[:, -1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len, 2])
            train_tan[:, 0] = 1 - train_label
            train_tan[:, 1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self, TF_set, direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()

        return adj


def load_data(args):

    modal_a_path = "D:/TotalWorkspace/pycharmProject/1_Multimodal_experiment/Multimodal_dataset/" + args.dataset + "/" + args.modal_a_file
    modal_b_path = "D:/TotalWorkspace/pycharmProject/1_Multimodal_experiment/Multimodal_dataset/" + args.dataset + "/" + args.modal_b_file

    modal_a_adata = sc.read_h5ad(modal_a_path)
    modal_b_adata = sc.read_h5ad(modal_b_path)

    return modal_a_adata, modal_b_adata


if __name__ == '__main__':
    run()
