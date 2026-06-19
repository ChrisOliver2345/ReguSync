import os
os.environ["OMP_NUM_THREADS"] = "4" 
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import torch
import warnings
import torch.nn.functional as F
import numpy as np
import h5py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def train(model: nn.Module, loader: DataLoader, args, vocab, pad_token, optimizer, scaler, epoch, logger, graph_embd, level=None) -> None:

    model.train()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    total_loss_disc = 0.0
    total_loss_recons = 0.0
    total_loss_reg_spatial = 0.0
    total_error = 0.0

    start_time = time.time()
    num_batches = len(loader)

    for batch_idx, (batch_a, batch_b) in enumerate(loader):

        if not torch.equal(batch_a["pair_labels"], batch_b["pair_labels"]):
            logger.error(f"Batch {batch_idx}: pair_labels mismatch! ")
            raise ValueError(f"Batch {batch_idx}: pair_labels mismatch between batch_a and batch_b")

        gene_ids, values, celltype_labels, value_basic_a, value_basic_b, value_true_a, value_true_b, values_grn, locs_a, locs_b = (
            get_batch_data(batch_a, batch_b, graph_embd, args))

        src_key_padding_mask = gene_ids.eq(vocab[pad_token])

        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            # 训练模型
            output_dict = model(
                src=gene_ids,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
                basic_a=value_basic_a,
                basic_b=value_basic_b,
                values_grn=values_grn,
                do_train=True,
                level=level
            )

            loss_total, loss_reg, loss_disc, loss_cls, loss_recons, loss_reg_spatial = model.compute_loss(output_dict=output_dict,
                                                                                        celltype_labels=celltype_labels,
                                                                                        true_value_a=value_true_a,
                                                                                        true_value_b=value_true_b,
                                                                                        locs_a=locs_a, 
                                                                                        locs_b=locs_b
                                                                                        )

            loss = 0.0
            metrics_to_log = {}

            loss = loss + loss_total

            metrics_to_log.update({"train/cls": loss_cls.item()})

            # error_rate = 1 - (
            #     (output_dict["cls_output"].argmax(1) == celltype_labels).sum().item()
            # ) / celltype_labels.size(0)
            error_rate = 0

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                print(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()
        total_loss_disc += loss_disc.item()
        total_loss_recons += loss_recons.item()
        total_loss_reg_spatial += loss_reg_spatial.item()
        total_error += error_rate

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            # cur_loss_cls = total_loss_cls / log_interval
            # cur_error = total_error / log_interval
            # cur_loss_reg = total_loss_reg / log_interval
            cur_loss_disc = total_loss_disc / args.log_interval
            cur_loss_recons = total_loss_recons / args.log_interval
            cur_loss_reg_spatial = total_loss_reg_spatial / args.log_interval

            logger.info(
                f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                f"ms/batch {ms_per_batch:5.2f} | "
                f"loss-all {cur_loss:5.3f} | "
                f"loss-D {cur_loss_disc:5.3f} | "
                f"loss-Rc {cur_loss_recons:5.3f} | "
                f"loss-Rs {cur_loss_reg_spatial:5.3f} | "
            )

            total_loss = 0.0
            total_loss_cls = 0.0
            total_loss_reg = 0.0
            total_loss_disc = 0.0
            total_loss_recons = 0.0
            total_error = 0
            start_time = time.time()


def evaluate(model: nn.Module, loader: DataLoader, args, vocab, pad_token, epoch, logger, graph_embd,
             output_fnames_a, output_fnames_b, test_cell_names_a, test_cell_names_b, fold, save_path=None, 
            level=None, svg_df=None):
    
    model.eval()
    total_loss_all = 0.0
    total_error = 0.0
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    total_loss_disc = 0.0
    total_loss_recons = 0.0

    true_matrix_a = []
    true_matrix_b = []
    pred_matrix_a = []
    pred_matrix_b = []
    true_embeddings_a = []
    true_embeddings_b = []
    pred_embeddings_a = []
    pred_embeddings_b = []

    true_labels_a = []
    true_labels_b = []
    pred_labels_a = []
    pred_labels_b = []

    total_num = 0

    with torch.no_grad():

        for (batch_a, batch_b) in loader:

            gene_ids, values, celltype_labels, value_basic_a, value_basic_b, value_true_a, value_true_b, values_grn, locs_a, locs_b = (
                get_batch_data(batch_a, batch_b, graph_embd, args))

            src_key_padding_mask = gene_ids.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=args.enable_amp):

                # 测试模型
                output_dict = model(
                    src=gene_ids,
                    values=values,
                    src_key_padding_mask=src_key_padding_mask,
                    basic_a=value_basic_a,
                    basic_b=value_basic_b,
                    values_grn=values_grn,
                    do_train=False,
                    level=level
                )

                loss_total, loss_reg, loss_disc, loss_cls, loss_recons, loss_reg_spatial = model.compute_loss(output_dict=output_dict,
                                                                                                            celltype_labels=celltype_labels,
                                                                                                            true_value_a=value_true_a,
                                                                                                            true_value_b=value_true_b,
                                                                                                            locs_a=locs_a,
                                                                                                            locs_b=locs_b
                                                                                                            )


            total_loss_all += loss_total.item() * len(gene_ids)
            total_loss_cls += loss_cls.item() * len(gene_ids)
            total_loss_reg += loss_reg.item() * len(gene_ids)
            total_loss_disc += loss_disc.item() * len(gene_ids)
            total_loss_recons += loss_recons.item() * len(gene_ids)

            # accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            accuracy = 0
            total_error += (1 - accuracy / len(gene_ids)) * len(gene_ids)
            total_num += len(gene_ids)
            
            if save_path is not None:
                true_mt_a = value_true_a.cpu().numpy()
                true_matrix_a.append(true_mt_a)

                true_mt_b = value_true_b.cpu().numpy()
                true_matrix_b.append(true_mt_b)

                pred_mt_a = output_dict["recons_b2a"].cpu().numpy()
                pred_matrix_a.append(pred_mt_a)

                pred_mt_b = output_dict["recons_a2b"].cpu().numpy()
                pred_matrix_b.append(pred_mt_b)

                true_embd_a = output_dict["cells_embd_a"].cpu().numpy()
                true_embeddings_a.append(true_embd_a)

                true_embd_b = output_dict["cells_embd_b"].cpu().numpy()
                true_embeddings_b.append(true_embd_b)

                pred_embd_a = output_dict["cells_t_embd_a"].cpu().numpy()
                pred_embeddings_a.append(pred_embd_a)

                pred_embd_b = output_dict["cells_t_embd_b"].cpu().numpy()
                pred_embeddings_b.append(pred_embd_b)

                true_labels_a.append(batch_a["celltype_labels"].cpu().numpy())
                true_labels_b.append(batch_b["celltype_labels"].cpu().numpy())

                # half_batch_size = len(gene_ids) // 2
                # pred_labels_a.append(output_values[:half_batch_size].argmax(1).cpu().numpy())
                # pred_labels_b.append(output_values[half_batch_size:].argmax(1).cpu().numpy())

        
        pred_matrix_a = np.concatenate(pred_matrix_a, axis=0)
        pred_matrix_b = np.concatenate(pred_matrix_b, axis=0)
        true_embeddings_a = np.concatenate(true_embeddings_a, axis=0)
        pred_embeddings_a = np.concatenate(pred_embeddings_a, axis=0)
        true_embeddings_b = np.concatenate(true_embeddings_b, axis=0)
        pred_embeddings_b = np.concatenate(pred_embeddings_b, axis=0)

        if args.evaluation_a == 'NMI' and args.evaluation_b == 'NMI':
            true_labels_a = np.concatenate(true_labels_a, axis=0)
            true_labels_b = np.concatenate(true_labels_b, axis=0)

            n_clusters = len(np.unique(true_labels_a))

            pca_a = PCA(n_components=50)
            reduced_a = pca_a.fit_transform(pred_matrix_a)
            # reduced_a = true_embeddings_a
            kmeans_a = KMeans(n_clusters=n_clusters, n_init=10)
            pred_clusters_a = kmeans_a.fit_predict(reduced_a)
            nmi_a = normalized_mutual_info_score(true_labels_a, pred_clusters_a)

            if args.modal_b == 'ATAC':
                pca_b = PCA(n_components=50)
            elif args.modal_b == 'ADT':
                pca_b = PCA(n_components=10)
            reduced_b = pca_b.fit_transform(pred_matrix_b)
            # reduced_b = true_embeddings_b
            kmeans_b = KMeans(n_clusters=n_clusters, n_init=10)
            pred_clusters_b = kmeans_b.fit_predict(reduced_b)
            nmi_b = normalized_mutual_info_score(true_labels_b, pred_clusters_b)
            logger.info(f"  epoch {epoch} - NMI for modal A: {nmi_a}, for modal B: {nmi_b}")
            
            
            if save_path is not None:
                true_matrix_a = np.concatenate(true_matrix_a, axis=0)
                # pred_matrix_a = np.concatenate(pred_matrix_a, axis=0)

                with h5py.File(save_path / f'{fold}_true_matrix_a.h5', 'w') as f:
                    f.create_dataset('true_matrix_a', data=true_matrix_a)
                with h5py.File(save_path / f'{fold}_pred_matrix_a.h5', 'w') as f:
                    f.create_dataset('pred_matrix_a', data=pred_matrix_a)
                with h5py.File(save_path / f'{fold}_true_embeddings_a.h5', 'w') as f:
                    f.create_dataset('true_embeddings_a', data=true_embeddings_a)
                with h5py.File(save_path / f'{fold}_pred_embeddings_a.h5', 'w') as f:
                    f.create_dataset('pred_embeddings_a', data=pred_embeddings_a)      

                pd.DataFrame(test_cell_names_a, columns=['cell_names']).to_csv(save_path / f'{fold}_test_cell_names_a.csv', index=False)
                pd.DataFrame(output_fnames_a, columns=['feature_names']).to_csv(save_path / f'{fold}_output_fnames_a.csv', index=False)


            
            if save_path is not None:
                true_matrix_b = np.concatenate(true_matrix_b, axis=0)
                # pred_matrix_b = np.concatenate(pred_matrix_b, axis=0)
            
                with h5py.File(save_path / f'{fold}_true_matrix_b.h5', 'w') as f:
                    f.create_dataset('true_matrix_b', data=true_matrix_b)
                with h5py.File(save_path / f'{fold}_pred_matrix_b.h5', 'w') as f:
                    f.create_dataset('pred_matrix_b', data=pred_matrix_b)
                with h5py.File(save_path / f'{fold}_true_embeddings_b.h5', 'w') as f:
                    f.create_dataset('true_embeddings_b', data=true_embeddings_b)
                with h5py.File(save_path / f'{fold}_pred_embeddings_b.h5', 'w') as f:
                    f.create_dataset('pred_embeddings_b', data=pred_embeddings_b)

                pd.DataFrame(test_cell_names_b, columns=['cell_names']).to_csv(save_path / f'{fold}_test_cell_names_b.csv', index=False)
                pd.DataFrame(output_fnames_b, columns=['feature_names']).to_csv(save_path / f'{fold}_output_fnames_b.csv', index=False)
        
            best_result_a = nmi_a
            best_result_b = nmi_b

        total_loss_all = total_loss_all / total_num
        total_loss_cls = total_loss_cls / total_num
        total_loss_reg = total_loss_reg / total_num
        total_loss_disc = total_loss_disc / total_num
        total_loss_recons = total_loss_recons / total_num
        total_error = total_error / total_num
        total_acc = 1.0 - total_error

        logger.info(
            {
                "test/loss-all": f"{total_loss_all:.3f}",
                "test/loss-reg": f"{total_loss_reg:.3f}",
                "test/loss-disc": f"{total_loss_disc:.3f}",
                "test/loss-recons": f"{total_loss_recons:.3f}",
                "epoch": epoch,
            }
        )

    return total_loss_all, best_result_a, best_result_b


def get_batch_data(batch_a, batch_b, graph_embd, args):
    batch_data = {
        "gene_ids": torch.cat([batch_a["gene_ids"], batch_b["gene_ids"]], dim=0),
        "values": torch.cat([batch_a["values"], batch_b["values"]], dim=0),
        "celltype_labels": torch.cat([batch_a["celltype_labels"], batch_b["celltype_labels"]], dim=0),
        "value_basic_a": batch_a["value_basic"],
        "value_basic_b": batch_b["value_basic"],
        "value_true_a": batch_a["value_true"],
        "value_true_b": batch_b["value_true"],
        "locs_a": batch_a.get("locs") if args.spatial == True else None,
        "locs_b": batch_b.get("locs") if args.spatial == True else None
    }

    gene_ids = batch_data["gene_ids"].to(args.device)
    values = batch_data["values"].to(args.device)
    celltype_labels = batch_data["celltype_labels"].to(args.device)
    value_basic_a = batch_data["value_basic_a"].to(args.device)
    value_basic_b = batch_data["value_basic_b"].to(args.device)
    value_true_a = batch_data["value_true_a"].to(args.device)
    value_true_b = batch_data["value_true_b"].to(args.device)
    values_grn = graph_embd['shared_grn'].to(args.device)
    locs_a = batch_data["locs_a"].to(args.device) if args.spatial == True else None
    locs_b = batch_data["locs_b"].to(args.device) if args.spatial == True else None

    return gene_ids, values, celltype_labels, value_basic_a, value_basic_b, value_true_a, value_true_b, values_grn, locs_a, locs_b

