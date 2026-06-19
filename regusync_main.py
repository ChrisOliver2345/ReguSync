import os
os.environ["OMP_NUM_THREADS"] = "8"
import torch
import argparse
from tqdm import tqdm
from torch import optim
from datasets import load_data, preprocess_data, prepare_data, prepare_dataloader, get_grn_embeddings, get_grn_embeddings_2
from sklearn.model_selection import KFold, StratifiedKFold
from utils import set_seed, sc_logger
from torchtext.vocab import Vocab
from torchtext._torchtext import (Vocab as VocabPybind,)
import numpy as np
from tokenizer import tokenize_and_pad_batch
import time
from model import ReguSync_Transformer
import torch.nn as nn
from train import train, evaluate
from pathlib import Path
import pandas as pd
import gc
import sys
import random

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--modal_a', type=str, default='RNA')
    parser.add_argument('--modal_b', type=str, default='ATAC')
    parser.add_argument('--species', type=str)
    parser.add_argument('--modal_a_train', type=str)
    parser.add_argument('--modal_b_train', type=str)
    parser.add_argument('--modal_a_test', type=str)
    parser.add_argument('--modal_b_test', type=str)
    parser.add_argument('--modal_a_loss', type=str, default='nb')
    parser.add_argument('--modal_b_loss', type=str, default='nb')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument("--n_bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument('--max_seq_len', type=int, default=2000)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument('--GRNs', type=str, default="STRINGdb")
    parser.add_argument('--d_graph', type=int, default=64)
    parser.add_argument('--n_hvg', type=int, default=2000)
    parser.add_argument('--hvg_flavor', type=str, default='seurat_v3')
    parser.add_argument('--hvg_flavor_2', type=str, default='cell_ranger')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--evaluation_a', type=str, default='NMI')
    parser.add_argument('--evaluation_b', type=str, default='NMI')
    
    parser.add_argument('--include_zero_gene', default=True)
    parser.add_argument('--enable_amp', default=True)
    parser.add_argument('--save_embds', default=True)
    parser.add_argument('--ram_usage_optimization', default=False)
    parser.add_argument('--spatial', default=False)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--have_labels', default=True)
    parser.add_argument('--ADT_preprocess_method', type=str, default='standard')
    parser.add_argument('--modal_a_loss_lambda', type=float, default=None)
    parser.add_argument('--modal_b_loss_lambda', type=float, default=None)

    return parser



def build_args(**kwargs):

    parser = get_parser()

    args = parser.parse_args([])

    for key, value in kwargs.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown argument: {key}")
        setattr(args, key, value)

    required_args = [
        'dataset',
        'species',
        'modal_a_train',
        'modal_b_train',
        'modal_a_test',
        'modal_b_test',
    ]

    missing_args = [
        name for name in required_args
        if getattr(args, name) is None
    ]

    if missing_args:
        raise ValueError(
            "Missing required arguments for function-based execution: "
            + ", ".join(missing_args)
        )

    return args


def run_ReguSync(**kwargs):

    args = build_args(**kwargs)

    dataset_name = args.dataset
    save_dir = Path(f"./Results/dev_{dataset_name}/{time.strftime('%Y-%m-%d_%H-%M-%S')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = sc_logger(save_path=save_dir / "run.log")
    logger.info(vars(args))

    set_seed(args.seed)

    cache_dir = Path(f"./Cache/{args.dataset}")
    os.makedirs(cache_dir, exist_ok=True)

    modal_a_adata, modal_b_adata, gene_sequence_df = load_data(args)

    adata_a, adata_b, basic_a, basic_b, svg_df = preprocess_data(modal_a_adata, modal_b_adata, args, logger)


    if args.GRNs == "STRINGdb":
        GRN_embds_path = cache_dir / args.GRNs / f"pretrain_gene_embeddings.csv"
        GRN_embds = pd.read_csv(GRN_embds_path, index_col=0, header=0)
        GRN_genes = GRN_embds.index.tolist()
        GRN_genes_set = set(GRN_genes)

        common_genes = adata_a.var_names.intersection(GRN_genes_set)
        if len(common_genes) == 0:
            raise ValueError("No overlapping genes found between adata_a and STRINGdb GRN embeddings.")
        
        adata_a = adata_a[:, common_genes].copy()
        adata_b = adata_b[:, common_genes].copy()
    
    unorder_genes = adata_a.var_names.tolist()
    genes_set = set(unorder_genes)
    gene_sequence = gene_sequence_df['gene'].tolist()
    sequence_set = set(gene_sequence)
    missing_genes = genes_set - sequence_set

    if len(genes_set - sequence_set) > 0:
        if len(missing_genes) > 50:
            display_genes = ", ".join(list(missing_genes)[:50]) + ", ..."
        else:
            display_genes = ", ".join(missing_genes)

    ordered_genes = [g for g in gene_sequence if g in genes_set]
    ordered_genes = ordered_genes + sorted(list(missing_genes))
    if len(ordered_genes) != len(unorder_genes):
        logger.warning(f"Final gene list has {len(ordered_genes)} genes, expected {len(unorder_genes)}")
    adata_a = adata_a[:, ordered_genes].copy()
    adata_b = adata_b[:, ordered_genes].copy()
    
    gene_to_level = dict(zip(gene_sequence_df['gene'], gene_sequence_df['level']))
    max_known_level = gene_sequence_df['level'].max()  
    args.max_la = max_known_level + 1  
    downstream_level = args.max_la
            
    levels_list = []
    for g in ordered_genes:  
        if g in gene_to_level:
            levels_list.append(gene_to_level[g])
        else:
            levels_list.append(downstream_level)  

    level_tensor = torch.tensor(levels_list, dtype=torch.long) 
    level_tensor = level_tensor.to(args.device)
    print(f"Level tensor shape: {level_tensor.shape}")

    count_a = adata_a.layers["X_binned"]
    count_b = adata_b.layers["X_binned"]

    count_basic_a = basic_a.layers["X_log1p"]
    if args.modal_a_loss == 'mse':
        true_a = basic_a.layers["X_log1p"]
    elif args.modal_a_loss == 'nb':
        true_a = basic_a.layers["X_raw"]

    if args.modal_b == 'ATAC':
        count_basic_b = basic_b.X
        true_b = basic_b.layers["X_binarize"]
    elif args.modal_b == 'ADT':
        if args.ADT_preprocess_method == 'CLR':
            count_basic_b = basic_b.layers["X_clr"]
            true_b = basic_b.layers["X_clr"]
        elif args.ADT_preprocess_method == 'standard':
            if args.modal_b_loss == 'mse':
                true_b = basic_b.layers["X_scale"]
            elif args.modal_b_loss == 'nb':
                true_b = basic_b.layers["X_log1p"]
            count_basic_b = basic_b.layers["X_scale"]

    feature_names_a = basic_a.var_names.tolist()
    feature_names_b = basic_b.var_names.tolist()

    genes = adata_a.var_names.tolist()
    celltypes_a = adata_a.obs["celltype_id"].to_numpy()
    celltypes_b = adata_b.obs["celltype_id"].to_numpy()

    pair_labels_a = adata_a.obs["pair_labels"].to_numpy()
    pair_labels_b = adata_b.obs["pair_labels"].to_numpy()

    spatial_locs_a = basic_a.obsm['spatial'] if args.spatial == True else None
    
    
    # Construct the train/test split using the predefined split labels loaded from the input AnnData files.
    splits = []

    train_mask_a = adata_a.obs["train_test_split"] == "train"
    test_mask_a = adata_a.obs["train_test_split"] == "test"

    train_mask_b = adata_b.obs["train_test_split"] == "train"
    test_mask_b = adata_b.obs["train_test_split"] == "test"

    # Ensure that both modalities use the same predefined train/test partition.
    if not np.array_equal(train_mask_a.values, train_mask_b.values):
        raise ValueError("The predefined training split is inconsistent between modal_a and modal_b.")

    if not np.array_equal(test_mask_a.values, test_mask_b.values):
        raise ValueError("The predefined test split is inconsistent between modal_a and modal_b.")

    train_idx = np.where(train_mask_a.values)[0]
    test_idx = np.where(test_mask_a.values)[0]

    train_cell_names_a = adata_a.obs_names[train_idx].tolist()
    train_cell_names_b = adata_b.obs_names[train_idx].tolist()
    test_cell_names_a = adata_a.obs_names[test_idx].tolist()
    test_cell_names_b = adata_b.obs_names[test_idx].tolist()

    splits.append({
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_cell_names_a": train_cell_names_a,
        "train_cell_names_b": train_cell_names_b,
        "test_cell_names_a": test_cell_names_a,
        "test_cell_names_b": test_cell_names_b,
    })


   

    pad_token = "<pad>"
    special_tokens = [pad_token]
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    pad_value = args.n_bins
    n_input_bins = args.n_bins + 1

    split = splits[0]

    logger.info(f"  Train cells: {len(split['train_idx'])}")
    logger.info(f"  Test cells: {len(split['test_idx'])}")

    train_idx = split['train_idx']
    test_idx = split['test_idx']
    
    train_cell_names_a = split['train_cell_names_a']
    train_cell_names_b = split['train_cell_names_b']

    test_cell_names_a = split['test_cell_names_a']
    test_cell_names_b = split['test_cell_names_b']

    train_data_a = count_a[train_idx]
    test_data_a = count_a[test_idx]
    train_data_b = count_b[train_idx]
    test_data_b = count_b[test_idx]
    
    train_true_a = true_a[train_idx]
    test_true_a = true_a[test_idx]
    train_true_b = true_b[train_idx]
    test_true_b = true_b[test_idx]
    
    train_basic_a = count_basic_a[train_idx]
    test_basic_a = count_basic_a[test_idx]
    train_basic_b = count_basic_b[train_idx]
    test_basic_b = count_basic_b[test_idx]
    
    train_celltypes_a = celltypes_a[train_idx]
    test_celltypes_a = celltypes_a[test_idx]
    train_celltypes_b = celltypes_b[train_idx]
    test_celltypes_b = celltypes_b[test_idx]
    
    train_pair_labels_a = pair_labels_a[train_idx]
    test_pair_labels_a = pair_labels_a[test_idx]
    train_pair_labels_b = pair_labels_b[train_idx]
    test_pair_labels_b = pair_labels_b[test_idx]
    
    train_spatial_locs = spatial_locs_a[train_idx] if args.spatial == True else None
    test_spatial_locs = spatial_locs_a[test_idx] if args.spatial == True else None

    gc.collect()

    tokenized_train_a = tokenize_and_pad_batch(
        train_data_a,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=False,
        include_zero_gene=args.include_zero_gene,
    )

    args.max_seq_len = tokenized_train_a['values'].shape[1]
    print(f"Updated args.max_seq_len to: {args.max_seq_len}")

    tokenized_train_b = tokenize_and_pad_batch(
        train_data_b,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=False,
        include_zero_gene=args.include_zero_gene,
    )

    tokenized_test_a = tokenize_and_pad_batch(
        test_data_a,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=False,
        include_zero_gene=args.include_zero_gene,
    )

    tokenized_test_b = tokenize_and_pad_batch(
        test_data_b,
        gene_ids,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=False,
        include_zero_gene=args.include_zero_gene,
    )

    gc.collect() 

    GRN_embds_path = cache_dir / args.GRNs / "pretrain_gene_embeddings.csv"
    GRN_embds = pd.read_csv(GRN_embds_path, index_col=0, header=0)
    GRN_embds = GRN_embds[GRN_embds.index.isin(genes)]
    
    genes_set = set(genes)
    grn_genes_set = set(GRN_embds.index)
    missing_genes = genes_set - grn_genes_set

    GRN_indices = np.array(GRN_embds.index)
    GRN_embds = GRN_embds.to_numpy()

    GRN_embds_train_a = get_grn_embeddings_2(tokenized_train_a, GRN_embds, GRN_indices, vocab, pad_token, cls_token="<cls>", dtype=torch.float32)
    GRN_embds_test_a = GRN_embds_train_a
    
    del GRN_embds, GRN_indices
    gc.collect()
    
    train_grn = {
        "shared_grn": GRN_embds_train_a[0],
    }

    test_grn = {
        "shared_grn": GRN_embds_test_a[0],
    }

    model = ReguSync_Transformer(
        args=args,
        ntoken=len(vocab),
        d_model=args.d_model,
        nhead=args.n_heads,
        d_hid=args.d_ff,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
        nlayers=args.n_layers,
        n_features_a=train_basic_a.shape[1],
        n_features_b=train_basic_b.shape[1]
    )

    model.to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-4 if args.enable_amp else 1e-8
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    train_data_pt, test_data_pt = prepare_data(tokenized_train=(tokenized_train_a, tokenized_train_b),
                                                tokenized_test=(tokenized_test_a, tokenized_test_b),
                                                train_celltype_labels=(train_celltypes_a, train_celltypes_b),
                                                test_celltype_labels=(test_celltypes_a, test_celltypes_b),
                                                train_pair_labels=(train_pair_labels_a, train_pair_labels_b),
                                                test_pair_labels=(test_pair_labels_a, test_pair_labels_b),
                                                train_basic=(train_basic_a, train_basic_b),
                                                test_basic=(test_basic_a, test_basic_b),
                                                train_true=(train_true_a, train_true_b),
                                                test_true=(test_true_a, test_true_b),
                                                train_spatial_locs=train_spatial_locs if args.spatial == True else None,
                                                test_spatial_locs=test_spatial_locs if args.spatial == True else None,
                                                args=args
                                                )

    
    del tokenized_train_a, tokenized_train_b, tokenized_test_a, tokenized_test_b
    gc.collect()

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=args.train_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
        pin_memory=False,
        args=args
    )

    test_loader = prepare_dataloader(
        test_data_pt,
        batch_size=args.test_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
        pin_memory=False,
        args=args
    )

    
    del train_data_pt, test_data_pt
    gc.collect()

    best_loss = float('inf')
    counter = 0
    best_model_path = save_dir / f'best_model.pth'

    train_losses = []
    test_losses = []

    if args.evaluation_a in ['NMI', 'Pearson', 'Pearson-SVG']:
        best_result_a = 0
    if args.evaluation_b in ['NMI', 'Pearson', 'Pearson-SVG']:
        best_result_b = 0
    if args.evaluation_a in ['MAE', 'MSE']:
        best_result_a = float('inf')
    if args.evaluation_b in ['MAE', 'MSE']:
        best_result_b = float('inf')

    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()

        train(
            model=model,
            loader=train_loader,
            args=args,
            vocab=vocab,
            pad_token=pad_token,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            logger=logger,
            graph_embd=train_grn,
            level=level_tensor,
        )

        save_path = save_dir if args.save_embds is True else None

        train_losses.append(0)

    test_loss, test_result_a, test_result_b = evaluate(
        model=model,
        loader=test_loader,
        vocab=vocab,
        pad_token=pad_token,
        epoch=epoch,
        args=args,
        logger=logger,
        save_path=save_path,
        graph_embd=test_grn,
        output_fnames_a=feature_names_a,
        output_fnames_b=feature_names_b,
        test_cell_names_a=test_cell_names_a,
        test_cell_names_b=test_cell_names_b,
        fold=0,
        best_result_a=best_result_a,
        best_result_b=best_result_b,
        level=level_tensor,
        svg_df=svg_df if args.evaluation_a == 'Pearson-SVG' else None,
    )

    logger.info("-" * 80)
    logger.info(f"| test loss {test_loss:5.4f} | ")
    logger.info("-" * 80)


       
        
