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
from model import Gene_Transformer
import torch.nn as nn
from train import train, evaluate
from pathlib import Path
import pandas as pd
import gc
import sys
import random

def convert_bool(arg_name, arg_value):
    """验证并转换为 bool；无效值报错"""
    lower_value = arg_value.lower()
    if lower_value == 'true':
        return True
    elif lower_value == 'false':
        return False
    else:
        raise ValueError(f"Invalid value for --{arg_name}: '{arg_value}'. Must be 'True' or 'False' (case-insensitive).")

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--modal_a', type=str, default='RNA')
    parser.add_argument('--modal_b', type=str, default='ATAC')
    parser.add_argument('--species', type=str)
    parser.add_argument('--modal_a_file', type=str)
    parser.add_argument('--modal_b_file', type=str)
    parser.add_argument('--modal_a_loss', type=str, required=True)
    parser.add_argument('--modal_b_loss', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument("--n_bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument('--max_seq_len', type=int, default=2049)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument('--GRNs', type=str, default="STRINGdb")
    parser.add_argument('--d_graph', type=int, default=64)
    parser.add_argument('--n_hvg', type=int, default=2000)
    parser.add_argument('--hvg_flavor', type=str, default='cell_ranger')
    parser.add_argument('--hvg_flavor_2', type=str, default='cell_ranger')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--evaluation_a', type=str, default='NMI')  # NMI or MAE
    parser.add_argument('--evaluation_b', type=str, default='NMI')
    
    parser.add_argument('--include_zero_gene', type=str, default='True')
    parser.add_argument('--enable_amp', type=str, default='True')
    parser.add_argument('--run', type=str, default='False')
    parser.add_argument('--save_embds', type=str, default='False')
    parser.add_argument('--ram_usage_optimization', type=str, default='False')
    parser.add_argument('--grn_la', type=str, default='True')
    parser.add_argument('--spatial', type=str, default='False')
    parser.add_argument('--have_labels', type=str, default='True')
    parser.add_argument('--first_run', type=str, default='False')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--ADT_preprocess_method', type=str, default='standard')
    parser.add_argument('--modal_a_loss_lambda', type=float, default=None)
    parser.add_argument('--modal_b_loss_lambda', type=float, default=None)
    parser.add_argument('--split', type=str, default='5-fold')

    args = parser.parse_args()

    args.include_zero_gene = convert_bool('include_zero_gene', args.include_zero_gene)
    args.enable_amp = convert_bool('enable_amp', args.enable_amp)
    args.run = convert_bool('run', args.run)
    args.save_embds = convert_bool('save_embds', args.save_embds)
    args.ram_usage_optimization = convert_bool('ram_usage_optimization', args.ram_usage_optimization)
    args.grn_la = convert_bool('grn_la', args.grn_la)
    args.spatial = convert_bool('spatial', args.spatial)
    args.have_labels = convert_bool('have_labels', args.have_labels)
    args.first_run = convert_bool('first_run', args.first_run)

    return args


def run():

    #  用cpu跑会报错: assert qkv.dtype in [torch.float16, torch.bfloat16]
    args = prepare()

    if not args.run:
        sys.exit()

    dataset_name = args.dataset
    save_dir = Path(f"./SAVE/dev_{dataset_name}/{time.strftime('%Y-%m-%d_%H-%M-%S')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = sc_logger(save_path=save_dir / "run.log")
    logger.info(vars(args))

    set_seed(args.seed)

    cache_dir = Path(f"./TEMP/{args.dataset}")
    os.makedirs(cache_dir, exist_ok=True)

    modal_a_adata, modal_b_adata, gene_sequence_df = load_data(args)

    adata_a, adata_b, basic_a, basic_b, svg_df = preprocess_data(modal_a_adata, modal_b_adata, args, logger)


    if args.GRNs == "STRINGdb" and args.first_run == False:
        GRN_embds_path = cache_dir / args.GRNs / f"pretrain_total_{args.folds}_fold_1_gene_embeddings.csv"
        GRN_embds = pd.read_csv(GRN_embds_path, index_col=0, header=0)
        GRN_genes = GRN_embds.index.tolist()
        GRN_genes_set = set(GRN_genes)

        logger.info(f"Before filtering: adata_a&b has {adata_a.n_vars} genes, GRN contains {len(GRN_genes_set)} genes.")

        # 筛选 adata_a 和 adata_b，只保留在 GRN_genes_set 中的基因
        common_genes = adata_a.var_names.intersection(GRN_genes_set)
        if len(common_genes) == 0:
            raise ValueError("No overlapping genes found between adata_a and STRINGdb GRN embeddings.")
        
        adata_a = adata_a[:, common_genes].copy()
        adata_b = adata_b[:, common_genes].copy()

        logger.info(f"Filtered genes based on GRN: {len(common_genes)} genes retained.")
    
    unorder_genes = adata_a.var_names.tolist()
    genes_set = set(unorder_genes)
    gene_sequence = gene_sequence_df['gene'].tolist()
    sequence_set = set(gene_sequence)
    missing_genes = genes_set - sequence_set

    logger.info(
        f"Gene sequence matching for adata_a: "
        f"Total genes = {len(genes_set)}, "
        f"Found in gene_sequence = {len(genes_set & sequence_set)}, "
        f"Missing = {len(genes_set - sequence_set)}"
    )

    if len(genes_set - sequence_set) > 0:
        if len(missing_genes) > 50:
            display_genes = ", ".join(list(missing_genes)[:50]) + ", ..."
        else:
            display_genes = ", ".join(missing_genes)
        logger.warning(f"{display_genes} genes are not present in gene_sequence.")

    ordered_genes = [g for g in gene_sequence if g in genes_set]
    ordered_genes = ordered_genes + sorted(list(missing_genes))
    if len(ordered_genes) != len(unorder_genes):
        logger.warning(f"Final gene list has {len(ordered_genes)} genes, expected {len(unorder_genes)}")
    adata_a = adata_a[:, ordered_genes].copy()
    adata_b = adata_b[:, ordered_genes].copy()

    if args.grn_la == True:
        gene_to_level = dict(zip(gene_sequence_df['gene'], gene_sequence_df['level']))
        max_known_level = gene_sequence_df['level'].max()  
        args.max_la = max_known_level + 1  
        downstream_level = args.max_la
               
        levels_list = []
        for g in ordered_genes:   # ordered_genes 已经是 GRN基因 + missing_genes（排在最后）
            if g in gene_to_level:
                levels_list.append(gene_to_level[g])
            else:
                levels_list.append(downstream_level)  

        level_tensor = torch.tensor(levels_list, dtype=torch.long)   # [seq_len]
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
            # true_b = basic_b.layers["X_raw"]
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

    if args.split == '5-fold':
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=25)

        splits = []
        for train_idx, test_idx in kf.split(count_a):
            output_file = cache_dir / f"Total_{args.folds}_fold_{len(splits) + 1}_test_cells.csv"
        
            if not output_file.exists():
                test_cell_names_a = adata_a.obs_names[test_idx]
                test_cell_names_b = adata_b.obs_names[test_idx]

                test_pair_labels_a = pair_labels_a[test_idx]
                test_pair_labels_b = pair_labels_b[test_idx]

                pair_labels_df = pd.DataFrame({
                    'cell_name_a': test_cell_names_a,
                    'pair_label_a': test_pair_labels_a,
                    'cell_name_b': test_cell_names_b,
                    'pair_label_b': test_pair_labels_b
                })
                pair_labels_df.to_csv(output_file, index=False)

            train_cell_names_a = adata_a.obs_names[train_idx].tolist()
            train_cell_names_b = adata_b.obs_names[train_idx].tolist()
            test_cell_names_a = adata_a.obs_names[test_idx].tolist()
            test_cell_names_b = adata_b.obs_names[test_idx].tolist()

            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_cell_names_a': train_cell_names_a,
                'train_cell_names_b': train_cell_names_b,
                'test_cell_names_a': test_cell_names_a,
                'test_cell_names_b': test_cell_names_b,
            })

    elif args.split in ['Xenium_Cross_Tissue', 'Human_A1_Cross_Tissue', 'SPOTS_Cross_Tissue']:
        tissues = adata_a.obs['tissue'].unique()

        splits = []
        for i, test_tissue in enumerate(tissues):
            output_file = cache_dir / f"Total_{args.folds}_fold_{i + 1}_test_cells.csv"
            
            if not output_file.exists():
                # 找出测试组织的掩码
                test_mask = adata_a.obs['tissue'] == test_tissue
                test_idx = np.where(test_mask)[0]
                test_cell_names_a = adata_a.obs_names[test_idx]
                test_cell_names_b = adata_b.obs_names[test_idx]

                test_pair_labels_a = pair_labels_a[test_idx]
                test_pair_labels_b = pair_labels_b[test_idx]

                pair_labels_df = pd.DataFrame({
                    'cell_name_a': test_cell_names_a,
                    'pair_label_a': test_pair_labels_a,
                    'cell_name_b': test_cell_names_b,
                    'pair_label_b': test_pair_labels_b
                })
                pair_labels_df.to_csv(output_file, index=False)

            # 计算 train_idx 和 test_idx
            test_mask = adata_a.obs['tissue'] == test_tissue
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            train_cell_names_a = adata_a.obs_names[train_idx].tolist()
            train_cell_names_b = adata_b.obs_names[train_idx].tolist()
            test_cell_names_a = adata_a.obs_names[test_idx].tolist()
            test_cell_names_b = adata_b.obs_names[test_idx].tolist()

            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_cell_names_a': train_cell_names_a,
                'train_cell_names_b': train_cell_names_b,
                'test_cell_names_a': test_cell_names_a,
                'test_cell_names_b': test_cell_names_b,
            })


    elif args.split in ['Human_Tonsil_Cross_Batch', 'Human_Lymph_Node_Cross_Batch', 'SPOTS_Cross_Batch']:
        batches = adata_a.obs['batch'].unique()

        splits = []
        for i, test_batch in enumerate(batches):
            output_file = cache_dir / f"Total_{args.folds}_fold_{i + 1}_test_cells.csv"
            
            if not output_file.exists():
                # 找出测试组织的掩码
                test_mask = adata_a.obs['batch'] == test_batch
                test_idx = np.where(test_mask)[0]
                test_cell_names_a = adata_a.obs_names[test_idx]
                test_cell_names_b = adata_b.obs_names[test_idx]

                test_pair_labels_a = pair_labels_a[test_idx]
                test_pair_labels_b = pair_labels_b[test_idx]

                pair_labels_df = pd.DataFrame({
                    'cell_name_a': test_cell_names_a,
                    'pair_label_a': test_pair_labels_a,
                    'cell_name_b': test_cell_names_b,
                    'pair_label_b': test_pair_labels_b
                })
                pair_labels_df.to_csv(output_file, index=False)

            # 计算 train_idx 和 test_idx
            test_mask = adata_a.obs['batch'] == test_batch
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            train_cell_names_a = adata_a.obs_names[train_idx].tolist()
            train_cell_names_b = adata_b.obs_names[train_idx].tolist()
            test_cell_names_a = adata_a.obs_names[test_idx].tolist()
            test_cell_names_b = adata_b.obs_names[test_idx].tolist()

            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_cell_names_a': train_cell_names_a,
                'train_cell_names_b': train_cell_names_b,
                'test_cell_names_a': test_cell_names_a,
                'test_cell_names_b': test_cell_names_b,
            })


    elif args.split in ['Cross_Protocols']:
        protocols = adata_a.obs['protocol'].unique()

        splits = []
        for i, test_protocol in enumerate(protocols):
            output_file = cache_dir / f"Total_{args.folds}_fold_{i + 1}_test_cells.csv"
            
            if not output_file.exists():
                # 找出测试组织的掩码
                test_mask = adata_a.obs['protocol'] == test_protocol
                test_idx = np.where(test_mask)[0]
                test_cell_names_a = adata_a.obs_names[test_idx]
                test_cell_names_b = adata_b.obs_names[test_idx]

                test_pair_labels_a = pair_labels_a[test_idx]
                test_pair_labels_b = pair_labels_b[test_idx]

                pair_labels_df = pd.DataFrame({
                    'cell_name_a': test_cell_names_a,
                    'pair_label_a': test_pair_labels_a,
                    'cell_name_b': test_cell_names_b,
                    'pair_label_b': test_pair_labels_b
                })
                pair_labels_df.to_csv(output_file, index=False)

            # 计算 train_idx 和 test_idx
            test_mask = adata_a.obs['protocol'] == test_protocol
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            train_cell_names_a = adata_a.obs_names[train_idx].tolist()
            train_cell_names_b = adata_b.obs_names[train_idx].tolist()
            test_cell_names_a = adata_a.obs_names[test_idx].tolist()
            test_cell_names_b = adata_b.obs_names[test_idx].tolist()

            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_cell_names_a': train_cell_names_a,
                'train_cell_names_b': train_cell_names_b,
                'test_cell_names_a': test_cell_names_a,
                'test_cell_names_b': test_cell_names_b,
            })
        
    elif args.split == 'data_type':
        splits = []
        data_type = adata_a.obs['data_type'].unique()

        train_key = 'control'
        test_key = 'perturbation'

        output_file = cache_dir / f"Total_{args.folds}_fold_1_test_cells.csv"
        
        if not output_file.exists():
            test_mask = adata_a.obs['data_type'] == test_key
            test_idx = np.where(test_mask)[0]
            test_cell_names_a = adata_a.obs_names[test_idx]
            test_cell_names_b = adata_b.obs_names[test_idx]

            test_pair_labels_a = pair_labels_a[test_idx]
            test_pair_labels_b = pair_labels_b[test_idx]

            pair_labels_df = pd.DataFrame({
                'cell_name_a': test_cell_names_a,
                'pair_label_a': test_pair_labels_a,
                'cell_name_b': test_cell_names_b,
                'pair_label_b': test_pair_labels_b
            })
            pair_labels_df.to_csv(output_file, index=False)

        # 计算 train_idx 和 test_idx
        test_mask = adata_a.obs['data_type'] == test_key
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        train_cell_names_a = adata_a.obs_names[train_idx].tolist()
        train_cell_names_b = adata_b.obs_names[train_idx].tolist()
        test_cell_names_a = adata_a.obs_names[test_idx].tolist()
        test_cell_names_b = adata_b.obs_names[test_idx].tolist()

        splits.append({
            'train_idx': train_idx,
            'test_idx': test_idx,
            'train_cell_names_a': train_cell_names_a,
            'train_cell_names_b': train_cell_names_b,
            'test_cell_names_a': test_cell_names_a,
            'test_cell_names_b': test_cell_names_b,
        })


    elif args.split == 'all':
        splits = []
        # 使用全部数据作为训练集和测试集
        train_idx = np.arange(len(adata_a.obs_names))  # 全部索引作为train_idx
        test_idx = train_idx  # 全部索引作为test_idx

        output_file = cache_dir / f"Total_{args.folds}_fold_1_test_cells.csv"
        
        if not output_file.exists():
            test_cell_names_a = adata_a.obs_names[test_idx]
            test_cell_names_b = adata_b.obs_names[test_idx]

            test_pair_labels_a = pair_labels_a[test_idx]
            test_pair_labels_b = pair_labels_b[test_idx]

            pair_labels_df = pd.DataFrame({
                'cell_name_a': test_cell_names_a,
                'pair_label_a': test_pair_labels_a,
                'cell_name_b': test_cell_names_b,
                'pair_label_b': test_pair_labels_b
            })
            pair_labels_df.to_csv(output_file, index=False)

        train_cell_names_a = adata_a.obs_names[train_idx].tolist()
        train_cell_names_b = adata_b.obs_names[train_idx].tolist()
        test_cell_names_a = adata_a.obs_names[test_idx].tolist()
        test_cell_names_b = adata_b.obs_names[test_idx].tolist()

        splits.append({
            'train_idx': train_idx,
            'test_idx': test_idx,
            'train_cell_names_a': train_cell_names_a,
            'train_cell_names_b': train_cell_names_b,
            'test_cell_names_a': test_cell_names_a,
            'test_cell_names_b': test_cell_names_b,
        })

    elif args.split == 'cell_type':
        # 1. 获取所有 cell_type 并按细胞数量从高到低排序
        cell_type_counts = adata_a.obs['cell_type'].value_counts().sort_values(ascending=False)
        sorted_cell_types = cell_type_counts.index.tolist()

        logger.info(f"细胞类型从高到低排序（共 {len(sorted_cell_types)} 类）：")
        for ct in sorted_cell_types:
            logger.info(f"   {ct:30s} : {cell_type_counts[ct]:6d} cells")

        # 2. 轮流分配到 4 个 group（5-5-5-4）
        groups = [[] for _ in range(4)]
        for i, ct in enumerate(sorted_cell_types):
            groups[i % 4].append(ct)

        # 3. 计算每个 group 的细胞数
        group_sizes = [sum(cell_type_counts[ct] for ct in g) for g in groups]

        logger.info("轮流分配结果：")
        for i, g in enumerate(groups):
            logger.info(f"  Fold {i+1} 测试集 ({len(g)} 类): {group_sizes[i]:5d} cells → {g}")

        # 4. 构造 splits
        splits = []
        for fold_idx in range(4):
            test_cell_types = groups[fold_idx]
            train_cell_types = [ct for j, grp in enumerate(groups) if j != fold_idx for ct in grp]

            test_mask = adata_a.obs['cell_type'].isin(test_cell_types)
            train_mask = ~test_mask

            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            logger.info(f"Fold {fold_idx + 1:2d}: Test = {len(test_idx):5d} cells | Train = {len(train_idx):5d} cells")

            output_file = cache_dir / f"Total_{args.folds}_fold_{fold_idx + 1}_test_cells.csv"

            if not output_file.exists():
                test_cell_names_a = adata_a.obs_names[test_idx]
                test_cell_names_b = adata_b.obs_names[test_idx]
                test_pair_labels_a = pair_labels_a[test_idx]
                test_pair_labels_b = pair_labels_b[test_idx]

                pair_labels_df = pd.DataFrame({
                    'cell_name_a': test_cell_names_a,
                    'pair_label_a': test_pair_labels_a,
                    'cell_name_b': test_cell_names_b,
                    'pair_label_b': test_pair_labels_b
                })
                pair_labels_df.to_csv(output_file, index=False)

            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_cell_names_a': adata_a.obs_names[train_idx].tolist(),
                'train_cell_names_b': adata_b.obs_names[train_idx].tolist(),
                'test_cell_names_a': adata_a.obs_names[test_idx].tolist(),
                'test_cell_names_b': adata_b.obs_names[test_idx].tolist(),
            })

       

    if args.first_run == True:
        return

    pad_token = "<pad>"
    special_tokens = [pad_token]
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    pad_value = args.n_bins
    n_input_bins = args.n_bins + 1

    for i, split in enumerate(splits):

        logger.info(f"Fold {i + 1}:")
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

        GRN_embds_path = cache_dir / args.GRNs / f"pretrain_total_{args.folds}_fold_{i + 1}_gene_embeddings.csv"
        GRN_embds = pd.read_csv(GRN_embds_path, index_col=0, header=0)
        GRN_embds = GRN_embds[GRN_embds.index.isin(genes)]

        if i == 0:
            genes_set = set(genes)
            grn_genes_set = set(GRN_embds.index)
            missing_genes = genes_set - grn_genes_set
            if missing_genes:
                logger.warning(f"GRN_embds missing {len(missing_genes)} genes: {missing_genes}")
            else:
                logger.info(f"GRN_embds contains all {len(genes)} genes")

        GRN_indices = np.array(GRN_embds.index)
        GRN_embds = GRN_embds.to_numpy()

        # 之前如果序列中包含GRN_embds中不存在的基因，则会报错，在小鼠基因序列中情况会变严重，因为小鼠基因是人类基因转过来的
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

        model = Gene_Transformer(
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
        best_model_path = save_dir / f'best_model_fold_{i + 1}.pth'

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
                level=level_tensor if args.grn_la == True else None,
            )

            save_path = save_dir if args.save_embds is True else None

            val_loss, best_result_a, best_result_b = evaluate(
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
                fold=i,
                best_result_a=best_result_a,
                best_result_b=best_result_b,
                level=level_tensor if args.grn_la == True else None,
                svg_df=svg_df if args.evaluation_a == 'Pearson-SVG' else None,
            )

            elapsed = time.time() - epoch_start_time
            logger.info("-" * 80)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | test loss {val_loss:5.4f} | "
            )
            logger.info("-" * 80)

            train_losses.append(0)
            test_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                # torch.save(model.state_dict(), best_model_path)
                logger.info(f"Epoch {epoch}: New best test loss {best_loss:.6f}")
            else:
                counter += 1
                if counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement")
                    break

       
        del train_loader, test_loader
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    run()