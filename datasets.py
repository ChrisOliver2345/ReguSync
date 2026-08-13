import sys
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import os
from utils import Binning, Binning_CITE_seq
from preprocess import RNA_Preprocessor, ATAC_Preprocessor
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path
import scipy.sparse as sp
import pickle 
import squidpy as sq
import gc


def load_data(args):
    # Retrieve file paths for pre-split paired modality data.
    modal_a_train_path = args.modal_a_train
    modal_a_test_path = args.modal_a_test
    modal_b_train_path = args.modal_b_train
    modal_b_test_path = args.modal_b_test
    
    # Load training and test AnnData objects for both modalities.
    modal_a_train_adata = sc.read_h5ad(modal_a_train_path)
    modal_a_test_adata = sc.read_h5ad(modal_a_test_path)
    modal_b_train_adata = sc.read_h5ad(modal_b_train_path)
    modal_b_test_adata = sc.read_h5ad(modal_b_test_path)

    # Annotate each AnnData object with its predefined train/test split label.
    modal_a_train_adata.obs["train_test_split"] = "train"
    modal_a_test_adata.obs["train_test_split"] = "test"
    modal_b_train_adata.obs["train_test_split"] = "train"
    modal_b_test_adata.obs["train_test_split"] = "test"

    # Verify that paired training sets contain the same number of cells.
    if modal_a_train_adata.n_obs != modal_b_train_adata.n_obs:
        raise ValueError("The training sets of modal_a and modal_b have different numbers of cells.")

    # Verify that paired test sets contain the same number of cells.
    if modal_a_test_adata.n_obs != modal_b_test_adata.n_obs:
        raise ValueError("The test sets of modal_a and modal_b have different numbers of cells.")

    # Ensure that paired training cells are aligned in the same order across modalities.
    if not all(modal_a_train_adata.obs_names == modal_b_train_adata.obs_names):
        raise ValueError("The training cells of modal_a and modal_b are not matched in order.")

    # Ensure that paired test cells are aligned in the same order across modalities.
    if not all(modal_a_test_adata.obs_names == modal_b_test_adata.obs_names):
        raise ValueError("The test cells of modal_a and modal_b are not matched in order.")

    # Concatenate training and test data within each modality while preserving cell order.
    modal_a_adata = sc.concat(
        [modal_a_train_adata, modal_a_test_adata],
        join="outer",
        label=None,
        index_unique=None
    )

    modal_b_adata = sc.concat(
        [modal_b_train_adata, modal_b_test_adata],
        join="outer",
        label=None,
        index_unique=None
    )

    # Confirm that the merged paired modalities remain cell-aligned after concatenation.
    if not all(modal_a_adata.obs_names == modal_b_adata.obs_names):
        raise ValueError("The merged modal_a and modal_b cells are not matched in order.")

    # Locate the directory containing gene sequences ordered by hierarchical topological sorting.
    gene_order_dir = Path(f"./Gene_order/{args.GRNs}")
    os.makedirs(gene_order_dir, exist_ok=True)
  
    # Load species-specific gene ordering information with regulatory layer annotations.
    if args.species == 'mouse':
        gene_sequence_df = pd.read_csv(gene_order_dir / "gene_sequence_mouse_layered.csv", index_col=None, header=0)
    elif args.species == 'human':
        gene_sequence_df = pd.read_csv(gene_order_dir / "gene_sequence_human_layered.csv", index_col=None, header=0)
    else:
        raise ValueError(f"Unsupported species: {args.species}")

    # Construct spatial coordinates from row and column annotations when spatial data are used.
    if args.spatial == True:
        rows = modal_a_adata.obs['row'].values.astype(float)
        cols = modal_a_adata.obs['col'].values.astype(float)
        spatial_locs = np.column_stack((cols, rows)) 
        modal_a_adata.obsm['spatial'] = spatial_locs
        modal_b_adata.obsm['spatial'] = spatial_locs
    
    return modal_a_adata, modal_b_adata, gene_sequence_df


def preprocess_data(modal_a_adata, modal_b_adata, args, logger):
    n_cells = modal_a_adata.n_obs
    pair_labels = np.arange(n_cells)
    modal_a_adata.obs["pair_labels"] = pair_labels
    modal_b_adata.obs["pair_labels"] = pair_labels
   
    if args.modal_a == 'RNA':
        adata_a = modal_a_adata

    if args.modal_b == 'ATAC':

        cache_dir = f"./Cache/{args.dataset}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"rp_score.pkl")

        if os.path.exists(cache_file):
            logger.info(f"Loading cached RP score matrix from {cache_file}")
            with open(cache_file, 'rb') as f:
                rp_sparse = pickle.load(f)
                index = pickle.load(f)
                columns = pickle.load(f)
        else:
            raise FileNotFoundError( f"Cached RP score matrix not found: {cache_file}. " 
                                    "Please download the precomputed RP score matrix from the GitHub repository" 
                                    "and place it at the expected path.")

        adata_b = ad.AnnData(
            X=rp_sparse, 
            obs=modal_b_adata.obs.loc[index], 
            var=pd.DataFrame(index=columns)
        )
    
    elif args.modal_b == 'ADT':
        adata_b = modal_b_adata

    gc.collect()

    data_is_raw = True
    preprocessor_a = RNA_Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=50,  # step 1
        filter_cell_by_counts=50,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=args.n_hvg,  # 5. whether to subset the raw data to highly variable genes
        # hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger"
        hvg_flavor=args.hvg_flavor,
        # hvg_use_key="X_log1p"
    )

    if args.modal_b == 'ATAC':
        data_is_raw = False
        preprocessor_b = ATAC_Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=50,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=False,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=True,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=args.n_hvg,  # 5. whether to subset the raw data to highly variable genes
            # hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            hvg_flavor=args.hvg_flavor_2,
            # hvg_use_key="X_log1p",
            basic_binary_data=True,
            basic_fpeaks=0.005,
            basic_tfidf=True,
            bssic_normalize=True,
            basic_filter_features=True,
            args=args
        )

    elif args.modal_b == 'ADT':

        print(f"ADT preprocess method: {args.ADT_preprocess_method}")

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

            clr_result = np.apply_along_axis(
                seurat_clr, 1, (adata.X.A if sp.issparse(adata.X) else adata.X)
            )

            adata.X = clr_result
            adata.layers['X_clr'] = clr_result

            return adata

        if args.ADT_preprocess_method == 'CLR':
            adata_b = clr_normalize_each_cell(adata_b)  
        elif args.ADT_preprocess_method == 'standard':
            if sp.issparse(adata_b.X):
                adata_b.X = adata_b.X.toarray()
            adata_b.layers['X_raw'] = adata_b.X.copy()
            sc.pp.normalize_total(adata_b)
            sc.pp.log1p(adata_b)
            adata_b.layers['X_log1p'] = adata_b.X.copy()
            sc.pp.scale(adata_b)
            adata_b.layers['X_scale'] = adata_b.X.copy()
            adata_b.X = adata_b.layers['X_log1p']

    adata_a = preprocessor_a(adata_a, batch_key=None)

    if args.modal_b == 'ATAC':
        common_genes = np.intersect1d(adata_a.var_names, adata_b.var_names)
        adata_a = adata_a[:, common_genes].copy()
        adata_b = adata_b[:, common_genes].copy()

        adata_b, adata_basic_b = preprocessor_b(adata_b, modal_b_adata, batch_key=None)

        common_genes = np.intersect1d(adata_a.var_names, adata_b.var_names)
        adata_a = adata_a[:, common_genes].copy()
        adata_b = adata_b[:, common_genes].copy()

        if args.evaluation_a == 'Pearson-SVG':
            sq.gr.spatial_neighbors(adata_a, coord_type="grid", n_neighs=6)
            sq.gr.spatial_autocorr(adata_a, mode="moran", genes=adata_a.var_names, n_perms=None, n_jobs=1)
            svg_df = adata_a.uns['moranI']
            if args.dataset == 'Spatial_RNA+ATAC_Mouse_P22_GSE205055':
                svg_df_sorted = svg_df.sort_values(by='I', ascending=False)
                top_svg = svg_df_sorted.index[:10000].tolist()
                adata_a = adata_a[:, top_svg].copy()

                common_genes = np.intersect1d(adata_a.var_names, adata_b.var_names)
                adata_a = adata_a[:, common_genes].copy()
                adata_b = adata_b[:, common_genes].copy()


        common_pair = set(adata_a.obs["pair_labels"]).intersection(set(adata_b.obs["pair_labels"]))
        adata_a = adata_a[adata_a.obs["pair_labels"].isin(common_pair)].copy()
        adata_b = adata_b[adata_b.obs["pair_labels"].isin(common_pair)].copy()
        adata_basic_b = adata_basic_b[adata_basic_b.obs["pair_labels"].isin(common_pair)].copy()

        adata_basic_a = adata_a

        hvg_a = adata_a.var[adata_a.var['highly_variable']].index
        hvg_b = adata_b.var[adata_b.var['highly_variable']].index
        hvg_union = np.union1d(hvg_a, hvg_b)

        adata_a = adata_a[:, hvg_union].copy()
        adata_b = adata_b[:, hvg_union].copy()

        binning = Binning(binning=args.n_bins, result_binned_key="X_binned")

        adata_a_bin, adata_b_bin = binning(adata_a, adata_b)

    elif args.modal_b == 'ADT':

        if args.evaluation_a == 'Pearson-SVG':
            sq.gr.spatial_neighbors(adata_a, coord_type="grid", n_neighs=6)
            sq.gr.spatial_autocorr(adata_a, mode="moran", genes=adata_a.var_names, n_perms=None, n_jobs=1)
            svg_df = adata_a.uns['moranI']

        common_pair = set(adata_a.obs["pair_labels"]).intersection(set(adata_b.obs["pair_labels"]))
        adata_a = adata_a[adata_a.obs["pair_labels"].isin(common_pair)].copy()
        adata_b = adata_b[adata_b.obs["pair_labels"].isin(common_pair)].copy()

        adata_basic_a = adata_a
        adata_basic_b = adata_b

        hvg_a = adata_a.var[adata_a.var['highly_variable']].index
        feature_b = adata_b.var_names
        feature_union = np.union1d(hvg_a, feature_b)
        binning = Binning_CITE_seq(binning=args.n_bins, result_binned_key="X_binned")
        adata_a_bin, adata_b_bin = binning(adata_a, adata_b, feature_union)


    if args.have_labels:
        celltype_id_labels = adata_a_bin.obs["cell_type"].astype("category").cat.codes.values
        celltypes = adata_a_bin.obs["cell_type"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata_a_bin.obs["cell_type"].astype("category").cat.categories))
        adata_a_bin.obs["celltype_id"] = celltype_id_labels
        adata_b_bin.obs["celltype_id"] = celltype_id_labels
    else:
        adata_a_bin.obs["celltype_id"] = np.nan
        adata_b_bin.obs["celltype_id"] = np.nan

    if args.evaluation_a == 'Pearson-SVG':
        return adata_a_bin, adata_b_bin, adata_basic_a, adata_basic_b, svg_df
    else:
        return adata_a_bin, adata_b_bin, adata_basic_a, adata_basic_b, None


def prepare_data(tokenized_train, tokenized_test, train_celltype_labels, test_celltype_labels, train_pair_labels,
                 test_pair_labels, train_basic, test_basic, train_true, test_true, train_spatial_locs, test_spatial_locs, args):
    
    tokenized_train_a, tokenized_train_b = tokenized_train
    tokenized_test_a, tokenized_test_b = tokenized_test
    train_celltypes_a, train_celltypes_b = train_celltype_labels
    test_celltypes_a, test_celltypes_b = test_celltype_labels
    train_pair_labels_a, train_pair_labels_b = train_pair_labels
    test_pair_labels_a, test_pair_labels_b = test_pair_labels
    train_basic_a, train_basic_b = train_basic
    test_basic_a, test_basic_b = test_basic
    train_true_a, train_true_b = train_true
    test_true_a, test_true_b = test_true
    
    values_train_a = tokenized_train_a["values"]
    values_train_b = tokenized_train_b["values"]

    values_test_a = tokenized_test_a["values"]
    values_test_b = tokenized_test_b["values"]

    if not args.ram_usage_optimization:
        values_train_a = values_train_a.to(args.device)
        values_train_b = values_train_b.to(args.device)
        values_test_a = values_test_a.to(args.device)
        values_test_b = values_test_b.to(args.device)

        tensor_basic_train_a = torch.from_numpy(train_basic_a.toarray()).float().to(args.device)
        tensor_basic_test_a = torch.from_numpy(test_basic_a.toarray()).float().to(args.device)
        tensor_basic_train_b = torch.from_numpy(train_basic_b.toarray() if args.modal_b == 'ATAC' else train_basic_b).float().to(args.device)
        tensor_basic_test_b = torch.from_numpy(test_basic_b.toarray() if args.modal_b == 'ATAC' else test_basic_b).float().to(args.device)

        tensor_true_train_a = torch.from_numpy(train_true_a.toarray()).float().to(args.device)
        tensor_true_test_a = torch.from_numpy(test_true_a.toarray()).float().to(args.device)
        tensor_true_train_b = torch.from_numpy(train_true_b.toarray() if args.modal_b == 'ATAC' else train_true_b).float().to(args.device)
        tensor_true_test_b = torch.from_numpy(test_true_b.toarray() if args.modal_b == 'ATAC' else test_true_b).float().to(args.device)

    gene_ids_train_a, gene_ids_train_b, gene_ids_test_a, gene_ids_test_b = (
        tokenized_train_a["genes"],
        tokenized_train_b["genes"],
        tokenized_test_a["genes"],
        tokenized_test_b["genes"],
    )

    tensor_celltypes_train_a = torch.from_numpy(train_celltypes_a).long()
    tensor_celltypes_train_b = torch.from_numpy(train_celltypes_b).long()
    tensor_celltypes_test_a = torch.from_numpy(test_celltypes_a).long()
    tensor_celltypes_test_b = torch.from_numpy(test_celltypes_b).long()

    tensor_pair_labels_train_a = torch.from_numpy(train_pair_labels_a).long()
    tensor_pair_labels_train_b = torch.from_numpy(train_pair_labels_b).long()
    tensor_pair_labels_test_a = torch.from_numpy(test_pair_labels_a).long()
    tensor_pair_labels_test_b = torch.from_numpy(test_pair_labels_b).long()

    tensor_spatial_locs_train = torch.from_numpy(train_spatial_locs).float() if args.spatial == True else None
    tensor_spatial_locs_test = torch.from_numpy(test_spatial_locs).float() if args.spatial == True else None

    if not args.ram_usage_optimization:
        train_data_pt_a = {
            "gene_ids": gene_ids_train_a,
            "values": values_train_a,
            "celltype_labels": tensor_celltypes_train_a,
            "pair_labels": tensor_pair_labels_train_a,
            "value_basic": tensor_basic_train_a,
            "value_true": tensor_true_train_a,
            "locs": tensor_spatial_locs_train if args.spatial == True else None
        }
    else:
        train_data_pt_a = {
        "gene_ids": gene_ids_train_a,
        "values": values_train_a,
        "celltype_labels": tensor_celltypes_train_a,
        "pair_labels": tensor_pair_labels_train_a,
        "value_basic": train_basic_a,  
        "value_true": train_true_a,   
        "locs": tensor_spatial_locs_train if args.spatial == True else None
    }

    if not args.ram_usage_optimization:
        train_data_pt_b = {
            "gene_ids": gene_ids_train_b,
            "values": values_train_b,
            "celltype_labels": tensor_celltypes_train_b,
            "pair_labels": tensor_pair_labels_train_b,
            "value_basic": tensor_basic_train_b,
            "value_true": tensor_true_train_b,
            "locs": tensor_spatial_locs_train if args.spatial == True else None
        }
    else:
        train_data_pt_b = {
        "gene_ids": gene_ids_train_b,
        "values": values_train_b,
        "celltype_labels": tensor_celltypes_train_b,
        "pair_labels": tensor_pair_labels_train_b,
        "value_basic": train_basic_b, 
        "value_true": train_true_b,   
        "locs": tensor_spatial_locs_train if args.spatial == True else None
    }

    if not args.ram_usage_optimization:
        test_data_pt_a = {
            "gene_ids": gene_ids_test_a,
            "values": values_test_a,
            "celltype_labels": tensor_celltypes_test_a,
            "pair_labels": tensor_pair_labels_test_a,
            "value_basic": tensor_basic_test_a,
            "value_true": tensor_true_test_a,
            "locs": tensor_spatial_locs_test if args.spatial == True else None
        }
    else:
        test_data_pt_a = {
        "gene_ids": gene_ids_test_a,
        "values": values_test_a,
        "celltype_labels": tensor_celltypes_test_a,
        "pair_labels": tensor_pair_labels_test_a,
        "value_basic": test_basic_a,   
        "value_true": test_true_a,
        "locs": tensor_spatial_locs_test if args.spatial == True else None
    }

    if not args.ram_usage_optimization:
        test_data_pt_b = {
            "gene_ids": gene_ids_test_b,
            "values": values_test_b,
            "celltype_labels": tensor_celltypes_test_b,
            "pair_labels": tensor_pair_labels_test_b,
            "value_basic": tensor_basic_test_b,
            "value_true": tensor_true_test_b,
            "locs": tensor_spatial_locs_test if args.spatial == True else None
        }
    else:
        test_data_pt_b = {
        "gene_ids": gene_ids_test_b,
        "values": values_test_b,
        "celltype_labels": tensor_celltypes_test_b,
        "pair_labels": tensor_pair_labels_test_b,
        "value_basic": test_basic_b,  
        "value_true": test_true_b,
        "locs": tensor_spatial_locs_test if args.spatial == True else None
    }


    return (train_data_pt_a, train_data_pt_b), (test_data_pt_a, test_data_pt_b)


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory=True,
    args=None

) -> DataLoader:

    data_pt_a, data_pt_b = data_pt

    if args.ram_usage_optimization:
        dataset_a = SeqDataset_sparse(data_pt_a, args)
        dataset_b = SeqDataset_sparse(data_pt_b, args)
    else:
        dataset_a = SeqDataset(data_pt_a, args)
        dataset_b = SeqDataset(data_pt_b, args)

    dataset = PairedDataset(dataset_a, dataset_b)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


# class SeqDataset(Dataset):
#     def __init__(self, data: Dict[str, torch.Tensor]):
#         self.data = data

#     def __len__(self):
#         return self.data["gene_ids"].shape[0]

#     def __getitem__(self, idx):
#         return {k: v[idx] for k, v in self.data.items()}

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], args):
        self.data = data
        self.args = args
        self.locs = data.get("locs") if args.spatial == True else None

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    # def __getitem__(self, idx):
    #     item = {k: v[idx] for k, v in self.data.items() if k != "locs"}
    #     item["locs"] = self.locs[idx] if self.args.spatial == True else None
    #     return item

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.data.items() if k != "locs"}
        if self.args.spatial == True:
            item["locs"] = self.locs[idx]
        return item


class SeqDataset_sparse(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], args):
        # Separate dense metadata from sparse expression matrices.
        self.gene_ids = data["gene_ids"]
        self.values = data["values"]
        self.celltype_labels = data["celltype_labels"]
        self.pair_labels = data["pair_labels"]
        self.value_basic = data["value_basic"]  # Sparse matrix.
        self.value_true = data["value_true"]    # Sparse matrix.
        self.locs = data["locs"] if args.spatial == True else None
        self.args = args

    def __len__(self):
        return self.gene_ids.shape[0]

    # def __getitem__(self, idx):
    #     # Extract sparse single-cell rows on demand and convert them to dense Torch tensors.
    #     if sp.issparse(self.value_basic):
    #         row_basic = self.value_basic[idx].toarray().ravel()
    #     else:
    #         row_basic = self.value_basic[idx]  # Slice directly when already dense.
    #     value_basic_tensor = torch.from_numpy(row_basic).float()

    #     if sp.issparse(self.value_true):
    #         row_true = self.value_true[idx].toarray().ravel()
    #     else:
    #         row_true = self.value_true[idx]
    #     value_true_tensor = torch.from_numpy(row_true).float()

    #     return {
    #         "gene_ids": self.gene_ids[idx],
    #         "values": self.values[idx],
    #         "celltype_labels": self.celltype_labels[idx],
    #         "pair_labels": self.pair_labels[idx],
    #         "value_basic": value_basic_tensor,
    #         "value_true": value_true_tensor,
    #         "locs": self.locs[idx] if self.args.spatial == True else None
    #     }

    def __getitem__(self, idx):
        # Extract sparse single-cell rows on demand and convert them to dense Torch tensors.
        if sp.issparse(self.value_basic):
            row_basic = self.value_basic[idx].toarray().ravel()
        else:
            row_basic = self.value_basic[idx]  # Slice directly when already dense.
        value_basic_tensor = torch.from_numpy(row_basic).float()

        if sp.issparse(self.value_true):
            row_true = self.value_true[idx].toarray().ravel()
        else:
            row_true = self.value_true[idx]
        value_true_tensor = torch.from_numpy(row_true).float()

        result = {
            "gene_ids": self.gene_ids[idx],
            "values": self.values[idx],
            "celltype_labels": self.celltype_labels[idx],
            "pair_labels": self.pair_labels[idx],
            "value_basic": value_basic_tensor,
            "value_true": value_true_tensor,
        }
        
        if self.args.spatial == True:
            result["locs"] = self.locs[idx]
        
        return result


class PairedDataset(Dataset):
    def __init__(self, dataset_a: Dataset, dataset_b: Dataset):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

        assert len(self.dataset_a) == len(self.dataset_b), \
            f"Dataset lengths do not match: {len(self.dataset_a)} != {len(self.dataset_b)}"

    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, idx):
        sample_a = self.dataset_a[idx]
        sample_b = self.dataset_b[idx]
        return sample_a, sample_b

def get_grn_embeddings(batch_padded, GRN_embds, GRN_genes, vocab, pad_token="<pad>", cls_token="<cls>", dtype=torch.float16):
    """
    Expand GRN_embds to (cell_num, gene_num, embds) and align it with tokenized gene sequences.

    Args:
        batch_padded (Dict[str, torch.Tensor]): Output from tokenize_and_pad_batch containing 'genes' and 'values'.
        GRN_embds (np.ndarray): Gene graph embeddings with shape (gene_num, embds).
        GRN_genes (np.ndarray): Gene names with shape (gene_num,) corresponding to GRN_embds.
        vocab (Vocab): Vocabulary mapping gene IDs and special tokens.
        pad_token (str): Padding token. Defaults to "<pad>".
        cls_token (str): Classification token. Defaults to "<cls>" and may be None.
        dtype (torch.dtype): Output tensor data type. Defaults to torch.float16.

    Returns:
        torch.Tensor: Tensor shaped (cell_num, gene_num, embds), aligned with batch_padded['genes'].
    """
    # Read dimensions from the inputs.
    cell_num, max_len = batch_padded['genes'].shape
    emb_dim = GRN_embds.shape[1]

    # Convert GRN_embds to a Torch tensor.
    GRN_embds = torch.from_numpy(GRN_embds).to(dtype=dtype, device=batch_padded['genes'].device)

    # Initialize the output tensor.
    aligned_embds = torch.zeros(cell_num, max_len, emb_dim, dtype=dtype, device=batch_padded['genes'].device)

    # Get special-token IDs.
    pad_id = vocab[pad_token]
    cls_id = vocab[cls_token] if cls_token is not None else None

    # Map gene names to GRN_embds indices.
    gene_to_idx = {str(gene): i for i, gene in enumerate(GRN_genes)}  # Normalize gene names to strings.
    vocab_to_grn_idx = torch.full((len(vocab),), -1, dtype=torch.long, device=batch_padded['genes'].device)

    # Get the vocabulary string-to-index mapping.
    vocab_stoi = vocab.get_stoi()

    # Map vocabulary gene IDs to GRN_embds indices.
    matched_genes = 0
    for gene, idx in vocab_stoi.items():
        gene_str = str(gene)  # Normalize the gene name to a string.
        if gene_str in gene_to_idx:
            vocab_to_grn_idx[idx] = gene_to_idx[gene_str]
            matched_genes += 1

    # Create masks for special tokens and regular genes.
    pad_mask = (batch_padded['genes'] == pad_id)
    cls_mask = (batch_padded['genes'] == cls_id) if cls_id is not None else torch.zeros_like(pad_mask, dtype=torch.bool)
    gene_mask = ~(pad_mask | cls_mask)

    # Populate embeddings for regular genes.
    if gene_mask.any():
        gene_ids = batch_padded['genes'][gene_mask]
        grn_indices = vocab_to_grn_idx[gene_ids]
        valid_mask = grn_indices != -1

        if valid_mask.any():
            valid_grn_indices = grn_indices[valid_mask]
            aligned_embds[gene_mask] = GRN_embds[valid_grn_indices]
        else:
            print("Warning: No valid gene embeddings were found; the gene IDs may be absent from GRN_embds.")

        # Report genes without matching embeddings.
        if (~valid_mask).any():
            invalid_gene_ids = gene_ids[~valid_mask]
            invalid_genes = [next((g for g, idx in vocab_stoi.items() if idx == gid.item()), None)
                             for gid in invalid_gene_ids]
            for gid, gene in zip(invalid_gene_ids, invalid_genes):
                print(f"Warning: Gene {gene} (ID: {gid.item()}) was not found in GRN_embds.")

    return aligned_embds


def get_grn_embeddings_2(batch_padded, GRN_embds, GRN_genes, vocab, pad_token="<pad>", cls_token="<cls>", dtype=torch.float16):
    """
    Expand GRN_embds to (cell_num, gene_num, embds) and align it with tokenized gene sequences.

    Args:
        batch_padded (Dict[str, torch.Tensor]): Output from tokenize_and_pad_batch containing 'genes' and 'values'.
        GRN_embds (np.ndarray): Gene graph embeddings with shape (gene_num, embds).
        GRN_genes (np.ndarray): Gene names with shape (gene_num,) corresponding to GRN_embds.
        vocab (Vocab): Vocabulary mapping gene IDs and special tokens.
        pad_token (str): Padding token. Defaults to "<pad>".
        cls_token (str): Classification token. Defaults to "<cls>" and may be None.
        dtype (torch.dtype): Output tensor data type. Defaults to torch.float16.

    Returns:
        torch.Tensor: Tensor shaped (cell_num, gene_num, embds), aligned with batch_padded['genes'].
    """
    # Read dimensions from the inputs.
    cell_num, max_len = batch_padded['genes'].shape
    emb_dim = GRN_embds.shape[1]

    # Convert GRN_embds to a Torch tensor.
    GRN_embds = torch.from_numpy(GRN_embds).to(dtype=dtype, device=batch_padded['genes'].device)

    # Initialize the output tensor.
    # aligned_embds = torch.zeros(cell_num, max_len, emb_dim, dtype=dtype, device=batch_padded['genes'].device)

    # Get special-token IDs.
    pad_id = vocab[pad_token]
    cls_id = vocab[cls_token] if cls_token is not None else None

    # Map gene names to GRN_embds indices.
    gene_to_idx = {str(gene): i for i, gene in enumerate(GRN_genes)}  # Normalize gene names to strings.
    vocab_to_grn_idx = torch.full((len(vocab),), -1, dtype=torch.long, device=batch_padded['genes'].device)

    # Get the vocabulary string-to-index mapping.
    vocab_stoi = vocab.get_stoi()

    # Map vocabulary gene IDs to GRN_embds indices.
    matched_genes = 0
    for gene, idx in vocab_stoi.items():
        gene_str = str(gene)  # Normalize the gene name to a string.
        if gene_str in gene_to_idx:
            vocab_to_grn_idx[idx] = gene_to_idx[gene_str]
            matched_genes += 1

    # All cells share one gene sequence, so compute using only the first cell.
    unique_genes = batch_padded['genes'][0]  # (max_len,)

    # Create masks from the shared sequence.
    pad_mask_single = (unique_genes == pad_id)
    cls_mask_single = (unique_genes == cls_id) if cls_id is not None else torch.zeros_like(pad_mask_single, dtype=torch.bool)
    gene_mask_single = ~(pad_mask_single | cls_mask_single)

    # Initialize the single-cell embedding tensor.
    aligned_embds_single = torch.zeros(1, max_len, emb_dim, dtype=dtype, device=batch_padded['genes'].device)

    # Populate regular-gene embeddings once.
    if gene_mask_single.any():
        gene_ids = unique_genes[gene_mask_single]
        grn_indices = vocab_to_grn_idx[gene_ids]
        valid_mask = grn_indices != -1

        if valid_mask.any():
            valid_grn_indices = grn_indices[valid_mask]
            aligned_embds_single[0, gene_mask_single] = GRN_embds[valid_grn_indices]
        else:
            print("Warning: No valid gene embeddings were found; the gene IDs may be absent from GRN_embds.")

        # Report genes without matching embeddings.
        if (~valid_mask).any():
            invalid_gene_ids = gene_ids[~valid_mask]
            invalid_genes = [next((g for g, idx in vocab_stoi.items() if idx == gid.item()), None)
                             for gid in invalid_gene_ids]
            for gid, gene in zip(invalid_gene_ids, invalid_genes):
                print(f"Warning: Gene {gene} (ID: {gid.item()}) was not found in GRN_embds.")

    # Broadcast the single-cell embeddings to all cells.
    # aligned_embds = aligned_embds_single.repeat(cell_num, 1, 1)

    return aligned_embds_single
