import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import random
import torch
from anndata import AnnData
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse
import logging
from pathlib import Path
import episcanpy.api as epi
import scipy
from typing import Dict, Optional, Union
from utils import TFIDF, TFIDF_sklearn



class RNA_Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
            self,
            use_key: Optional[str] = None,
            filter_gene_by_counts: Union[int, bool] = False,
            filter_cell_by_counts: Union[int, bool] = False,
            normalize_total: Union[float, bool] = 1e4,
            result_normed_key: Optional[str] = "X_normed",
            log1p: bool = False,
            result_log1p_key: str = "X_log1p",
            subset_hvg: Union[int, bool] = False,
            hvg_use_key: Optional[str] = None,
            hvg_flavor: str = "seurat_v3"
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)
    
        # step 1: filter genes
        if self.filter_gene_by_counts:
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )
       
        # step 2: filter cells
        if (
                isinstance(self.filter_cell_by_counts, int)
                and self.filter_cell_by_counts > 0
        ):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )
        
        raw_counts = _get_obs_rep(adata, layer=key_to_process)
        _set_obs_rep(adata, raw_counts, layer="X_raw")

        # step 3: normalize total
        if self.normalize_total:
            print("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            print("Log1p transforming ...")
            if is_logged:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            print("Subsetting highly variable genes ...")
            if batch_key is None:
                print(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            
            print("adata features (n_vars):", adata.n_vars)
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=False,
            )

        print('Preprocess finished.')
        return adata

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


class ATAC_Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
            self,
            use_key: Optional[str] = None,
            filter_gene_by_counts: Union[int, bool] = False,
            filter_cell_by_counts: Union[int, bool] = False,
            normalize_total: Union[float, bool] = 1e4,
            result_normed_key: Optional[str] = "X_normed",
            log1p: bool = False,
            result_log1p_key: str = "X_log1p",
            subset_hvg: Union[int, bool] = False,
            hvg_use_key: Optional[str] = None,
            hvg_flavor: str = "seurat_v3",

            basic_binary_data=True,
            basic_fpeaks=0.005,
            basic_tfidf=True,
            bssic_normalize=True,
            basic_filter_features=False,

            args=None

    ):

        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor

        self.basic_binary_data = basic_binary_data
        self.basic_fpeaks = basic_fpeaks
        self.basic_tfidf = basic_tfidf
        self.basic_normalize = bssic_normalize
        self.basic_filter_features = basic_filter_features

        self.args = args

    def __call__(self,
                 adata: AnnData,
                 adata_basic: AnnData,
                 batch_key: Optional[str] = None,
                 ) -> Dict:

        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        if self.basic_binary_data:
            print('binarizing data.')
            epi.pp.binarize(adata_basic)

        if self.args.dataset == 'Ma-2020_SHARE-seq_Mouse_Skin' or self.args.dataset == 'Chen-2019_SNARE-seq_Mouse_Cortex':
            pass
            # self.basic_filter_features = False

        if self.basic_filter_features:
            print('filter out peaks appear lower than ' + str(self.basic_fpeaks * 100) + '% cells.')
            epi.pp.filter_features(adata_basic, min_cells=np.ceil(self.basic_fpeaks * adata_basic.shape[0]))

        if self.args.dataset == 'Spatial_RNA+ATAC_Mouse_P22_GSE205055':
            print('selecting high variable features...')
            epi.pp.select_var_feature(adata_basic, nb_features=30000, show=False) 

        binarize_counts = _get_obs_rep(adata_basic)
        _set_obs_rep(adata_basic, binarize_counts, layer="X_binarize")

        if self.basic_tfidf:
            print('TF-IDF transformation.')
            count_mat = adata_basic.X.copy()
            # adata_basic.X, divide_title, multiply_title = TFIDF(count_mat)
            adata_basic.X, _, _ = TFIDF_sklearn(count_mat)

        if self.basic_normalize:
            print('normalizing data.')
            max_temp = np.max(adata_basic.X)
            adata_basic.X = adata_basic.X / max_temp

        
        # step 1: filter genes
        if self.filter_gene_by_counts:
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if (
                isinstance(self.filter_cell_by_counts, int)
                and self.filter_cell_by_counts > 0
        ):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # step 3: normalize total
        if self.normalize_total:
            print("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            print("Log1p transforming ...")
            if is_logged:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            print("Subsetting highly variable genes ...")
            if batch_key is None:
                print(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=False,
            )

        return adata, adata_basic

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
            self,
            use_key: Optional[str] = None,
            filter_gene_by_counts: Union[int, bool] = False,
            filter_cell_by_counts: Union[int, bool] = False,
            normalize_total: Union[float, bool] = 1e4,
            result_normed_key: Optional[str] = "X_normed",
            log1p: bool = False,
            result_log1p_key: str = "X_log1p",
            subset_hvg: Union[int, bool] = False,
            hvg_use_key: Optional[str] = None,
            hvg_flavor: str = "seurat_v3"
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.filter_gene_by_counts:
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if (
                isinstance(self.filter_cell_by_counts, int)
                and self.filter_cell_by_counts > 0
        ):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # step 3: normalize total
        if self.normalize_total:
            print("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            print("Log1p transforming ...")
            if is_logged:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            print("Subsetting highly variable genes ...")
            if batch_key is None:
                print(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=False,
            )

        return adata

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True