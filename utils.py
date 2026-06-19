import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import random
import torch
from anndata import AnnData
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse, csr_matrix
import logging
from pathlib import Path
import episcanpy.api as epi
import scipy
from typing import Dict, Optional, Union
import anndata
from collections import Counter


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
        row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.max() == 0:
        print(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


class Binning:

    def __init__(
            self,
            binning: Optional[int] = None,
            result_binned_key: str = "X_binned",
    ):
        self.binning = binning
        self.result_binned_key = result_binned_key
        self.use_key = 'X'

    def __call__(self, adata_a, adata_b):

        adata_a = self.run_binning(adata_a)
        adata_b = self.run_binning(adata_b)

        return adata_a, adata_b

    def run_binning(self, adata):

        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None

        if self.binning:
            print("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.toarray() if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
            for row in layer_data:
                if row.max() == 0:
                    print(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

        return adata


class Binning_CITE_seq:

    def __init__(
            self,
            binning: Optional[int] = None,
            result_binned_key: str = "X_binned",
            use_key: str = "X",
    ):
        self.binning = binning
        self.result_binned_key = result_binned_key
        self.use_key = use_key

    def __call__(self, adata_a, adata_b, feature_union):

        adata_a = self.run_binning(adata_a, feature_union)
        adata_b = self.run_binning(adata_b, feature_union)

        return adata_a, adata_b

    def run_binning(self, adata, feature_union):

        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None

        if self.binning:
            print("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            
            # 获取 adata 的特征名
            feature_names = adata.var_names if key_to_process is None else adata.obsm[key_to_process].columns
            feature_names = np.array(feature_names)
            # 找到共同特征
            common_features = np.intersect1d(feature_names, feature_union)
            print("Number of common features: " + str(len(common_features)))
            # 创建特征映射，从 feature_names 到列索引
            feature_map = {f: i for i, f in enumerate(feature_names)}
            # 映射共同特征到 layer_data 的列索引
            common_indices = [feature_map[f] for f in common_features if f in feature_map]
            # 映射 feature_union 的特征到输出数组的索引
            feature_union_map = {f: i for i, f in enumerate(feature_union)}
            
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.toarray() if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )

            n_cells = layer_data.shape[0]
            n_features = len(feature_union)

            for i in range(n_cells):
                row = layer_data[i]
                if row.max() == 0:
                    print(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    # 输出全零数组，长度为 feature_union 的特征数
                    binned_rows.append(np.zeros(n_features, dtype=np.int64))
                    bin_edges.append(np.zeros(n_bins - 1))
                    continue

                # --- 修改：仅对共同特征分箱 ---
                non_zero_ids = row[common_indices].nonzero()[0]
                non_zero_row = row[common_indices][non_zero_ids]
                if len(non_zero_row) == 0:
                    # 共同特征全为零，输出全零数组
                    binned_rows.append(np.zeros(n_features, dtype=np.int64))
                    bin_edges.append(np.zeros(n_bins - 1))
                    continue

                # 计算分箱
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1

                # --- 修改：将分箱结果映射到 feature_union 顺序 ---
                binned_row = np.zeros(n_features, dtype=np.int64)  # 初始化为 0，自动处理缺失特征
                # 将分箱结果填入 feature_union 对应的位置
                for j, idx in enumerate(non_zero_ids):
                    feature = common_features[idx]
                    binned_row[feature_union_map[feature]] = non_zero_digits[j]
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
                # --- 修改结束 ---

            # shapes = [np.array(edge).shape for edge in bin_edges]
            # shape_counts = Counter(shapes)
            # print("All unique shapes and counts:", shape_counts)
            

            # 将分箱结果和边界转换为 numpy 数组
            binned_data = np.stack(binned_rows)  # 形状 (n_cells, len(feature_union))
            bin_edges_data = np.stack(bin_edges)  # 形状 (n_cells, n_bins - 1)
            # 创建新的 AnnData 对象
            new_adata = anndata.AnnData(
                X=csr_matrix((n_cells, n_features), dtype=np.int8), 
                obs=adata.obs.copy(),  
                var=pd.DataFrame(index=feature_union)  
            )
            # 存储分箱结果到 layers
            new_adata.layers[self.result_binned_key] = binned_data
            # 存储分箱边界到 obsm
            new_adata.obsm["bin_edges"] = bin_edges_data

            return new_adata

        return adata


def sc_logger(save_path):
    logger = logging.getLogger("Project")
    if not logger.hasHandlers() or len(logger.handlers) == 0:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    add_file_handler(logger, save_path)

    return logger


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def TFIDF(count_mat):
    """
    TF-IDF transformation for matrix.

    Parameters
    ----------
    count_mat
        numpy matrix with cells as rows and peak as columns, cell * peak.

    Returns
    ----------
    tfidf_mat
        matrix after TF-IDF transformation.

    divide_title
        matrix divided in TF-IDF transformation process, would be used in "inverse_TFIDF".

    multiply_title
        matrix multiplied in TF-IDF transformation process, would be used in "inverse_TFIDF".

    """

    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat, axis=0), (count_mat.shape[0], 1))
    nfreqs = 1.0 * count_mat / divide_title
    multiply_title = np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat, axis=1)).reshape(-1, 1),
                             (1, count_mat.shape[1]))
    tfidf_mat = scipy.sparse.csr_matrix(np.multiply(nfreqs, multiply_title)).T
    return tfidf_mat, divide_title, multiply_title


from sklearn.feature_extraction.text import TfidfTransformer


def TFIDF_sklearn(count_mat):
    """
    TF-IDF transformation for matrix using sklearn's TfidfTransformer.

    Parameters
    ----------
    count_mat
        SciPy sparse matrix (CSR or CSC) or NumPy array with cells as rows and peaks as columns, shape (n_cells, n_peaks).

    Returns
    -------
    tfidf_mat
        Sparse matrix after TF-IDF transformation, shape (n_cells, n_peaks).
    """
    # 确保输入是稀疏矩阵
    if not scipy.sparse.issparse(count_mat):
        count_mat = scipy.sparse.csr_matrix(count_mat)

    # 初始化 TfidfTransformer
    # norm='l1' 对应原函数的 TF 归一化 (count_mat / rowSums)
    # smooth_idf=True 对应 log((n_cells + 1) / (doc_freq + 1)) + 1
    # sublinear_tf=False 保持线性 TF
    transformer = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True, sublinear_tf=False)

    # 应用 TF-IDF 变换
    tfidf_mat = transformer.fit_transform(count_mat)

    return tfidf_mat, 0, 0



def inverse_TFIDF(TDIDFed_mat, divide_title, multiply_title, max_temp):
    """
    Inversed TF-IDF transformation for matrix.

    Parameters
    ----------
    TDIDFed_mat: csr_matrix
        matrix after TFIDF transformation with peaks as rows and cells as columns, peak * cell.

    divide_title: numpy matrix
        matrix divided in TF-IDF transformation process, could get from "ATAC_data_preprocessing".

    multiply_title: numpy matrix
        matrix multiplied in TF-IDF transformation process, could get from "ATAC_data_preprocessing".

    max_temp: float
        max scale factor divided in ATAC preprocessing, could get from "ATAC_data_preprocessing".

    Returns
    ----------
    count_mat: csr_matrix
        recovered count matrix from matrix after TFIDF transformation.
    """

    count_mat = TDIDFed_mat.T
    count_mat = count_mat * max_temp
    nfreqs = np.divide(count_mat, multiply_title)
    count_mat = np.multiply(nfreqs, divide_title).T
    return count_mat
