import sys
import numpy as np
from torchtext.vocab import Vocab
import torch
from typing import Dict, Iterable, List, Optional, Tuple, Union

def tokenize_and_pad_batch(
    data: np.ndarray,  # Gene expression matrix.
    gene_ids: np.ndarray,  # Gene ID array corresponding to the features in data.
    max_len: int,  # Maximum sequence length used for padding or truncation.
    vocab: Vocab,  # Maps gene IDs and special tokens such as <cls> and <pad> to indices.
    pad_token: str,  # Padding token.
    pad_value: int,  # Padding value.
    append_cls: bool = True,  # Whether to append the <cls> token to the sequence.
    include_zero_gene: bool = False,  # Whether to retain genes with zero expression.
    cls_token: str = "<cls>",  # Classification token.
    return_pt: bool = True,  # Whether to return PyTorch tensors.
    mod_type: np.ndarray = None,  # Optional modality type array.
    vocab_mod: Vocab = None,  # Optional vocabulary that maps modality types to indices.
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]

    # Tokenize each cell into gene-expression pairs, optionally retaining zeros and appending <cls>.
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    # Pad tokenized sequences to max_len so all model inputs in the batch have equal length.
    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_id=cls_id,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
            if mod_type is not None:
                mod_types = mod_type
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
            if mod_type is not None:
                mod_types = mod_type[idx]
        if append_cls:
            # genes = np.insert(genes, 0, cls_id)
            genes = np.append(genes, cls_id)   # Append the CLS token.
            # values = np.insert(values, 0, 0)
            values = np.append(values, 0)  # Append a zero value for the CLS token.
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((genes, values, mod_types))

    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
    cls_id:int = 0
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    print('Original sequence length: ', max_ori_len)
    print('Configured sequence length: ', max_len)
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            # sample max_len genes
            if not cls_appended:
                # idx = np.random.choice(len(gene_ids), max_len, replace=False)
                # idx = np.sort(idx)

                idx = np.arange(max_len)  # Generate indices from 0 to max_len - 1.

            else:
                # idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                # idx = np.sort(idx)
                # Randomly select max_len - 1 unique indices from 0 to len(gene_ids) - 2.

                idx = np.arange(max_len - 1)  # Generate indices from 0 to max_len - 2.

                # idx = idx + 1  # Shift indices by one to reserve index 0 for the <cls> token.
                # idx = np.insert(idx, 0, 0)  # Include the <cls> token stored at gene_ids[0].
                idx = np.append(idx, len(gene_ids) - 1)  # Append the <cls> token index.
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded
