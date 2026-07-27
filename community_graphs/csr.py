from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse


def relabel_consecutively(graph: nx.Graph) -> nx.Graph:
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping, copy=True)


def graph_to_csr(
    graph: nx.Graph,
    *,
    dtype: np.dtype,
    binary: bool,
) -> sparse.csr_matrix:
    graph = relabel_consecutively(graph)
    node_order = list(range(graph.number_of_nodes()))
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=node_order,
        dtype=dtype,
        format="csr",
        weight="weight",
    )
    adjacency = sparse.csr_matrix(adjacency, dtype=dtype)
    if binary:
        adjacency.data[:] = 1
    normalize_csr(adjacency, dtype=dtype)
    if not isinstance(adjacency, sparse.csr_matrix):
        raise TypeError("Serialized CommunityGraphs matrices must be scipy.sparse.csr_matrix")
    return adjacency


def normalize_csr(adjacency: sparse.csr_matrix, *, dtype: np.dtype | None = None) -> sparse.csr_matrix:
    if not sparse.isspmatrix_csr(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    if dtype is not None and adjacency.dtype != np.dtype(dtype):
        adjacency = adjacency.astype(dtype, copy=False)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def sparse_is_symmetric(adjacency: sparse.csr_matrix) -> bool:
    return (adjacency - adjacency.T).nnz == 0


def has_self_loops(adjacency: sparse.csr_matrix) -> bool:
    return bool(np.any(adjacency.diagonal() != 0))


def is_binary(adjacency: sparse.csr_matrix) -> bool:
    if adjacency.nnz == 0:
        return True
    return bool(np.all(adjacency.data == 1))


def validate_csr_matrix(
    adjacency: Any,
    *,
    expected_binary: bool | None = None,
    expected_symmetric: bool | None = None,
    expected_self_loops: bool | None = None,
    index: int | None = None,
    normalize_explicit_zeros: bool = False,
) -> sparse.csr_matrix:
    label = f"Graph {index}" if index is not None else "Graph"
    if not isinstance(adjacency, sparse.csr_matrix):
        raise TypeError(f"{label} is {type(adjacency)!r}, expected scipy.sparse.csr_matrix")
    if len(adjacency.shape) != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"{label} is not square: shape={adjacency.shape}")
    if adjacency.shape[0] <= 0:
        raise ValueError(f"{label} has no nodes")
    if np.issubdtype(adjacency.dtype, np.floating):
        if not np.all(np.isfinite(adjacency.data)):
            raise ValueError(f"{label} contains NaN or infinite values")
    if len(adjacency.indptr) != adjacency.shape[0] + 1:
        raise ValueError(f"{label} has invalid indptr length")
    if adjacency.indptr[0] != 0 or adjacency.indptr[-1] != len(adjacency.indices):
        raise ValueError(f"{label} has inconsistent indptr")
    if np.any(np.diff(adjacency.indptr) < 0):
        raise ValueError(f"{label} has non-monotonic indptr")
    if adjacency.indices.size and (
        int(adjacency.indices.min()) < 0 or int(adjacency.indices.max()) >= adjacency.shape[1]
    ):
        raise ValueError(f"{label} contains column indices out of bounds")
    if not adjacency.has_sorted_indices:
        raise ValueError(f"{label} CSR indices are not sorted")
    explicit_zeros = adjacency.nnz > 0 and bool(np.any(adjacency.data == 0))
    if explicit_zeros:
        if normalize_explicit_zeros:
            adjacency = normalize_csr(adjacency.copy())
        else:
            raise ValueError(f"{label} contains explicit zero entries")
    if expected_binary is True and not is_binary(adjacency):
        raise ValueError(f"{label} is not binary with stored values equal to 1")
    if expected_symmetric is True and not sparse_is_symmetric(adjacency):
        raise ValueError(f"{label} is not symmetric")
    if expected_self_loops is False and has_self_loops(adjacency):
        raise ValueError(f"{label} contains self-loops")
    return adjacency
