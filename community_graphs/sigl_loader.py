from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

from .csr import validate_csr_matrix


def _load_sibling_metadata(path: Path) -> dict | None:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_community_graphs_sparse(path: str | Path) -> list[sparse.csr_matrix]:
    """Carica e valida il formato CommunityGraphs senza densificarlo.

    Warning: pickle can execute code while loading. Only load trusted datasets.
    """

    path = Path(path)
    with path.open("rb") as file:
        graphs = pickle.load(file)
    if not isinstance(graphs, list):
        raise TypeError(f"{path} must contain a Python list, got {type(graphs)!r}")
    metadata = _load_sibling_metadata(path)
    adjacency_meta = (metadata or {}).get("adjacency", {})
    expected_binary = adjacency_meta.get("binary")
    expected_symmetric = adjacency_meta.get("symmetric")
    expected_self_loops = adjacency_meta.get("self_loops")
    validated: list[sparse.csr_matrix] = []
    for index, graph in enumerate(graphs):
        validated.append(
            validate_csr_matrix(
                graph,
                expected_binary=expected_binary,
                expected_symmetric=expected_symmetric,
                expected_self_loops=expected_self_loops,
                index=index,
            )
        )
    return validated


def densify_graph(
    adjacency: sparse.csr_matrix,
    *,
    dtype: np.dtype = np.float32,
    max_dense_bytes: int | None = None,
) -> np.ndarray:
    """Converte una CSR in ndarray C-contiguous con controllo memoria."""

    dtype = np.dtype(dtype)
    n = int(adjacency.shape[0])
    required_bytes = n * n * dtype.itemsize
    if max_dense_bytes is not None and required_bytes > max_dense_bytes:
        required_gib = required_bytes / (1024**3)
        limit_mib = max_dense_bytes / (1024**2)
        raise MemoryError(
            f"Graph requires {required_gib:.2f} GiB when converted to {dtype.name} dense, "
            f"exceeding the configured limit of {limit_mib:.0f} MiB."
        )
    dense = adjacency.toarray().astype(dtype, copy=False)
    return np.ascontiguousarray(dense)


def load_community_graphs_for_sigl(
    path: str | Path,
    *,
    dtype: np.dtype = np.float32,
    max_dense_bytes_per_graph: int | None = None,
) -> list[np.ndarray]:
    """
    Carica una lista di CSR e restituisce una lista di matrici NumPy dense
    adatta al codice SIGL.
    """

    graphs = load_community_graphs_sparse(path)
    return [
        densify_graph(graph, dtype=dtype, max_dense_bytes=max_dense_bytes_per_graph)
        for graph in graphs
    ]
