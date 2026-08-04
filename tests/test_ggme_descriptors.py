from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ggme"))

from metrics.descriptor_functions import normalised_laplacian_spectrum  # noqa: E402


def test_normalised_laplacian_spectrum_handles_uint8_sparse_graphs():
    adjacency = sparse.csr_matrix(
        nx.to_numpy_array(nx.barabasi_albert_graph(100, 3, seed=0), dtype=np.uint8),
        dtype=np.uint8,
    )
    graph = nx.from_scipy_sparse_matrix(adjacency)

    hist = normalised_laplacian_spectrum(graph, density=True)

    assert hist.shape == (200,)
    assert np.isfinite(hist).all()
