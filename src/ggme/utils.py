"""Utility functions used by multiple scripts."""
import gzip
import bz2
import pickle
import re
#from io import BytesIO
from scipy.sparse import csr_matrix
import shutil
import numpy as np
import networkx as nx
import os

def get_open_fn(filepath):
    if filepath.endswith('.gz'):
        open_fn = gzip.open
    elif filepath.endswith('.bz2'):
        open_fn = bz2.open
    else:
        # Assume plain pickle
        open_fn = open
    return open_fn


def load_graphs(filepath):
    """Load graphs from eventually compressed pickle."""
    open_fn = get_open_fn(filepath)

    with open_fn(filepath, 'rb') as f:
        graphs = pickle.load(f)
    if isinstance(graphs[0], nx.Graph):
        pass
    elif isinstance(graphs[0], np.ndarray):
        graphs = adj_to_networkx(graphs)
    elif isinstance(graphs[0], csr_matrix):
        graphs = sparse_to_networkx(graphs)
    else:
        raise ValueError(
            'Unsupported input type.')
    return graphs


def adj_to_networkx(graphs):
    """Convert adj matrices to networkx graphs."""
    return [nx.from_numpy_array(g) for g in graphs]


def sparse_to_networkx(graphs):
    """Convert adj matrices to networkx graphs."""
    return [nx.from_scipy_sparse_matrix(g) for g in graphs]


def networkx_to_adj(graphs):
    """Convert networkx graphs to adjacency matrices."""
    return [nx.to_numpy_array(g) for g in graphs]


def load_digress_generated_samples(filepath):
    """
    Load graphs from DiGress generated_samples1.txt format.
    Each graph is stored as N=..., X: (node types), E: (n×n edge-type matrix).
    Returns list of networkx graphs (undirected, 0/1 adjacency: non-zero E entry = edge).
    """
    graphs = []
    with open(filepath, "r") as f:
        content = f.read()
    # Split into blocks by "N=" at start of line
    blocks = re.split(r"\n(?=N=)", content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines or not lines[0].startswith("N="):
            continue
        n = int(lines[0][2:].strip())
        # Find "X:" and "E:" and parse
        idx_x = next((i for i, ln in enumerate(lines) if ln == "X:"), None)
        idx_e = next((i for i, ln in enumerate(lines) if ln == "E:"), None)
        if idx_x is None or idx_e is None:
            continue
        # X line: next line after "X:"
        x_line = lines[idx_x + 1] if idx_x + 1 < len(lines) else ""
        # E lines: n lines after "E:"
        e_lines = lines[idx_e + 1 : idx_e + 1 + n] if idx_e + 1 + n <= len(lines) else []
        if len(e_lines) != n:
            continue
        # Parse E into n×n matrix of integers
        adj = np.zeros((n, n), dtype=np.float64)
        for i, eln in enumerate(e_lines):
            row = [int(x) for x in eln.split()]
            for j, val in enumerate(row):
                if j < n:
                    # Any non-zero edge type => edge present (binary adjacency)
                    adj[i, j] = 1 if val != 0 else 0
        # Undirected: symmetrise; zero diagonal
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)
        graphs.append(nx.from_numpy_array(adj))
    return graphs
