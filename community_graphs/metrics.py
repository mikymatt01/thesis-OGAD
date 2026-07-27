from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse


def basic_graph_stats(graph: nx.Graph, adjacency: sparse.csr_matrix) -> dict[str, Any]:
    n = int(adjacency.shape[0])
    directed = graph.is_directed()
    if directed:
        edges = int(graph.number_of_edges())
        density = float(nx.density(graph))
        connected = bool(nx.is_weakly_connected(graph)) if n else False
        components = list(nx.weakly_connected_components(graph)) if n else []
    else:
        edges = int(adjacency.nnz // 2)
        density = float((2 * edges) / (n * (n - 1))) if n > 1 else 0.0
        connected = bool(nx.is_connected(graph)) if n else False
        components = list(nx.connected_components(graph)) if n else []
    degrees = np.array([degree for _, degree in graph.degree()], dtype=np.float64)
    if degrees.size == 0:
        degrees = np.array([0.0])
    return {
        "num_nodes": n,
        "num_edges": edges,
        "nnz": int(adjacency.nnz),
        "density": density,
        "degree_min": float(degrees.min()),
        "degree_max": float(degrees.max()),
        "degree_mean": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "connected": connected,
        "num_components": int(len(components)),
        "largest_component_size": int(max((len(c) for c in components), default=0)),
    }


def expensive_graph_stats(graph: nx.Graph) -> dict[str, Any]:
    undirected = graph.to_undirected() if graph.is_directed() else graph
    out: dict[str, Any] = {}
    out["average_clustering"] = float(nx.average_clustering(undirected)) if undirected.number_of_nodes() else 0.0
    out["transitivity"] = float(nx.transitivity(undirected))
    try:
        assortativity = nx.degree_assortativity_coefficient(undirected)
        out["assortativity"] = None if math.isnan(assortativity) else float(assortativity)
    except Exception:
        out["assortativity"] = None
    return out
