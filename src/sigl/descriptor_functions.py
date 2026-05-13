"""Provide evaluation functions and auxiliary functionality."""

import numpy as np
import networkx as nx
# import igraph as ig

# from metrics.kernels import KernelDistributionWrapper

# from metrics.mmd import mmd
# from metrics.mmd import mmd_linear_approximation

# from metrics.utils import ensure_padded


def degree_distribution(G, n_bins=200, density=False, **kwargs):
    """Fixed-length histogram of normalized degrees d/(n-1) in [0, 1]."""
    n = G.number_of_nodes()
    if n <= 1:
        return np.zeros(n_bins, dtype=float)
    degrees = np.fromiter((d for _, d in G.degree()), dtype=float, count=n) / (n - 1)
    hist, _ = np.histogram(degrees, bins=n_bins, range=(0.0, 1.0), density=density)
    return hist


def clustering_coefficient(G, n_bins=200, density=False, **kwargs):
    """Calculate clustering coefficient histogram of a graph."""
    coefficient_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        coefficient_list, bins=n_bins, range=(0.0, 1.0), density=density
    )

    return hist


def normalised_laplacian_spectrum(
    G, n_bins=200, bin_range=(0, 2), density=False, **kwargs
):
    """Calculate normalised Laplacian spectrum of a graph."""
    spectrum = nx.normalized_laplacian_spectrum(G)
    hist, _ = np.histogram(
        spectrum, bins=n_bins, density=density, range=bin_range
    )

    return hist

