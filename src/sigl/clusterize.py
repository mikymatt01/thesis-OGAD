from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from descriptor_functions import (
    degree_distribution,
    clustering_coefficient,
    normalised_laplacian_spectrum,
)

FEATURE_BLOCK_SIZE = 200
FEATURE_BLOCKS = [(0, 200), (200, 400), (400, 600)]


def ndarray_to_graph(A):
    return nx.from_numpy_array(A)


def graph_to_features(G):
    f1 = degree_distribution(G, n_bins=FEATURE_BLOCK_SIZE, density=True)
    f2 = clustering_coefficient(G, n_bins=FEATURE_BLOCK_SIZE, density=True)
    f3 = normalised_laplacian_spectrum(G, n_bins=FEATURE_BLOCK_SIZE, density=True)
    return np.concatenate([f1, f2, f3]).astype(np.float64)


def cluster_graphs_from_arrays(graph_arrays, n_clusters=100):
    features = [graph_to_features(ndarray_to_graph(A)) for A in graph_arrays]
    X = np.stack(features)

    # Per-block standardization so the Laplacian spectrum (typically higher
    # variance) does not drown the degree / clustering blocks.
    for lo, hi in FEATURE_BLOCKS:
        X[:, lo:hi] = StandardScaler().fit_transform(X[:, lo:hi])

    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(X)

    clusters = defaultdict(list)
    for graph, label in zip(graph_arrays, labels):
        clusters[label].append(graph)

    out = [clusters[k] for k in clusters]
    sizes = [len(c) for c in out]
    singletons = sum(1 for s in sizes if s < 2)
    if singletons:
        print(
            f"[clusterize] WARNING: {singletons}/{len(out)} singleton clusters; "
            f"Step1 will train on a single graph in those buckets. "
            f"Consider lowering n_clusters."
        )
    return out
