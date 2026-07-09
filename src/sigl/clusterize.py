from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from descriptor_functions import (
    degree_distribution,
    clustering_coefficient,
    normalised_laplacian_spectrum
)

# -------------------------------
# 1. ndarray -> graph
# -------------------------------
def ndarray_to_graph(A):
    return nx.from_numpy_array(A)


# -------------------------------
# 2. feature extraction
# -------------------------------
def graph_to_features(G):
    f1 = degree_distribution(G, density=True)
    f2 = clustering_coefficient(G, density=True)
    f3 = normalised_laplacian_spectrum(G, density=True)

    return np.concatenate([f1, f2, f3])


# -------------------------------
# 3. FULL PIPELINE
# -------------------------------
def cluster_graphs_from_arrays(graph_arrays, n_clusters=100, return_indices=False):
    features = []

    for A in graph_arrays:
        G = ndarray_to_graph(A)
        x = graph_to_features(G)
        features.append(x)
    
    max_len = max(len(x) for x in features)

    padded_features = []
    for x in features:
        x = np.pad(x, (0, max_len - len(x)))
        padded_features.append(x)
    
    X = np.stack(padded_features)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=0))
    ])

    labels = model.fit_predict(X)
    group_idx = defaultdict(list)
    original_idx = defaultdict(list)
    
    for index, (graph, label) in enumerate(zip(graph_arrays, labels)):
        group_idx[label].append(graph)
        original_idx[label].append(index)
        
    if return_indices:
        return [
            {
                "graphs": group_idx[key],
                "indices": original_idx[key],
            }
            for key in group_idx.keys()
        ]
    return [group_idx[key] for key in group_idx.keys()]
