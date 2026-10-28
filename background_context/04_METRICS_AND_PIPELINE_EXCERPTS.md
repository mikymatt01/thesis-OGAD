# Metrics And Pipeline Excerpts

This file contains compact excerpts from the repository behavior, rewritten for context efficiency.

## GGME Descriptor Functions

OGAD uses GGME-style descriptors to compare graph distributions.

### Degree Distribution

For each graph, the degree histogram is computed. This captures how node connectivity is distributed across the graph.

```python
def degree_distribution(G, density=False):
    hist = nx.degree_histogram(G)
    if density:
        hist = np.divide(hist, np.sum(hist))
    return np.asarray(hist)
```

### Clustering Coefficient

For each graph, local clustering coefficients are computed and summarized as a histogram over `[0, 1]`.

```python
def clustering_coefficient(G, n_bins=200, density=False):
    coefficient_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        coefficient_list,
        bins=n_bins,
        range=(0.0, 1.0),
        density=density,
    )
    return hist
```

### Normalized Laplacian Spectrum

For each graph, the eigenvalues of the normalized Laplacian are computed and summarized as a histogram over `[0, 2]`.

```python
def normalised_laplacian_spectrum(G, n_bins=200, bin_range=(0, 2), density=False):
    graph = nx.from_numpy_array(nx.to_numpy_array(G, dtype=float, weight="weight"))
    spectrum = nx.normalized_laplacian_spectrum(graph)
    hist, _ = np.histogram(spectrum, bins=n_bins, density=density, range=bin_range)
    return hist
```

## MMD And Kernels

The evaluation compares two graph distributions by applying descriptor functions to graphs and then comparing the resulting descriptor distributions. Maximum Mean Discrepancy is used with kernels such as Gaussian or linear kernels. The Background should explain MMD conceptually rather than reproduce implementation details.

## Pipeline Execution Logic

The repository pipeline can be summarized as:

1. prepare graph datasets;
2. run SIGL to estimate graphons from training graphs;
3. train/sample D3PM over graphon-like tensors;
4. optionally run DiGress as a direct graph generation baseline;
5. optionally run classical graph baselines;
6. evaluate generated graphs with GGME against test graphs.

## Experiment Grid Logic

Experiment configurations derive stage-specific paths from a canonical dataset object. Important conceptual fields are:

- `train_graphs_path`: input training graphs;
- `test_graphs_path`: reference/test graphs for evaluation;
- `sigl_graphons_dir`: graphons estimated from training graphs;
- `ogad_generated_graphons_path`: D3PM-generated graphon-like tensors;
- `digress_generated_samples_path`: DiGress-generated graphs;
- `classical_baselines`: paths to baseline graph samples;
- `generated_graph_count`: number of generated graphs evaluated.

These details are useful for explaining the flow of data in the thesis, but they should not dominate the Background chapter.
