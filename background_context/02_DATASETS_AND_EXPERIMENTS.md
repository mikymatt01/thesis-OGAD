# Datasets And Experimental Context

This file contains the minimal dataset and experiment information needed for the Background chapter.

## Dataset Format

OGAD uses graph datasets stored in a CommunityGraphs-compatible format. The canonical persistent format is a pickle containing a Python list of `scipy.sparse.csr_matrix` adjacency matrices. Sparse CSR storage reduces disk usage, although some SIGL stages densify matrices graph by graph.

For the Background chapter, the important point is that the input data are sets of graphs represented through adjacency matrices.

## Validation Dataset Suite

The repository includes a validation suite of synthetic graph datasets. Each dataset contains approximately 300 graphs, split 80/20 into training and test sets, with seed 42 and typically `n = 100` nodes.

| Dataset | Graph family | Purpose |
|---|---|---|
| `er_n100_p005` | Erdos-Renyi | Constant graphon ground-truth; degree behavior |
| `sbm_n100_4c` | Stochastic Block Model with 4 communities | Block graphon ground-truth; community and spectral structure |
| `expdeg_n100_lin` | Expected-degree graph | Smooth rank-1 graphon-like structure |
| `ba_n100_m3` | Barabasi-Albert | Scale-free stress case; heavy-tailed degree behavior |
| `ws_n100_k8_p01` | Watts-Strogatz | Small-world structure; clustering behavior |

## Probabilistic Graph Models To Explain

The Background should introduce:

- Erdos-Renyi graphs, where edges are sampled independently with a fixed probability;
- Barabasi-Albert graphs, which model preferential attachment and scale-free degree distributions;
- Watts-Strogatz graphs, which model small-world structure and high clustering;
- Stochastic Block Models, which model community structure through block-dependent edge probabilities;
- Degree-Corrected SBM, which extends SBM by allowing heterogeneous expected degrees within communities;
- Configuration Model, which preserves or matches a degree sequence;
- expected-degree graphs, where edge probabilities are driven by node-specific expected-degree parameters.

## Minimal Experiment Configuration

The representative experiment `test_merged.json` includes:

- dataset: `community_graphs`;
- training graphs path: `/app/datasets/community_graphs/CommunityGraphs_train.pkl`;
- test graphs path: `/app/datasets/community_graphs/CommunityGraphs_test.pkl`;
- generated graph count: 100;
- SIGL parameters including graphon resolution `Res = 128`, number of clusters `n_clusters = 1`, and bootstrap replicates;
- D3PM parameters including batch size, model dimension, number of discrete classes, image size, and training settings;
- graph selection using a validation split;
- DiGress enabled as an alternative graph generator;
- classical baselines enabled: Erdos-Renyi, Configuration Model, SBM, and Degree-Corrected SBM;
- GGME used for evaluation.

## Background Relevance

These datasets justify the Background sections on probabilistic graph models and graphons. ER corresponds naturally to a constant graphon, SBM to a block-structured graphon, expected-degree graphs to smoother graphon-like structures, while BA and WS act as structurally different stress cases.
