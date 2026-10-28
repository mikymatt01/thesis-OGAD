# OGAD — On Graphons And Diffusion

OGAD is an experimental pipeline for **graph generation through graphon representations and discrete diffusion**. Starting from a collection of observed graphs, it estimates finite graphon representations, learns their distribution with a categorical diffusion model, and samples new undirected graphs from the generated edge-probability matrices.

This repository contains the complete research workflow used by the accompanying thesis: dataset utilities, graphon estimation, diffusion training and inference, graph sampling, model selection, evaluation, baselines, experiment orchestration, and result collection.

The thesis is available in [`thesis/`](thesis/). The full mathematical and implementation-level description of OGAD is in [`thesis/chapters/method.tex`](thesis/chapters/method.tex).

## Method overview

A graphon is a symmetric function $W:[0,1]^2 \to [0,1]$ that describes edge probabilities independently of a particular finite adjacency matrix. OGAD uses finite graphon grids as an intermediate representation between observed graphs and the generative model:

```text
training graphs
  → SIGL graphon estimation
  → finite probability grids
  → categorical quantization
  → D3PM training and generation
  → dequantized graphon-like grids
  → Bernoulli edge sampling
  → generated graphs
  → GGME selection and evaluation
```

### 1. Graphon estimation with SIGL

[SIGL](src/sigl/) learns a latent ordering of the nodes, pools the reordered adjacency matrices, and fits an implicit SIREN representation of the graphon. The function is evaluated on a fixed grid, symmetrized, and stored with a zero diagonal.

Graphs can first be grouped by structural descriptors—degree distributions, clustering coefficients, and normalized-Laplacian spectra—and bootstrap samples can be created within each cluster. This produces multiple graphon-estimation jobs from the training distribution.

### 2. Discrete diffusion with D3PM

[D3PM](src/d3pm/) quantizes every graphon-grid value into one of $K$ categorical states. A timestep-conditioned Transformer then learns the reverse process of a uniform categorical diffusion model. Generated categorical tensors are dequantized, symmetrized, clipped to $[0,1]$, and interpreted as graphon-like probability grids.

The implementation supports hyperparameter grids and early stopping. Each candidate keeps its best graphon-validation checkpoint; when graph-level selection is enabled, GGME subsequently selects the candidate whose sampled graphs best match a held-out graph set.

### 3. Sampling finite graphs

Generated probability grids are not treated as adjacency matrices. For every output graph, OGAD:

1. samples a node count from the empirical reference distribution;
2. samples one latent grid position per node;
3. reads the corresponding edge probabilities from the generated grid;
4. draws each undirected edge independently from a Bernoulli distribution.

The procedure preserves symmetry and removes self-loops, but does not force connectivity or remove isolated nodes.

### 4. Selection and evaluation with GGME

[GGME](src/ggme/) compares generated and reference graph distributions using Maximum Mean Discrepancy (MMD) over three structural descriptors:

- node-degree distribution;
- local clustering-coefficient distribution;
- normalized-Laplacian spectral distribution.

Kernel and bandwidth choices are calibrated with controlled graph perturbations. GGME is used both for D3PM candidate selection and for the final test-set evaluation; the final test set does not participate in model selection.

## Baselines

The same evaluation workflow can compare OGAD with:

- [DiGress](src/baselines/DiGress/), a direct discrete graph-diffusion model;
- Erdős–Rényi;
- the Configuration Model;
- the Stochastic Block Model (SBM);
- the Degree-Corrected Stochastic Block Model (DC-SBM).

## Repository layout

```text
community_graphs/         synthetic dataset generation and serialization
experiments/              JSON experiment configurations
scripts/                  orchestration, parallel runs, cleanup and reports
src/sigl/                 graphon estimation
src/d3pm/                 diffusion training, selection and inference
src/ggme/                 graph sampling and distributional evaluation
src/baselines/            DiGress and classical graph generators
tests/                    unit and integration tests
runs/                     experiment artifacts and logs
runs_export/              collected metrics and reports
thesis/                   thesis source and experimental analysis
```

## Input data

The canonical dataset format is a trusted Python pickle containing a list of `scipy.sparse.csr_matrix` adjacency matrices. OGAD expects finite, simple, undirected graphs. Sparse matrices are converted to dense floating-point matrices only when required by SIGL.

> [!WARNING]
> Python pickle files can execute code while loading. Only use datasets from trusted sources.

Datasets are intentionally excluded from version control. Their container paths and all cross-stage artifact paths are declared in the `dataset` block of an experiment JSON file. Existing configurations in [`experiments/`](experiments/) show the complete schema.

Because adjacency matrices are densified during graphon estimation, memory use grows quadratically with graph size. The graphon-grid resolution must also accommodate the reference node counts used during sampling: node counts larger than the grid are rejected and resampled.

### Generating synthetic data

The bundled generator supports fixed or variable graph sizes and several NetworkX graph families. List all available families with:

```bash
python generate_community_graphs.py --list-families
```

Example:

```bash
python generate_community_graphs.py \
  --family erdos-renyi \
  --instances 300 \
  --nodes 100 \
  --params p=0.05 \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir datasets/er_n100_p005
```

This creates reproducible train/test `CommunityGraphs` pickle files and their metadata.

## Experiment configuration

Experiment JSON files connect all pipeline stages. Their main sections are:

- `dataset`: input datasets, generated artifacts, and output graph count;
- `sigl`: graphon resolution, optimization, clustering, and bootstrap settings;
- `d3pm`: model dimensions and candidate hyperparameter grids;
- `d3pm_training`: validation split, early stopping, and epoch limit;
- `graph_selection`: held-out graph split and candidate-selection rule;
- `digress` and `baselines`: optional comparison models;
- `ggme`: evaluation settings and random seed.

The runner normalizes these values into a per-run `experiment.json` and records both the original and resolved configuration in `manifest.json`.

## Running an experiment

The complete pipeline is designed for:

- Linux and Bash;
- Docker Engine with Docker Compose v2;
- an NVIDIA GPU exposed to Docker through the NVIDIA Container Toolkit;
- Python 3.10 for local dataset utilities and tests.

Build or refresh the component images with:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.ogad-only.yml \
  build
```

Run one experiment:

```bash
scripts/run_ogad_experiment.sh experiments/<config>.json [run-name]
```

The orchestrator executes SIGL, optional DiGress and classical baselines, D3PM training and inference, and GGME evaluation. Set `OGAD_GPU_ID` to choose the GPU:

```bash
OGAD_GPU_ID=1 scripts/run_ogad_experiment.sh experiments/<config>.json my-run
```

If `experiments/` contains only scalar configurations, run every top-level JSON file with:

```bash
scripts/run_all_experiments.sh
```

Merged configurations can define multiple datasets and SIGL parameter combinations. Preview or execute their expansion across multiple GPUs with:

```bash
python scripts/run_from_merged_parallel.py \
  --merged experiments/test_merged.json \
  --gpus 0,1 \
  --dry-run
```

Remove `--dry-run` to launch the jobs. By default, compatible DiGress and classical-baseline outputs are shared across jobs for the same dataset.

Each execution creates an isolated directory under `runs/<run-id>/` containing the normalized experiment configuration, manifest, stage logs, checkpoints, estimated and generated graphons, generated graphs, and evaluation results.

## Tests

The root test suite covers dataset handling, graph generators, descriptor computation, and pipeline helpers:

```bash
python -m pytest
```

The lightweight local test dependencies are listed in [`requirements.txt`](requirements.txt); the full research environment is described by [`pyproject.toml`](pyproject.toml) and the component-specific Docker files.
