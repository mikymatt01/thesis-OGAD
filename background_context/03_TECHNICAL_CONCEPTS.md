# Technical Concepts To Cover

This file lists the technical concepts that must be clear before the reader reaches the Method and Experiments chapters.

## Basic Graph Theory

An undirected graph can be represented as `G = (V, E)`, where `V` is a set of nodes and `E` is a set of unordered node pairs. For a graph with `n` nodes, the adjacency matrix `A in {0,1}^{n x n}` has entry `A_ij = 1` if there is an edge between nodes `i` and `j`, and `A_ij = 0` otherwise. For undirected graphs, `A` is symmetric.

Relevant graph statistics:

- node degree: number of incident edges;
- degree distribution: distribution or histogram of node degrees;
- graph density: fraction of possible edges that are present;
- clustering coefficient: local tendency of nodes to form triangles;
- graph Laplacian: matrix representation of graph connectivity;
- normalized Laplacian spectrum: eigenvalue distribution of the normalized Laplacian, used as a structural descriptor.

## Graph Generation

Graph generation means sampling new graphs from a learned or specified graph distribution. Unlike images, graphs are discrete, unordered, and often sparse. Node labels are usually arbitrary, so the same graph structure can be represented by many permuted adjacency matrices. This creates challenges for learning and evaluation.

The thesis evaluates whether generated graphs match the distribution of target/test graphs, not whether a generated graph matches one specific graph.

## Graphons

A graphon is commonly defined as a symmetric measurable function:

```text
W: [0,1]^2 -> [0,1]
```

Intuitively, `W(u, v)` gives the probability of an edge between two latent positions `u` and `v`. A graph can be sampled by drawing latent positions for nodes and then sampling edges according to the graphon values.

Examples:

- Erdos-Renyi corresponds to a constant graphon;
- SBM corresponds to a block-structured graphon;
- expected-degree models can correspond to smoother graphon-like structures.

Graphons matter in OGAD because they provide a continuous intermediate representation between observed discrete graphs and generative modeling.

## Graphon Estimation

Graphon estimation attempts to recover or approximate an underlying graphon from finite observed graphs. This is difficult because:

- graphs are finite and noisy samples;
- node ordering is arbitrary;
- graphons are not identifiable up to measure-preserving transformations;
- sparse graphs and heterogeneous degrees can complicate estimation.

SIGL is relevant because it estimates graphons using scalable implicit graphon learning. For Background, explain the estimation problem and the role of implicit/neural representations without detailing the full SIGL algorithm.

## Diffusion Models And D3PM

Diffusion models generate data by learning to reverse a corruption process. The forward process gradually adds noise; the reverse process learns to denoise and sample new data.

D3PM is a discrete-state diffusion model. It is relevant because OGAD treats graphon-like tensors as discretized objects that can be modeled with a discrete diffusion process.

For the Background, explain:

- forward noising;
- reverse denoising;
- continuous vs discrete diffusion;
- why discrete diffusion can be used for discretized graphon representations.

## Evaluation With GGME

GGME evaluates graph generative models by comparing graph distributions through descriptors and kernels. The key idea is to compute structural descriptors for graphs and then compare the descriptor distributions between generated graphs and reference/test graphs.

Relevant descriptors in OGAD:

- degree distribution;
- clustering coefficient histogram;
- normalized Laplacian spectrum histogram.

Relevant statistical concept:

- Maximum Mean Discrepancy, a distance between distributions induced by a kernel.

This justifies explaining distributional evaluation in the Background.
