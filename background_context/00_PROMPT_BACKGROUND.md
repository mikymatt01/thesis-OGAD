# Prompt: Write The Thesis Background Chapter

I need to write the "Background" chapter of my thesis in English.

The thesis is based on OGAD, a repository implementing a pipeline for graph generation and evaluation through graphons. Use the context files provided with this prompt. The goal is to write a foundational Background chapter, not a Related Works chapter.

## Thesis Context

OGAD works on graph datasets. It estimates graphons from observed training graphs using SIGL, generates new graphon-like representations using a D3PM model, converts or samples those representations into graphs, and evaluates the generated graph distributions using GGME. The thesis also compares against DiGress and classical graph-generation baselines such as Erdos-Renyi, Configuration Model, Stochastic Block Model, and Degree-Corrected SBM.

The datasets include synthetic graph families such as Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, Stochastic Block Model, and expected-degree graphs.

## Scope

Write a Background chapter that explains the concepts needed to understand the thesis. Do not write a literature survey. The Related Works chapter will be separate.

You may mention SIGL, D3PM, DiGress, and GGME to motivate concepts, but avoid detailed paper comparison. Keep the focus theoretical, introductory, and foundational.

## Suggested Structure

### 2.1 Basic Graph Theory

Explain undirected graphs, nodes, edges, adjacency matrices, node degree, degree distribution, graph density, clustering coefficient, graph Laplacian, and normalized Laplacian spectrum.

### 2.2 Probabilistic Graph Models

Explain Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, Stochastic Block Model, Degree-Corrected SBM, Configuration Model, and expected-degree graphs. Clarify their role as synthetic generators, baselines, or case studies.

### 2.3 The Problem Of Graph Generation

Explain what graph generation means, why graphs differ from regular data such as images or sequences, and why graph generation can be seen as learning a distribution over graphs. Discuss node permutation, discreteness, sparsity, variable structure, and structural constraints.

### 2.4 Graphons

Give an intuitive and, if useful, formal definition. Explain graphons as continuous representations of graph distributions, their relation to adjacency matrices, and how graphs can be sampled from graphons. Include examples: constant graphons for Erdos-Renyi graphs, block graphons for SBM, and smooth graphons for expected-degree graphs. Explain why graphons are useful as an intermediate representation in this thesis.

### 2.5 Graphon Estimation From Observed Graphs

Explain graphon estimation, latent positions or node ordering, finite-graph noise, and non-identifiability under node permutations. Introduce the general idea of neural or implicit neural representations. Connect this conceptually to SIGL without turning it into a method chapter.

### 2.6 Diffusion Models And Discrete Generation

Explain diffusion models, the forward noising process, reverse denoising, and the difference between continuous diffusion and discrete-state diffusion. Explain why D3PM is coherent with generating discretized graphons or graphon-like tensors.

### 2.7 Evaluation Of Graph Generative Models

Explain why evaluation is difficult, especially because graph generators should match graph distributions rather than individual graphs. Discuss structural descriptors such as degree distribution, clustering coefficient, and normalized Laplacian spectrum. Explain Maximum Mean Discrepancy and the role of kernels.

### 2.8 Conceptual Overview Of The Thesis Pipeline

Describe the overall logic:

training graphs -> graphon estimation -> generation of new graphons -> conversion/sampling into graphs -> distributional evaluation -> comparison against baselines and alternative models.

This section should bridge toward Related Works, Method, and Experiments.

## Requirements

- Write in clear academic English.
- Do not invent experimental results.
- Do not invent bibliographic references.
- Mark missing references as `[citation needed]`.
- Keep the focus on Background, not Method.
- Avoid an overly popular-science tone.
- Explain why each concept matters for this specific thesis.
- Use mathematical notation only when useful.
- If formulas are introduced, explain all symbols.

## Requested Output

First produce:

1. a detailed outline of the chapter;
2. the communicative goal of each section;
3. a list of places where bibliographic citations are needed;
4. then a complete Background chapter draft with numbered sections and subsections.
