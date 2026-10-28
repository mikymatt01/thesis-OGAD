# OGAD Thesis Context Summary

This file summarizes the minimum repository context needed to write the Background chapter.

## High-Level Thesis Idea

OGAD studies graph generation through an intermediate graphon representation. Instead of generating graphs directly only in the discrete graph domain, the pipeline first estimates graphons from observed training graphs, then trains a discrete diffusion model to generate new graphon-like tensors, and finally converts or samples those generated representations into graphs. The generated graph distributions are evaluated against test graphs and baseline generators.

## Conceptual Pipeline

```text
training graphs
  -> graphon estimation with SIGL
  -> D3PM generation of graphon-like tensors
  -> conversion or sampling into graphs
  -> GGME distributional evaluation
  -> comparison with DiGress and classical baselines
```

## Main Components

### SIGL

SIGL stands for Scalable Implicit Graphon Learning. In OGAD it is used to estimate graphons from observed graphs. The important Background concept is not the exact implementation of SIGL, but the problem it addresses: learning a continuous graphon representation from finite, discrete, permuted adjacency matrices.

### D3PM

D3PM stands for Discrete Denoising Diffusion Probabilistic Model. In OGAD it is used to generate graphon-like tensors. The important Background concept is discrete diffusion: a forward process corrupts discrete states, while a learned reverse process denoises them to generate new samples.

### GGME

GGME stands for Graph Generative Model Evaluation. In OGAD it is used to compare graph distributions through structural descriptors and Maximum Mean Discrepancy. The Background should explain why graph generation is evaluated distributionally.

### DiGress

DiGress is used as a graph diffusion baseline. The Background may mention it only as motivation for graph generation and graph diffusion; detailed discussion belongs in Related Works.

### Classical Baselines

The pipeline compares against classical graph models such as Erdos-Renyi, Configuration Model, Stochastic Block Model, and Degree-Corrected SBM. These should be introduced as probabilistic graph models and baselines.

## What The Background Should Teach

The reader should understand:

- what graphs are and how they are represented;
- which structural graph statistics matter for this thesis;
- why graph generation is a distribution-learning problem;
- what graphons are and why they can bridge discrete graphs and continuous representations;
- why graphon estimation is challenging;
- why discrete diffusion can generate discretized graphon representations;
- how graph generators are evaluated using descriptors, kernels, and MMD.

## What Should Stay Out Of The Background

Avoid:

- a detailed review of SIGL, D3PM, DiGress, or GGME papers;
- implementation details such as Docker, exact command-line flags, or path handling;
- experimental results;
- detailed comparison of previous methods.

Those details belong in Related Works, Method, or Experiments.
