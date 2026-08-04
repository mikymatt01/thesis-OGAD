from __future__ import annotations

import bz2
import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from scipy import sparse


MODEL_CHOICES = (
    "erdos_renyi",
    "configuration_model",
    "sbm",
    "degree_corrected_sbm",
)


@dataclass(frozen=True)
class SBMTemplate:
    block_probs: np.ndarray
    block_sizes: np.ndarray
    block_weights: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FittedClassicalModel:
    name: str
    edge_probability: float = 0.0
    degree_pool: tuple[int, ...] = ()
    sbm_templates: tuple[SBMTemplate, ...] = ()


def _open_pickle(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def load_graphs(path: str | Path) -> list[nx.Graph]:
    path = Path(path)
    with _open_pickle(path) as file:
        graphs = pickle.load(file)
    if not graphs:
        raise ValueError(f"No graphs found in {path}")
    converted = [_to_simple_graph(graph) for graph in graphs]
    if not converted:
        raise ValueError(f"No usable graphs found in {path}")
    return converted


def save_graphs(graphs: Iterable[nx.Graph], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(list(graphs), file, protocol=pickle.HIGHEST_PROTOCOL)


def _to_simple_graph(graph) -> nx.Graph:
    if isinstance(graph, nx.Graph):
        simple = nx.Graph(graph)
    elif isinstance(graph, np.ndarray):
        simple = nx.from_numpy_array(np.asarray(graph))
    elif sparse.issparse(graph):
        if hasattr(nx, "from_scipy_sparse_array"):
            simple = nx.from_scipy_sparse_array(graph)
        else:
            simple = nx.from_scipy_sparse_matrix(graph)
    else:
        raise ValueError(f"Unsupported graph type: {type(graph)!r}")
    simple.remove_edges_from(nx.selfloop_edges(simple))
    return nx.convert_node_labels_to_integers(simple, first_label=0)


def _sanitize_graph(graph: nx.Graph) -> nx.Graph:
    simple = nx.Graph(graph)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    return nx.convert_node_labels_to_integers(simple, first_label=0)


def _sample_sizes(test_graphs: list[nx.Graph], num_graphs: int, rng: np.random.Generator) -> list[int]:
    if num_graphs < 1:
        raise ValueError("num_graphs must be at least 1")
    sizes = [graph.number_of_nodes() for graph in test_graphs]
    if not sizes or min(sizes) < 1:
        raise ValueError("test_graphs must contain at least one non-empty graph")
    return [int(rng.choice(sizes)) for _ in range(num_graphs)]


def _fit_erdos_renyi(train_graphs: list[nx.Graph]) -> FittedClassicalModel:
    possible_edges = sum(graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2 for graph in train_graphs)
    observed_edges = sum(graph.number_of_edges() for graph in train_graphs)
    p = 0.0 if possible_edges == 0 else observed_edges / possible_edges
    return FittedClassicalModel(name="erdos_renyi", edge_probability=float(np.clip(p, 0.0, 1.0)))


def _fit_configuration_model(train_graphs: list[nx.Graph]) -> FittedClassicalModel:
    degrees = tuple(int(degree) for graph in train_graphs for _, degree in graph.degree())
    if not degrees:
        degrees = (0,)
    return FittedClassicalModel(name="configuration_model", degree_pool=degrees)


def _communities_for(graph: nx.Graph) -> list[list[int]]:
    if graph.number_of_nodes() <= 1 or graph.number_of_edges() == 0:
        return [list(graph.nodes())]
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    ordered = sorted((sorted(c) for c in communities), key=lambda c: (-len(c), c[0] if c else -1))
    return [community for community in ordered if community]


def _estimate_sbm_template(graph: nx.Graph) -> SBMTemplate:
    communities = _communities_for(graph)
    node_to_block = {
        node: block_index
        for block_index, community in enumerate(communities)
        for node in community
    }
    block_sizes = np.asarray([len(community) for community in communities], dtype=np.int64)
    edge_counts = np.zeros((len(block_sizes), len(block_sizes)), dtype=np.float64)
    for u, v in graph.edges():
        bu = node_to_block[u]
        bv = node_to_block[v]
        edge_counts[bu, bv] += 1.0
        if bu != bv:
            edge_counts[bv, bu] += 1.0
    possible = np.zeros_like(edge_counts)
    for i, size_i in enumerate(block_sizes):
        for j, size_j in enumerate(block_sizes):
            if i == j:
                possible[i, j] = size_i * (size_i - 1) / 2
            else:
                possible[i, j] = size_i * size_j
    block_probs = np.divide(edge_counts, possible, out=np.zeros_like(edge_counts), where=possible > 0)
    block_probs = np.clip(block_probs, 0.0, 1.0)
    weights = []
    for community in communities:
        degrees = np.asarray([max(0, graph.degree(node)) for node in community], dtype=np.float64)
        mean_degree = float(degrees.mean()) if degrees.size else 0.0
        if mean_degree <= 0.0:
            weights.append(np.ones(max(1, len(community)), dtype=np.float64))
        else:
            weights.append(np.clip(degrees / mean_degree, 0.05, 5.0))
    return SBMTemplate(
        block_probs=block_probs,
        block_sizes=block_sizes,
        block_weights=tuple(weights),
    )


def _fit_sbm(train_graphs: list[nx.Graph], name: str) -> FittedClassicalModel:
    templates = tuple(_estimate_sbm_template(graph) for graph in train_graphs if graph.number_of_nodes() > 0)
    if not templates:
        raise ValueError("Cannot fit SBM baseline without non-empty training graphs")
    return FittedClassicalModel(name=name, sbm_templates=templates)


def fit_model(model_name: str, train_graphs: list[nx.Graph]) -> FittedClassicalModel:
    train_graphs = [_sanitize_graph(graph) for graph in train_graphs]
    if model_name == "erdos_renyi":
        return _fit_erdos_renyi(train_graphs)
    if model_name == "configuration_model":
        return _fit_configuration_model(train_graphs)
    if model_name == "sbm":
        return _fit_sbm(train_graphs, "sbm")
    if model_name == "degree_corrected_sbm":
        return _fit_sbm(train_graphs, "degree_corrected_sbm")
    raise ValueError(f"Unknown classical baseline model: {model_name}")


def _scaled_block_sizes(template: SBMTemplate, total_nodes: int) -> list[int]:
    if total_nodes <= 0:
        raise ValueError("Generated graph size must be positive")
    if total_nodes < len(template.block_sizes):
        return [1] * total_nodes
    proportions = template.block_sizes / template.block_sizes.sum()
    raw = proportions * total_nodes
    sizes = np.floor(raw).astype(int)
    sizes[sizes == 0] = 1
    while int(sizes.sum()) > total_nodes:
        index = int(np.argmax(sizes))
        sizes[index] -= 1
    if int(sizes.sum()) < total_nodes:
        remainders = raw - np.floor(raw)
        order = list(np.argsort(-remainders))
        cursor = 0
        while int(sizes.sum()) < total_nodes:
            sizes[int(order[cursor % len(order)])] += 1
            cursor += 1
    return [int(size) for size in sizes if int(size) > 0]


def _sample_degree_sequence(model: FittedClassicalModel, n: int, rng: np.random.Generator) -> list[int]:
    degrees = np.asarray(model.degree_pool or (0,), dtype=np.int64)
    sequence = rng.choice(degrees, size=n, replace=True).astype(np.int64)
    sequence = np.clip(sequence, 0, max(0, n - 1))
    if int(sequence.sum()) % 2 == 1:
        candidates = np.flatnonzero(sequence < max(0, n - 1))
        if candidates.size:
            sequence[int(rng.choice(candidates))] += 1
        else:
            sequence[0] = max(0, sequence[0] - 1)
    return [int(value) for value in sequence]


def _generate_erdos_renyi(model: FittedClassicalModel, n: int, seed: int) -> nx.Graph:
    return nx.gnp_random_graph(n, model.edge_probability, seed=seed, directed=False)


def _generate_configuration_model(model: FittedClassicalModel, n: int, rng: np.random.Generator, seed: int) -> nx.Graph:
    sequence = _sample_degree_sequence(model, n, rng)
    graph = nx.configuration_model(sequence, seed=seed)
    return _sanitize_graph(graph)


def _generate_sbm(model: FittedClassicalModel, n: int, rng: np.random.Generator, seed: int) -> nx.Graph:
    template = model.sbm_templates[int(rng.integers(0, len(model.sbm_templates)))]
    sizes = _scaled_block_sizes(template, n)
    probs = template.block_probs[: len(sizes), : len(sizes)]
    return nx.stochastic_block_model(sizes, probs.tolist(), seed=seed, directed=False, selfloops=False)


def _sample_block_weights(template: SBMTemplate, sizes: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    sampled = []
    for block_index, size in enumerate(sizes):
        source = template.block_weights[min(block_index, len(template.block_weights) - 1)]
        sampled.append(rng.choice(source, size=size, replace=True).astype(np.float64))
    return sampled


def _generate_degree_corrected_sbm(
    model: FittedClassicalModel,
    n: int,
    rng: np.random.Generator,
) -> nx.Graph:
    template = model.sbm_templates[int(rng.integers(0, len(model.sbm_templates)))]
    sizes = _scaled_block_sizes(template, n)
    probs = template.block_probs[: len(sizes), : len(sizes)]
    block_weights = _sample_block_weights(template, sizes, rng)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    block_offsets = np.cumsum([0, *sizes])
    block_nodes = [
        list(range(int(block_offsets[index]), int(block_offsets[index + 1])))
        for index in range(len(sizes))
    ]
    for block_i, _ in enumerate(sizes):
        nodes_i = block_nodes[block_i]
        for block_j in range(block_i, len(sizes)):
            nodes_j = block_nodes[block_j]
            base_prob = float(probs[block_i, block_j])
            if base_prob <= 0.0:
                continue
            for local_i, u in enumerate(nodes_i):
                start_j = local_i + 1 if block_i == block_j else 0
                for local_j, v in enumerate(nodes_j[start_j:]):
                    probability = base_prob * block_weights[block_i][local_i] * block_weights[block_j][start_j + local_j]
                    if rng.random() < min(1.0, probability):
                        graph.add_edge(u, v)
    return graph


def generate_from_fit(
    model: FittedClassicalModel,
    test_graphs: list[nx.Graph],
    *,
    num_graphs: int,
    seed: int,
) -> list[nx.Graph]:
    rng = np.random.default_rng(seed)
    sizes = _sample_sizes(test_graphs, num_graphs, rng)
    generated = []
    for index, n in enumerate(sizes):
        nx_seed = int(np.random.SeedSequence([seed, index, n]).generate_state(1, dtype=np.uint32)[0])
        if model.name == "erdos_renyi":
            graph = _generate_erdos_renyi(model, n, nx_seed)
        elif model.name == "configuration_model":
            graph = _generate_configuration_model(model, n, rng, nx_seed)
        elif model.name == "sbm":
            graph = _generate_sbm(model, n, rng, nx_seed)
        elif model.name == "degree_corrected_sbm":
            graph = _generate_degree_corrected_sbm(model, n, rng)
        else:
            raise ValueError(f"Unknown fitted model: {model.name}")
        generated.append(_sanitize_graph(graph))
    return generated


def generate_baseline_graphs(
    *,
    model_name: str,
    train_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    num_graphs: int,
    seed: int,
) -> list[nx.Graph]:
    model = fit_model(model_name, train_graphs)
    return generate_from_fit(model, test_graphs, num_graphs=num_graphs, seed=seed)


def generate_many_baselines(
    *,
    train_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    model_names: Iterable[str],
    num_graphs: int,
    seed: int,
) -> dict[str, list[nx.Graph]]:
    return {
        model_name: generate_baseline_graphs(
            model_name=model_name,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            num_graphs=num_graphs,
            seed=seed,
        )
        for model_name in model_names
    }
