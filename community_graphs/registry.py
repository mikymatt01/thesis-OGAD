from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx
import numpy as np


GraphBuilder = Callable[[int | None, dict[str, Any], int], nx.Graph]


@dataclass(frozen=True)
class FamilySpec:
    slug: str
    name: str
    function_name: str
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    directed: bool = False
    normally_simple: bool = True
    can_self_loop: bool = False
    can_parallel_edges: bool = False
    uses_nodes: bool = True
    supports_require_connected: bool = True
    builder: GraphBuilder = field(compare=False, repr=False, default=lambda n, p, s: nx.Graph())

    def describe(self) -> str:
        return "\n".join(
            [
                f"{self.slug}: {self.name}",
                f"NetworkX: {self.function_name}",
                f"Required params: {', '.join(self.required_params) or '-'}",
                f"Optional params: {', '.join(self.optional_params) or '-'}",
                f"Aliases: {', '.join(self.aliases) or '-'}",
                f"Directed: {self.directed}",
                f"Normally simple: {self.normally_simple}",
                f"Can self-loop: {self.can_self_loop}",
                f"Can parallel edges: {self.can_parallel_edges}",
                f"Uses --nodes: {self.uses_nodes}",
                f"Supports --require-connected: {self.supports_require_connected}",
                f"Constraints: {'; '.join(self.constraints) or '-'}",
            ]
        )


def _require_n(n: int | None) -> int:
    if n is None:
        raise ValueError("This family requires a node count")
    return int(n)


def _as_int(params: dict[str, Any], key: str, default: int | None = None) -> int:
    if key not in params:
        if default is None:
            raise ValueError(f"Missing required parameter: {key}")
        return default
    return int(params[key])


def _as_float(params: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in params:
        if default is None:
            raise ValueError(f"Missing required parameter: {key}")
        return default
    return float(params[key])


def _er(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.gnp_random_graph(_require_n(n), _as_float(params, "p"), seed=seed, directed=False)


def _gnm(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.gnm_random_graph(_require_n(n), _as_int(params, "m"), seed=seed, directed=False)


def _ba(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(_require_n(n), _as_int(params, "m"), seed=seed)


def _dual_ba(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.dual_barabasi_albert_graph(
        _require_n(n),
        _as_int(params, "m1"),
        _as_int(params, "m2"),
        _as_float(params, "p"),
        seed=seed,
    )


def _extended_ba(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.extended_barabasi_albert_graph(
        _require_n(n),
        _as_int(params, "m"),
        _as_float(params, "p"),
        _as_float(params, "q"),
        seed=seed,
    )


def _powerlaw_cluster(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.powerlaw_cluster_graph(_require_n(n), _as_int(params, "m"), _as_float(params, "p"), seed=seed)


def _watts(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(_require_n(n), _as_int(params, "k"), _as_float(params, "p"), seed=seed)


def _newman_watts(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.newman_watts_strogatz_graph(_require_n(n), _as_int(params, "k"), _as_float(params, "p"), seed=seed)


def _connected_watts(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.connected_watts_strogatz_graph(
        _require_n(n),
        _as_int(params, "k"),
        _as_float(params, "p"),
        tries=_as_int(params, "tries", 100),
        seed=seed,
    )


def _random_regular(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.random_regular_graph(_as_int(params, "d"), _require_n(n), seed=seed)


def _configuration(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    sequence = params.get("degree_sequence")
    if sequence is None:
        degree = _as_int(params, "degree", 3)
        sequence = [degree] * _require_n(n)
    return nx.configuration_model([int(v) for v in sequence], seed=seed)


def _expected_degree(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    weights = params.get("weights")
    if weights is None:
        rng = np.random.default_rng(seed)
        weights = rng.uniform(_as_float(params, "min_weight", 1.0), _as_float(params, "max_weight", 4.0), _require_n(n))
    return nx.expected_degree_graph([float(v) for v in weights], selfloops=bool(params.get("selfloops", False)), seed=seed)


def _sbm(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    sizes = [int(v) for v in params["sizes"]]
    probabilities = params["probabilities"]
    if n is not None and sum(sizes) != int(n):
        raise ValueError("For stochastic-block-model, --nodes must equal sum(sizes)")
    return nx.stochastic_block_model(sizes, probabilities, seed=seed, directed=False, selfloops=False)


def _planted_partition(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.planted_partition_graph(
        _as_int(params, "l"),
        _as_int(params, "k"),
        _as_float(params, "p_in"),
        _as_float(params, "p_out"),
        seed=seed,
        directed=False,
    )


def _gaussian_random_partition(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.gaussian_random_partition_graph(
        _require_n(n),
        _as_int(params, "s"),
        _as_float(params, "v"),
        _as_float(params, "p_in"),
        _as_float(params, "p_out"),
        directed=False,
        seed=seed,
    )


def _lfr(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.LFR_benchmark_graph(
        _require_n(n),
        tau1=_as_float(params, "tau1"),
        tau2=_as_float(params, "tau2"),
        mu=_as_float(params, "mu"),
        average_degree=params.get("average_degree"),
        min_degree=params.get("min_degree"),
        max_degree=params.get("max_degree"),
        min_community=params.get("min_community"),
        max_community=params.get("max_community"),
        seed=seed,
    )


def _random_geometric(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.random_geometric_graph(_require_n(n), _as_float(params, "radius"), seed=seed)


def _waxman(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.waxman_graph(
        _require_n(n),
        beta=_as_float(params, "beta", 0.4),
        alpha=_as_float(params, "alpha", 0.1),
        seed=seed,
    )


def _threshold_geometric(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.thresholded_random_geometric_graph(
        _require_n(n),
        _as_float(params, "radius"),
        _as_float(params, "theta"),
        seed=seed,
    )


def _random_labelled_tree(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.random_labeled_tree(_require_n(n), seed=seed)


def _random_lobster(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.random_lobster(_require_n(n), _as_float(params, "p1"), _as_float(params, "p2"), seed=seed)


def _balanced_tree(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    return nx.balanced_tree(_as_int(params, "r"), _as_int(params, "h"))


def _grid_2d(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    rows = params.get("rows")
    cols = params.get("cols")
    if rows is None or cols is None:
        nn = _require_n(n)
        rows = int(np.floor(np.sqrt(nn)))
        cols = int(np.ceil(nn / rows))
    graph = nx.grid_2d_graph(int(rows), int(cols))
    if n is not None and graph.number_of_nodes() > int(n):
        keep = list(graph.nodes())[: int(n)]
        graph = graph.subgraph(keep).copy()
    return graph


def _bipartite_er(n: int | None, params: dict[str, Any], seed: int) -> nx.Graph:
    n1 = _as_int(params, "n1")
    n2 = _as_int(params, "n2")
    if n is not None and n1 + n2 != int(n):
        raise ValueError("For bipartite-erdos-renyi, --nodes must equal n1+n2")
    return nx.bipartite.random_graph(n1, n2, _as_float(params, "p"), seed=seed, directed=False)


def _directed_er(n: int | None, params: dict[str, Any], seed: int) -> nx.DiGraph:
    return nx.gnp_random_graph(_require_n(n), _as_float(params, "p"), seed=seed, directed=True)


def _scale_free_directed(n: int | None, params: dict[str, Any], seed: int) -> nx.MultiDiGraph:
    return nx.scale_free_graph(
        _require_n(n),
        alpha=_as_float(params, "alpha", 0.41),
        beta=_as_float(params, "beta", 0.54),
        gamma=_as_float(params, "gamma", 0.05),
        delta_in=_as_float(params, "delta_in", 0.2),
        delta_out=_as_float(params, "delta_out", 0.0),
        seed=seed,
    )


FAMILIES: dict[str, FamilySpec] = {}


def _register(spec: FamilySpec) -> None:
    FAMILIES[spec.slug] = spec
    for alias in spec.aliases:
        FAMILIES[alias] = spec


for _spec in [
    FamilySpec("erdos-renyi", "Erdos-Renyi G(n,p)", "nx.gnp_random_graph", ("p",), aliases=("er", "gnp"), builder=_er),
    FamilySpec("gnm-random", "Uniform G(n,m)", "nx.gnm_random_graph", ("m",), aliases=("gnm",), builder=_gnm),
    FamilySpec("barabasi-albert", "Barabasi-Albert preferential attachment", "nx.barabasi_albert_graph", ("m",), aliases=("ba",), builder=_ba),
    FamilySpec("dual-barabasi-albert", "Dual Barabasi-Albert", "nx.dual_barabasi_albert_graph", ("m1", "m2", "p"), builder=_dual_ba),
    FamilySpec("extended-barabasi-albert", "Extended Barabasi-Albert", "nx.extended_barabasi_albert_graph", ("m", "p", "q"), builder=_extended_ba),
    FamilySpec("powerlaw-cluster", "Holme-Kim powerlaw cluster", "nx.powerlaw_cluster_graph", ("m", "p"), builder=_powerlaw_cluster),
    FamilySpec("watts-strogatz", "Watts-Strogatz small-world", "nx.watts_strogatz_graph", ("k", "p"), aliases=("ws",), builder=_watts),
    FamilySpec("newman-watts-strogatz", "Newman-Watts-Strogatz", "nx.newman_watts_strogatz_graph", ("k", "p"), builder=_newman_watts),
    FamilySpec("connected-watts-strogatz", "Connected Watts-Strogatz", "nx.connected_watts_strogatz_graph", ("k", "p"), optional_params=("tries",), builder=_connected_watts),
    FamilySpec("random-regular", "Random d-regular graph", "nx.random_regular_graph", ("d",), builder=_random_regular),
    FamilySpec("configuration-model", "Configuration model", "nx.configuration_model", (), optional_params=("degree_sequence", "degree"), normally_simple=False, can_self_loop=True, can_parallel_edges=True, builder=_configuration),
    FamilySpec("expected-degree", "Expected-degree graph", "nx.expected_degree_graph", (), optional_params=("weights", "min_weight", "max_weight", "selfloops"), can_self_loop=True, builder=_expected_degree),
    FamilySpec("stochastic-block-model", "Stochastic block model", "nx.stochastic_block_model", ("sizes", "probabilities"), uses_nodes=False, builder=_sbm),
    FamilySpec("planted-partition", "Planted partition graph", "nx.planted_partition_graph", ("l", "k", "p_in", "p_out"), uses_nodes=False, builder=_planted_partition),
    FamilySpec("gaussian-random-partition", "Gaussian random partition graph", "nx.gaussian_random_partition_graph", ("s", "v", "p_in", "p_out"), builder=_gaussian_random_partition),
    FamilySpec("lfr", "LFR benchmark graph", "nx.LFR_benchmark_graph", ("tau1", "tau2", "mu"), optional_params=("average_degree", "min_degree", "max_degree", "min_community", "max_community"), can_self_loop=True, builder=_lfr),
    FamilySpec("random-geometric", "Random geometric graph", "nx.random_geometric_graph", ("radius",), builder=_random_geometric),
    FamilySpec("waxman", "Waxman graph", "nx.waxman_graph", (), optional_params=("alpha", "beta"), builder=_waxman),
    FamilySpec("threshold-geometric", "Thresholded random geometric graph", "nx.thresholded_random_geometric_graph", ("radius", "theta"), builder=_threshold_geometric),
    FamilySpec("random-labelled-tree", "Random labelled tree", "nx.random_labeled_tree", (), aliases=("random-labeled-tree",), builder=_random_labelled_tree),
    FamilySpec("random-lobster", "Random lobster", "nx.random_lobster", ("p1", "p2"), builder=_random_lobster),
    FamilySpec("balanced-tree", "Balanced tree", "nx.balanced_tree", ("r", "h"), uses_nodes=False, builder=_balanced_tree),
    FamilySpec("grid-2d", "2D grid graph", "nx.grid_2d_graph", (), optional_params=("rows", "cols"), builder=_grid_2d),
    FamilySpec("bipartite-erdos-renyi", "Bipartite Erdos-Renyi graph", "nx.bipartite.random_graph", ("n1", "n2", "p"), uses_nodes=False, builder=_bipartite_er),
    FamilySpec("directed-erdos-renyi", "Directed Erdos-Renyi G(n,p)", "nx.gnp_random_graph", ("p",), directed=True, supports_require_connected=False, builder=_directed_er),
    FamilySpec("scale-free-directed", "Directed scale-free graph", "nx.scale_free_graph", (), optional_params=("alpha", "beta", "gamma", "delta_in", "delta_out"), directed=True, normally_simple=False, can_self_loop=True, can_parallel_edges=True, supports_require_connected=False, builder=_scale_free_directed),
]:
    _register(_spec)


def get_family(slug: str) -> FamilySpec:
    try:
        return FAMILIES[slug]
    except KeyError as exc:
        raise ValueError(f"Unknown family {slug!r}; use --list-families") from exc


def canonical_family_slugs() -> list[str]:
    return sorted({spec.slug for spec in FAMILIES.values()})
