from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import scipy
from scipy import sparse

from .config import DatasetConfig, validate_protocol
from .csr import graph_to_csr, has_self_loops, is_binary, sparse_is_symmetric, validate_csr_matrix
from .hashing import csr_structure_sha256
from .metrics import basic_graph_stats, expensive_graph_stats
from .registry import FamilySpec, get_family
from .seeds import derive_seed_plan
from .serialization import atomic_json_dump, atomic_pickle_dump, sha256_for_written_pickles, write_text_atomic, write_validator_script
from .splitting import assign_splits, resolve_sizes, resolve_split_counts


MANIFEST_FIELDS = [
    "graph_id",
    "global_index",
    "split",
    "split_index",
    "family",
    "instance_seed",
    "num_nodes",
    "original_num_nodes",
    "num_edges",
    "nnz",
    "density",
    "dtype",
    "directed",
    "symmetric",
    "binary",
    "has_self_loops",
    "connected",
    "num_components",
    "generation_attempts",
    "structure_sha256",
    "cross_split_duplicate",
    "parameters_json",
    "status",
    "error",
]


@dataclass
class GeneratedGraph:
    global_index: int
    split: str
    split_index: int
    graph: nx.Graph
    adjacency: sparse.csr_matrix
    row: dict[str, Any]


def _postprocess_graph(graph: nx.Graph, config: DatasetConfig, spec: FamilySpec) -> nx.Graph:
    if config.symmetrize and graph.is_directed():
        graph = graph.to_undirected()
    if not graph.is_directed() and isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        graph = nx.Graph(graph)
    elif graph.is_directed() and isinstance(graph, nx.MultiDiGraph):
        graph = nx.DiGraph(graph)
    if config.remove_self_loops and not config.allow_self_loops:
        graph = graph.copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))
    if config.largest_component:
        if graph.is_directed():
            components = nx.weakly_connected_components(graph)
        else:
            components = nx.connected_components(graph)
        largest = max(components, key=len)
        graph = graph.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(graph, first_label=0, ordering="default")


def _is_connected(graph: nx.Graph) -> bool:
    if graph.number_of_nodes() == 0:
        return False
    if graph.is_directed():
        return nx.is_weakly_connected(graph)
    return nx.is_connected(graph)


def generate_one(
    *,
    global_index: int,
    split: str,
    split_index: int,
    nodes: int | None,
    instance_seed: int,
    config: DatasetConfig,
    spec: FamilySpec,
) -> GeneratedGraph:
    attempts = 0
    last_error = ""
    while attempts < config.max_attempts:
        attempts += 1
        seed = int(np.random.SeedSequence([instance_seed, attempts]).generate_state(1, dtype=np.uint32)[0])
        try:
            original = spec.builder(nodes, dict(config.params), seed)
            original_nodes = original.number_of_nodes()
            graph = _postprocess_graph(original, config, spec)
            if config.require_connected and not _is_connected(graph):
                last_error = "graph is not connected"
                continue
            adjacency = graph_to_csr(graph, dtype=config.np_dtype, binary=config.binary_adjacency)
            expected_symmetric = not graph.is_directed()
            expected_loops = True if config.allow_self_loops else False
            adjacency = validate_csr_matrix(
                adjacency,
                expected_binary=config.binary_adjacency,
                expected_symmetric=expected_symmetric,
                expected_self_loops=expected_loops,
                index=global_index,
            )
            stats = basic_graph_stats(graph, adjacency)
            if config.compute_metrics:
                stats.update(expensive_graph_stats(graph))
            structure_hash = csr_structure_sha256(adjacency)
            row = {
                "graph_id": f"{split}_{split_index:06d}",
                "global_index": global_index,
                "split": split,
                "split_index": split_index,
                "family": spec.slug,
                "instance_seed": instance_seed,
                "num_nodes": stats["num_nodes"],
                "original_num_nodes": original_nodes,
                "num_edges": stats["num_edges"],
                "nnz": stats["nnz"],
                "density": f"{stats['density']:.12g}",
                "dtype": str(adjacency.dtype),
                "directed": graph.is_directed(),
                "symmetric": sparse_is_symmetric(adjacency),
                "binary": is_binary(adjacency),
                "has_self_loops": has_self_loops(adjacency),
                "connected": stats["connected"],
                "num_components": stats["num_components"],
                "generation_attempts": attempts,
                "structure_sha256": structure_hash,
                "parameters_json": json.dumps(config.params, sort_keys=True, separators=(",", ":")),
                "status": "ok",
                "error": "",
            }
            if not graph.is_directed() and not row["has_self_loops"] and row["nnz"] != 2 * row["num_edges"]:
                raise ValueError("Invariant failed: nnz != 2 * num_edges")
            return GeneratedGraph(global_index, split, split_index, graph, adjacency, row)
        except Exception as exc:
            last_error = str(exc)
            if not config.require_connected:
                break
    raise RuntimeError(f"Failed to generate graph {global_index} after {attempts} attempts: {last_error}")


def _write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["split"], int(r["split_index"]))):
        writer.writerow(row)
    write_text_atomic(buffer.getvalue(), path)


def _check_cross_split_duplicates(rows: list[dict[str, Any]]) -> list[str]:
    train_hashes = {row["structure_sha256"] for row in rows if row["split"] == "train"}
    test_hashes = {row["structure_sha256"] for row in rows if row["split"] == "test"}
    return sorted(train_hashes & test_hashes)


def build_metadata(
    *,
    config: DatasetConfig,
    spec: FamilySpec,
    split_mode: str,
    train_count: int,
    test_count: int,
    split_seed: int,
    size_seed: int,
    instance_seeds: list[int],
    rows: list[dict[str, Any]],
    pickle_hashes: dict[str, str],
    command: list[str],
    failures: int,
) -> dict[str, Any]:
    directed = bool(spec.directed and not config.symmetrize)
    symmetric = not directed
    return {
        "schema_name": "CommunityGraphs",
        "schema_version": "1.0",
        "serialization": "pickle",
        "container_type": "list",
        "matrix_type": "scipy.sparse.csr_matrix",
        "security": {
            "pickle_warning": "Pickle can execute code during loading; only load trusted datasets.",
            "dense_memory_warning": "SIGL densifies each n x n CSR; float32 requires approximately 4*n^2 bytes per graph.",
        },
        "adjacency": {
            "binary": bool(config.binary_adjacency),
            "weighted": not bool(config.binary_adjacency),
            "directed": directed,
            "symmetric": symmetric,
            "self_loops": bool(config.allow_self_loops),
            "default_dtype": config.dtype,
        },
        "splits": {
            "train_file": "CommunityGraphs_train.pkl",
            "test_file": "CommunityGraphs_test.pkl",
            "train_instances": train_count,
            "test_instances": test_count,
            "strategy": split_mode,
            "shuffle_before_split": config.shuffle_before_split,
        },
        "family": {
            "slug": spec.slug,
            "name": spec.name,
            "networkx_function": spec.function_name,
            "directed": spec.directed,
            "normally_simple": spec.normally_simple,
            "can_self_loop": spec.can_self_loop,
            "can_parallel_edges": spec.can_parallel_edges,
        },
        "parameters": config.params,
        "seed": {
            "master_seed": config.seed,
            "split_seed": split_seed,
            "size_seed": size_seed,
            "instance_seeds": instance_seeds,
        },
        "size": {
            "mode": config.size_mode,
            "nodes": config.nodes,
            "nodes_min": config.nodes_min,
            "nodes_max": config.nodes_max,
        },
        "versions": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "hashes": pickle_hashes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": command,
        "config": config.to_json_dict(),
        "counts": {"successes": len(rows), "failures": failures},
    }


def generate_dataset(config: DatasetConfig, *, command: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.DEBUG if config.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    validate_protocol(config.protocol)
    if config.require_connected and config.largest_component:
        raise ValueError("--require-connected and --largest-component are mutually exclusive")
    if config.allow_self_loops:
        config.remove_self_loops = False
    if config.family is None:
        raise ValueError("--family is required")
    spec = get_family(config.family)
    total, train_count, test_count, split_mode = resolve_split_counts(
        instances=config.instances,
        train_ratio=config.train_ratio,
        train_instances=config.train_instances,
        test_instances=config.test_instances,
    )
    split_seed, size_seed, instance_seeds = derive_seed_plan(config.seed, total)
    sizes = resolve_sizes(
        size_mode=config.size_mode,
        instances=total,
        nodes=config.nodes,
        nodes_min=config.nodes_min,
        nodes_max=config.nodes_max,
        size_seed=size_seed,
    )
    split_map = assign_splits(
        total,
        train_count,
        shuffle_before_split=config.shuffle_before_split,
        split_seed=split_seed,
    )
    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite and not config.dry_run:
        raise FileExistsError(f"{output_dir} already exists and is not empty; pass --overwrite to replace files")
    if config.dry_run:
        return {
            "dry_run": True,
            "family": spec.slug,
            "instances": total,
            "train_instances": train_count,
            "test_instances": test_count,
            "sizes": sizes,
            "config": config.to_json_dict(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedGraph] = []
    for global_index in range(total):
        split, split_index = split_map[global_index]
        generated.append(
            generate_one(
                global_index=global_index,
                split=split,
                split_index=split_index,
                nodes=sizes[global_index],
                instance_seed=instance_seeds[global_index],
                config=config,
                spec=spec,
            )
        )
    rows = [item.row for item in generated]
    duplicates = _check_cross_split_duplicates(rows)
    for row in rows:
        row["cross_split_duplicate"] = row["structure_sha256"] in duplicates
    if duplicates:
        _write_manifest(rows, output_dir / "manifest.csv")
        raise ValueError(f"Detected {len(duplicates)} duplicate graph structures across train/test splits")

    train_graphs = [g.adjacency for g in sorted(generated, key=lambda item: item.split_index) if g.split == "train"]
    test_graphs = [g.adjacency for g in sorted(generated, key=lambda item: item.split_index) if g.split == "test"]
    atomic_pickle_dump(train_graphs, output_dir / "CommunityGraphs_train.pkl", config.protocol)
    atomic_pickle_dump(test_graphs, output_dir / "CommunityGraphs_test.pkl", config.protocol)
    pickle_hashes = sha256_for_written_pickles(output_dir)
    metadata = build_metadata(
        config=config,
        spec=spec,
        split_mode=split_mode,
        train_count=train_count,
        test_count=test_count,
        split_seed=split_seed,
        size_seed=size_seed,
        instance_seeds=instance_seeds,
        rows=rows,
        pickle_hashes=pickle_hashes,
        command=command or sys.argv,
        failures=0,
    )
    _write_manifest(rows, output_dir / "manifest.csv")
    atomic_json_dump(metadata, output_dir / "metadata.json")
    atomic_json_dump(config.to_json_dict(), output_dir / "config.resolved.json")
    write_validator_script(output_dir / "validate_dataset.py")
    return metadata
