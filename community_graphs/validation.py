from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .csr import has_self_loops, is_binary, sparse_is_symmetric, validate_csr_matrix
from .hashing import csr_structure_sha256, file_sha256
from .sigl_loader import densify_graph, load_community_graphs_sparse


class ValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _shape_summary(graphs: list[sparse.csr_matrix]) -> str:
    shapes = sorted({graph.shape for graph in graphs})
    if len(shapes) == 1:
        return f"{shapes[0][0]}x{shapes[0][1]}"
    return ", ".join(f"{r}x{c}" for r, c in shapes[:8]) + ("..." if len(shapes) > 8 else "")


def _dtype_summary(graphs: list[sparse.csr_matrix]) -> str:
    return ", ".join(sorted({str(graph.dtype) for graph in graphs}))


def _validate_graphs(
    graphs: list[sparse.csr_matrix],
    *,
    split: str,
    metadata: dict[str, Any],
) -> list[str]:
    adjacency_meta = metadata.get("adjacency", {})
    expected_binary = adjacency_meta.get("binary")
    expected_symmetric = adjacency_meta.get("symmetric")
    expected_self_loops = adjacency_meta.get("self_loops")
    hashes = []
    for index, graph in enumerate(graphs):
        validate_csr_matrix(
            graph,
            expected_binary=expected_binary,
            expected_symmetric=expected_symmetric,
            expected_self_loops=expected_self_loops,
            index=index,
        )
        if expected_binary and graph.nnz and set(graph.data.tolist()) - {1}:
            raise ValidationError(f"{split} graph {index} has non-one stored data")
        if expected_symmetric and not sparse_is_symmetric(graph):
            raise ValidationError(f"{split} graph {index} is not symmetric")
        if expected_self_loops is False and has_self_loops(graph):
            raise ValidationError(f"{split} graph {index} contains self loops")
        hashes.append(csr_structure_sha256(graph))
    return hashes


def validate_dataset_dir(dataset_dir: str | Path, *, quiet: bool = False) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    manifest_path = dataset_dir / "manifest.csv"
    train_path = dataset_dir / "CommunityGraphs_train.pkl"
    test_path = dataset_dir / "CommunityGraphs_test.pkl"
    for path in (metadata_path, manifest_path, train_path, test_path):
        if not path.exists():
            raise ValidationError(f"Missing required file: {path}")
    metadata = _load_json(metadata_path)
    manifest = _load_manifest(manifest_path)
    if metadata.get("schema_name") != "CommunityGraphs":
        raise ValidationError("metadata.json schema_name is not CommunityGraphs")
    if metadata.get("matrix_type") != "scipy.sparse.csr_matrix":
        raise ValidationError("metadata.json matrix_type is not scipy.sparse.csr_matrix")
    expected_train = int(metadata["splits"]["train_instances"])
    expected_test = int(metadata["splits"]["test_instances"])
    train_graphs = load_community_graphs_sparse(train_path)
    test_graphs = load_community_graphs_sparse(test_path)
    if len(train_graphs) != expected_train:
        raise ValidationError(f"Train count mismatch: metadata={expected_train}, pickle={len(train_graphs)}")
    if len(test_graphs) != expected_test:
        raise ValidationError(f"Test count mismatch: metadata={expected_test}, pickle={len(test_graphs)}")
    for file_name, path in [
        ("CommunityGraphs_train.pkl", train_path),
        ("CommunityGraphs_test.pkl", test_path),
    ]:
        expected_hash = metadata.get("hashes", {}).get(file_name)
        actual_hash = file_sha256(path)
        if expected_hash != actual_hash:
            raise ValidationError(f"File hash mismatch for {file_name}: metadata={expected_hash}, actual={actual_hash}")
    train_hashes = _validate_graphs(train_graphs, split="train", metadata=metadata)
    test_hashes = _validate_graphs(test_graphs, split="test", metadata=metadata)
    duplicates = sorted(set(train_hashes) & set(test_hashes))
    if duplicates:
        raise ValidationError(f"Cross-split duplicates detected: {len(duplicates)}")
    if len(manifest) != expected_train + expected_test:
        raise ValidationError("manifest.csv row count does not match split counts")
    by_key = {(row["split"], int(row["split_index"])): row for row in manifest}
    for split, graphs, hashes in [("train", train_graphs, train_hashes), ("test", test_graphs, test_hashes)]:
        for index, graph in enumerate(graphs):
            row = by_key.get((split, index))
            if row is None:
                raise ValidationError(f"manifest.csv missing row for {split}[{index}]")
            if row["structure_sha256"] != hashes[index]:
                raise ValidationError(f"manifest hash mismatch for {split}[{index}]")
            if int(row["num_nodes"]) != graph.shape[0]:
                raise ValidationError(f"manifest num_nodes mismatch for {split}[{index}]")
            if int(row["nnz"]) != graph.nnz:
                raise ValidationError(f"manifest nnz mismatch for {split}[{index}]")
            if metadata["adjacency"].get("symmetric") and not has_self_loops(graph):
                num_edges = int(row["num_edges"])
                if graph.nnz != 2 * num_edges:
                    raise ValidationError(f"nnz invariant failed for {split}[{index}]")
    if train_graphs:
        dense = densify_graph(train_graphs[0], dtype=np.float32, max_dense_bytes=512 * 1024 * 1024)
        if not isinstance(dense, np.ndarray) or dense.__class__.__name__ == "matrix":
            raise ValidationError("SIGL dense conversion did not return ndarray")
    if test_graphs:
        dense = densify_graph(test_graphs[0], dtype=np.float32, max_dense_bytes=512 * 1024 * 1024)
        if not dense.flags["C_CONTIGUOUS"]:
            raise ValidationError("SIGL dense conversion is not C-contiguous")
    result = {
        "train_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "matrix_type": "scipy.sparse.csr_matrix",
        "shapes": _shape_summary(train_graphs + test_graphs),
        "dtype": _dtype_summary(train_graphs + test_graphs),
        "cross_split_duplicates": len(duplicates),
    }
    if not quiet:
        print("CommunityGraphs validation passed")
        print(f"Train graphs: {result['train_graphs']}")
        print(f"Test graphs: {result['test_graphs']}")
        print(f"Matrix type: {result['matrix_type']}")
        print(f"Shapes: {result['shapes']}")
        print(f"Dtype: {result['dtype']}")
        print(f"Cross-split duplicates: {result['cross_split_duplicates']}")
        print("SIGL dense conversion: passed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a CommunityGraphs dataset directory.")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_dataset_dir(args.dataset_dir)
    except Exception as exc:
        print(f"CommunityGraphs validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
