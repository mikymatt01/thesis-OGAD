from __future__ import annotations

import numpy as np


def resolve_split_counts(
    *,
    instances: int | None,
    train_ratio: float | None,
    train_instances: int | None,
    test_instances: int | None,
) -> tuple[int, int, int, str]:
    explicit = train_instances is not None or test_instances is not None
    ratio = train_ratio is not None
    if explicit and ratio:
        raise ValueError("--train-ratio cannot be combined with --train-instances/--test-instances")
    if explicit:
        if train_instances is None or test_instances is None:
            raise ValueError("--train-instances and --test-instances must be provided together")
        total = train_instances + test_instances
        if instances is not None and instances != total:
            raise ValueError("--instances must equal train_instances + test_instances in explicit split mode")
        if train_instances <= 0 or test_instances <= 0:
            raise ValueError("Explicit train/test counts must both be positive")
        return total, train_instances, test_instances, "explicit"
    if instances is None:
        raise ValueError("--instances is required when using --train-ratio")
    if train_ratio is None:
        raise ValueError("--train-ratio is required unless explicit split counts are provided")
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must satisfy 0 < train_ratio < 1")
    train_count = int(round(instances * train_ratio))
    test_count = instances - train_count
    if train_count <= 0 or test_count <= 0:
        raise ValueError("Split produced an empty train or test set")
    return instances, train_count, test_count, "ratio"


def assign_splits(
    instances: int,
    train_count: int,
    *,
    shuffle_before_split: bool,
    split_seed: int,
) -> dict[int, tuple[str, int]]:
    indices = list(range(instances))
    if shuffle_before_split:
        rng = np.random.default_rng(split_seed)
        indices = list(rng.permutation(indices))
    train = indices[:train_count]
    test = indices[train_count:]
    mapping: dict[int, tuple[str, int]] = {}
    for split_index, global_index in enumerate(train):
        mapping[int(global_index)] = ("train", split_index)
    for split_index, global_index in enumerate(test):
        mapping[int(global_index)] = ("test", split_index)
    return mapping


def resolve_sizes(
    *,
    size_mode: str,
    instances: int,
    nodes: int | None,
    nodes_min: int | None,
    nodes_max: int | None,
    size_seed: int,
) -> list[int]:
    if size_mode == "fixed":
        if nodes is None or nodes <= 0:
            raise ValueError("--nodes must be a positive integer in fixed size mode")
        return [int(nodes)] * instances
    if nodes_min is None or nodes_max is None:
        raise ValueError("--nodes-min and --nodes-max are required for variable size modes")
    if nodes_min <= 0 or nodes_max < nodes_min:
        raise ValueError("Invalid variable size bounds")
    if size_mode == "uniform":
        rng = np.random.default_rng(size_seed)
        return [int(v) for v in rng.integers(nodes_min, nodes_max + 1, size=instances)]
    if size_mode == "linspace":
        return [int(round(v)) for v in np.linspace(nodes_min, nodes_max, instances)]
    raise ValueError(f"Unsupported size mode: {size_mode}")
