"""Pure-Python graph split helpers for SIGL/GGME validation."""

import random
from typing import Iterable, List, Tuple


def deterministic_graph_split(
    dataset_size: int, validation_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Return disjoint model-train and GGME-validation graph indices."""
    if dataset_size < 2:
        raise ValueError("Graph selection split requires at least two graphs")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, min(dataset_size - 1, round(dataset_size * validation_fraction)))
    validation_indices = sorted(indices[:validation_size])
    model_train_indices = sorted(indices[validation_size:])
    return model_train_indices, validation_indices


def select_by_indices(items: Iterable[object], indices: Iterable[int]) -> List[object]:
    """Select items by integer indices while preserving the index order."""
    materialized = list(items)
    return [materialized[index] for index in indices]
