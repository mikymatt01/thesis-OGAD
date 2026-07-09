"""Dependency-light helpers shared by graphon sampling and local tests."""

from __future__ import annotations

from typing import Any, Sequence


def probability_matrix(graphon: Any, node_indices: Any) -> Any:
    """Select W[eta_i, eta_j] using one latent index vector for both axes."""
    return graphon[node_indices][:, node_indices]


def sanitize_graphon_matrix(graphon: Any) -> Any:
    """Return a symmetric [0, 1] matrix with a zero diagonal.

    The implementation deliberately relies only on ndarray/Tensor-compatible
    operations so it can be tested locally without importing Torch.
    """
    sanitized = ((graphon + graphon.transpose(-1, -2)) / 2).clip(0, 1)
    sanitized = sanitized.clone() if hasattr(sanitized, "clone") else sanitized.copy()
    diagonal_indices = list(range(sanitized.shape[-1]))
    sanitized[diagonal_indices, diagonal_indices] = 0
    return sanitized


def cyclic_graphon_index(sample_index: int, graphon_count: int) -> int:
    """Map a generated-graph index to an available graphon cyclically."""
    if graphon_count < 1:
        raise ValueError("At least one graphon is required")
    return sample_index % graphon_count


def validate_generation_request(num_graphs: int, node_sizes: Sequence[int]) -> None:
    if num_graphs < 1:
        raise ValueError("num_generated_graphs must be at least 1")
    if not node_sizes:
        raise ValueError("test node-size distribution is empty")
