"""Pure-Python helpers for SIGL cluster bootstrap planning."""

import random
from typing import Dict, Iterable, List


def _validate_cluster_indices(cluster_indices: Iterable[Iterable[int]]) -> List[List[int]]:
    clusters = [list(indices) for indices in cluster_indices]
    if not clusters:
        raise ValueError("At least one cluster is required for bootstrap planning")
    for cluster_id, indices in enumerate(clusters):
        if not indices:
            raise ValueError("Cluster {} is empty".format(cluster_id))
    return clusters


def build_cluster_bootstrap_plan(
    cluster_indices: Iterable[Iterable[int]],
    bootstrap_replicates: int = 1,
    seed: int = 0,
    include_original_clusters: bool = False,
) -> List[Dict[str, object]]:
    """Return deterministic SIGL graphon jobs for clustered training graphs.

    For ``bootstrap_replicates <= 1`` and ``include_original_clusters=False``,
    the plan returns one original job per cluster. This preserves the previous
    one-graphon-per-cluster behavior unless bootstrap is explicitly increased.
    """
    clusters = _validate_cluster_indices(cluster_indices)
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be at least 1")

    rng = random.Random(seed)
    jobs = []
    sample_id = 0

    for cluster_id, indices in enumerate(clusters):
        if include_original_clusters or bootstrap_replicates == 1:
            jobs.append(
                {
                    "sample_id": sample_id,
                    "cluster_id": cluster_id,
                    "bootstrap_id": None,
                    "is_original": True,
                    "original_graph_indices": list(indices),
                }
            )
            sample_id += 1

        if bootstrap_replicates == 1:
            continue

        for bootstrap_id in range(bootstrap_replicates):
            sampled_indices = [rng.choice(indices) for _ in indices]
            jobs.append(
                {
                    "sample_id": sample_id,
                    "cluster_id": cluster_id,
                    "bootstrap_id": bootstrap_id,
                    "is_original": False,
                    "original_graph_indices": sampled_indices,
                }
            )
            sample_id += 1

    return jobs


def expected_graphon_count(
    n_clusters: int,
    bootstrap_replicates: int = 1,
    include_original_clusters: bool = False,
) -> int:
    """Return the number of SIGL graphons implied by the bootstrap settings."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be at least 1")
    if bootstrap_replicates == 1:
        return n_clusters
    return n_clusters * bootstrap_replicates + (
        n_clusters if include_original_clusters else 0
    )
