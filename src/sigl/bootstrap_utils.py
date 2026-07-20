"""Pure-Python helpers for SIGL cluster bootstrap planning."""

import random
from typing import Dict, Iterable, List, Optional, Sequence, Set


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
    min_bootstrap_cluster_size: int = 1,
) -> List[Dict[str, object]]:
    """Return deterministic SIGL graphon jobs for clustered training graphs.

    For ``bootstrap_replicates <= 1`` and ``include_original_clusters=False``,
    the plan returns one original job per cluster. This preserves the previous
    one-graphon-per-cluster behavior unless bootstrap is explicitly increased.
    """
    clusters = _validate_cluster_indices(cluster_indices)
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be at least 1")
    if min_bootstrap_cluster_size < 1:
        raise ValueError("min_bootstrap_cluster_size must be at least 1")

    rng = random.Random(seed)
    jobs = []
    sample_id = 0

    for cluster_id, indices in enumerate(clusters):
        use_bootstrap = (
            bootstrap_replicates > 1
            and len(indices) >= min_bootstrap_cluster_size
        )
        if include_original_clusters or not use_bootstrap:
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

        if not use_bootstrap:
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


def parse_job_ids(value: Optional[str]) -> Optional[Set[int]]:
    """Parse a comma-separated job-id list."""
    if value is None or str(value).strip() == "":
        return None
    job_ids = set()
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        job_id = int(item)
        if job_id < 0:
            raise ValueError("job ids must be non-negative")
        job_ids.add(job_id)
    return job_ids


def select_graphon_jobs(
    jobs: Sequence[Dict[str, object]],
    job_ids: Optional[Set[int]] = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> List[Dict[str, object]]:
    """Return the subset assigned to one worker.

    Sharding is based on sample_id modulo shard_count so each worker writes a
    disjoint deterministic subset.
    """
    if shard_count < 1:
        raise ValueError("job_shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("job_shard_index must be in [0, job_shard_count)")

    selected = []
    for job in jobs:
        sample_id = int(job["sample_id"])
        if job_ids is not None and sample_id not in job_ids:
            continue
        if sample_id % shard_count != shard_index:
            continue
        selected.append(dict(job))
    return selected
