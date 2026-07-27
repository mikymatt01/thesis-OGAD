from __future__ import annotations

import numpy as np


def derive_seed_plan(master_seed: int, instances: int) -> tuple[int, int, list[int]]:
    """Return split seed, size seed, and one deterministic seed per global index."""

    root = np.random.SeedSequence(int(master_seed))
    split_ss, size_ss, instance_parent = root.spawn(3)
    split_seed = int(split_ss.generate_state(1, dtype=np.uint32)[0])
    size_seed = int(size_ss.generate_state(1, dtype=np.uint32)[0])
    children = instance_parent.spawn(instances)
    instance_seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return split_seed, size_seed, instance_seeds
