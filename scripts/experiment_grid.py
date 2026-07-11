"""Experiment-grid helpers shared by serial and parallel launchers."""

from itertools import product
from typing import Any, Dict, List


def expand_block(block: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(block.keys())
    values_lists = [block[key] for key in keys]
    return [
        {key: value for key, value in zip(keys, values)}
        for values in product(*values_lists)
    ]


def expand_merged_experiment(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand only the SIGL block into scalar experiment configs.

    D3PM remains a grid inside each scalar SIGL run, preserving the current
    pipeline shape.
    """
    experiments = []
    for index, sigl_run in enumerate(expand_block(raw["sigl"])):
        experiment = dict(raw)
        experiment["shared"] = dict(raw.get("shared") or {})
        experiment["sigl"] = dict(sigl_run)
        experiment["d3pm"] = dict(raw["d3pm"])
        experiment["d3pm"]["image_size"] = [sigl_run["Res"]]
        experiment["name"] = "test_nc{}".format(index)
        experiments.append(experiment)
    return experiments


def assign_gpu(job_index: int, gpus: List[str]) -> str:
    if not gpus:
        raise ValueError("At least one GPU id is required")
    return gpus[job_index % len(gpus)]
