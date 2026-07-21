"""Experiment-grid helpers shared by serial and parallel launchers."""

import copy
from itertools import product
import re
from typing import Any, Dict, List


REQUIRED_DATASET_FIELDS = (
    "name",
    "train_graphs_path",
    "test_graphs_path",
    "paper_baseline_graphs_path",
    "sigl_graphons_dir",
    "ogad_generated_graphons_path",
    "digress_generated_samples_path",
    "generated_graph_count",
)


def expand_block(block: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(block.keys())
    values_lists = [block[key] for key in keys]
    return [
        {key: value for key, value in zip(keys, values)}
        for values in product(*values_lists)
    ]


def validate_dataset_config(dataset: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_DATASET_FIELDS if key not in dataset]
    if missing:
        raise ValueError(
            "dataset is missing required field(s): {}".format(", ".join(missing))
        )


def _as_dataset_list(raw_dataset: Any) -> List[Dict[str, Any]]:
    if raw_dataset is None:
        raise ValueError("experiment config must define dataset")
    if isinstance(raw_dataset, list):
        if not raw_dataset:
            raise ValueError("dataset list must contain at least one dataset")
        datasets = raw_dataset
    else:
        datasets = [raw_dataset]
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("dataset must be an object or a list of objects")
        validate_dataset_config(dataset)
    return datasets


def apply_dataset_paths(experiment: Dict[str, Any], dataset: Dict[str, Any]) -> None:
    """Derive every cross-stage path from the canonical dataset object."""
    experiment.setdefault("sigl", {})
    experiment.setdefault("d3pm", {})
    experiment.setdefault("digress", {})
    experiment.setdefault("ggme", {})

    experiment["sigl"]["Adjs_dir"] = dataset["train_graphs_path"]
    experiment["sigl"]["output_dir"] = dataset["sigl_graphons_dir"]
    experiment["d3pm"]["train_graphons_dir"] = dataset["sigl_graphons_dir"]
    experiment["d3pm"]["graphons_pt"] = dataset["ogad_generated_graphons_path"]
    experiment["digress"]["samples_path"] = dataset["digress_generated_samples_path"]
    experiment["ggme"]["test_graphs"] = dataset["test_graphs_path"]
    experiment["ggme"]["paper_graphs"] = dataset["paper_baseline_graphs_path"]
    experiment["ggme"]["graphons_pt"] = dataset["ogad_generated_graphons_path"]
    experiment["ggme"]["digress_samples"] = dataset["digress_generated_samples_path"]
    experiment["ggme"]["num_generated_graphs"] = dataset["generated_graph_count"]


def normalize_experiment_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return one experiment config with dataset-derived stage fields."""
    experiment = copy.deepcopy(raw)
    dataset = experiment.get("dataset")
    if isinstance(dataset, list):
        raise ValueError("normalize_experiment_config expects one dataset, not a list")
    if dataset is None:
        raise ValueError("experiment config must define dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    validate_dataset_config(dataset)

    apply_dataset_paths(experiment, dataset)
    return experiment


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or "dataset"


def expand_merged_experiment(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand datasets and the SIGL block into scalar experiment configs.

    D3PM remains a grid inside each scalar SIGL run, preserving the current
    pipeline shape.
    """
    experiments = []
    datasets = _as_dataset_list(raw.get("dataset"))
    sigl_runs = expand_block(raw["sigl"])
    for dataset in datasets:
        for sigl_run in sigl_runs:
            index = len(experiments)
            experiment = copy.deepcopy(raw)
            experiment["shared"] = dict(raw.get("shared") or {})
            experiment["dataset"] = copy.deepcopy(dataset)
            experiment["sigl"] = dict(sigl_run)
            experiment["d3pm"] = dict(raw.get("d3pm") or {})
            experiment["d3pm"]["image_size"] = [sigl_run["Res"]]
            experiment["name"] = "{}_nc{}".format(
                slugify(str(dataset["name"])),
                index,
            )
            experiments.append(normalize_experiment_config(experiment))
    return experiments


def assign_gpu(job_index: int, gpus: List[str]) -> str:
    if not gpus:
        raise ValueError("At least one GPU id is required")
    return gpus[job_index % len(gpus)]
