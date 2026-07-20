"""Pure-Python helpers for D3PM configuration, loss composition, and splitting."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

SUPPORTED_GRAPH_SELECTION_METRICS = {
    "mean_rank",
    "digress_target_ratio",
}


DEFAULT_PARAM_GRID = {
    "dim": [128, 256],
    "lr": [1e-5, 2e-5, 5e-5],
    "num_classes": [8],
    "image_size": [128],
    "batch_size": [16, 32],
    "hybrid_loss_coeff": [1e-2, 0.0, 1e-3],
    "num_workers": [0],
}

DEFAULT_TRAINING_CONFIG = {
    "validation_fraction": 0.2,
    "split_seed": 0,
    "patience": 10,
    "min_delta": 1e-4,
    "max_epochs": 200,
    "match_training_density": False,
}

DEFAULT_GRAPH_SELECTION_CONFIG = {
    "validation_fraction": 0.2,
    "split_seed": 0,
    "selection_metric": "mean_rank",
}


def combine_d3pm_losses(vb_loss: Any, ce_loss: Any, auxiliary_ce_coeff: float) -> Any:
    """Return the D3PM paper objective: variational bound plus weighted CE."""
    return vb_loss + auxiliary_ce_coeff * ce_loss


def deterministic_split(
    dataset_size: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return deterministic, disjoint train/validation indices."""
    if dataset_size < 2:
        raise ValueError("D3PM validation requires at least two graphon samples")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, min(dataset_size - 1, round(dataset_size * validation_fraction)))
    validation_indices = sorted(indices[:validation_size])
    train_indices = sorted(indices[validation_size:])
    return train_indices, validation_indices


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def load_experiment_config(config_path: str | None) -> tuple[dict, dict]:
    """Load the D3PM parameter grid and non-grid training controls."""
    resolved_path = config_path or os.environ.get("OGAD_EXPERIMENT_JSON")
    if not resolved_path:
        return (
            {key: list(values) for key, values in DEFAULT_PARAM_GRID.items()},
            dict(DEFAULT_TRAINING_CONFIG),
        )

    with open(resolved_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    shared = config.get("shared") or {}
    d3pm = dict(config.get("d3pm") or {})
    training = {**DEFAULT_TRAINING_CONFIG, **(config.get("d3pm_training") or {})}

    # Backward compatibility: old files placed n_epoch in the parameter grid.
    legacy_epochs = d3pm.pop("n_epoch", None)
    if "d3pm_training" not in config and legacy_epochs is not None:
        epoch_values = _as_list(legacy_epochs)
        if len(epoch_values) != 1:
            raise ValueError(
                "Legacy d3pm.n_epoch must contain one value; use "
                "d3pm_training.max_epochs for the new configuration"
            )
        training["max_epochs"] = int(epoch_values[0])

    grid = {key: _as_list(value) for key, value in d3pm.items()}
    for key in ("lr", "batch_size"):
        if key in shared and key not in grid:
            grid[key] = _as_list(shared[key])
    for key, default_values in DEFAULT_PARAM_GRID.items():
        grid.setdefault(key, list(default_values))

    training["validation_fraction"] = float(training["validation_fraction"])
    training["split_seed"] = int(training["split_seed"])
    training["patience"] = int(training["patience"])
    training["min_delta"] = float(training["min_delta"])
    training["max_epochs"] = int(training["max_epochs"])
    training["match_training_density"] = bool(training["match_training_density"])
    if training["patience"] < 1:
        raise ValueError("patience must be at least 1")
    if training["max_epochs"] < 1:
        raise ValueError("max_epochs must be at least 1")
    if training["min_delta"] < 0:
        raise ValueError("min_delta must be non-negative")

    return grid, training


def load_graph_selection_config(config_path: str | None) -> dict:
    """Load optional GGME-based graph selection controls."""
    resolved_path = config_path or os.environ.get("OGAD_EXPERIMENT_JSON")
    if not resolved_path:
        return {}

    with open(resolved_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    if "graph_selection" not in config:
        return {}

    graph_selection = {
        **DEFAULT_GRAPH_SELECTION_CONFIG,
        **(config.get("graph_selection") or {}),
    }
    graph_selection["validation_fraction"] = float(
        graph_selection["validation_fraction"]
    )
    graph_selection["split_seed"] = int(graph_selection["split_seed"])
    if not 0.0 < graph_selection["validation_fraction"] < 1.0:
        raise ValueError("graph_selection.validation_fraction must be between 0 and 1")
    if graph_selection["selection_metric"] not in SUPPORTED_GRAPH_SELECTION_METRICS:
        raise ValueError(
            "Unsupported graph_selection.selection_metric '{}'. Supported values: {}".format(
                graph_selection["selection_metric"],
                ", ".join(sorted(SUPPORTED_GRAPH_SELECTION_METRICS)),
            )
        )
    return graph_selection


def update_run_manifest(run_dir: str | None, payload: dict) -> None:
    """Merge runtime D3PM metadata into the run manifest, when available."""
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = manifest.setdefault("runtime", {})
    runtime["d3pm"] = payload
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
