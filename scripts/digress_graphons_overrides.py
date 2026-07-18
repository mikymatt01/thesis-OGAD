#!/usr/bin/env python3
"""Emit Hydra overrides for the optional digress_graphons experiment section."""

import json
import sys
from pathlib import Path
from typing import List


FIELD_TO_OVERRIDE = {
    "num_bins": "dataset.num_bins",
    "quantization": "dataset.quantization",
    "resolution": "dataset.resolution",
    "batch_size": "train.batch_size",
    "accumulate_grad_batches": "train.accumulate_grad_batches",
    "final_samples": "general.final_model_samples_to_generate",
    "samples_batch_size": "general.samples_batch_size",
    "max_epochs": "train.n_epochs",
    "match_training_density": "dataset.match_training_density",
    "lr": "train.lr",
    "ema_decay": "train.ema_decay",
    "transition": "model.transition",
    "extra_features": "model.extra_features",
    "diffusion_steps": "model.diffusion_steps",
    "n_layers": "model.n_layers",
}


def format_value(value) -> str:
    """Render a JSON value the way Hydra's override grammar expects it."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def hydra_overrides(config_path: str) -> List[str]:
    if not config_path:
        return []
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    digress_config = config.get("digress_graphons") or {}
    if not isinstance(digress_config, dict):
        raise ValueError("digress_graphons must be a JSON object")

    unknown = sorted(set(digress_config) - set(FIELD_TO_OVERRIDE))
    if unknown:
        raise ValueError(
            "Unknown digress_graphons fields {}; they would be silently ignored".format(unknown)
        )

    return [
        f"{override_key}={format_value(digress_config[field])}"
        for field, override_key in FIELD_TO_OVERRIDE.items()
        if field in digress_config
    ]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: digress_graphons_overrides.py EXPERIMENT_JSON", file=sys.stderr)
        return 2
    for override in hydra_overrides(sys.argv[1]):
        print(override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
