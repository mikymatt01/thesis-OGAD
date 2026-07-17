#!/usr/bin/env python3
"""Emit Hydra overrides for the optional digress_graphons experiment section."""

import json
import sys
from pathlib import Path
from typing import List


FIELD_TO_OVERRIDE = {
    "num_bins": "dataset.num_bins",
    "resolution": "dataset.resolution",
    "batch_size": "train.batch_size",
    "accumulate_grad_batches": "train.accumulate_grad_batches",
    "final_samples": "general.final_model_samples_to_generate",
    "max_epochs": "train.n_epochs",
    "match_training_density": "dataset.match_training_density",
}


def hydra_overrides(config_path: str) -> List[str]:
    if not config_path:
        return []
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    digress_config = config.get("digress_graphons") or {}
    if not isinstance(digress_config, dict):
        raise ValueError("digress_graphons must be a JSON object")

    overrides = []
    for field, override_key in FIELD_TO_OVERRIDE.items():
        if field in digress_config:
            value = digress_config[field]
            if isinstance(value, bool):
                value = str(value).lower()
            overrides.append(f"{override_key}={value}")
    return overrides


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: digress_graphons_overrides.py EXPERIMENT_JSON", file=sys.stderr)
        return 2
    for override in hydra_overrides(sys.argv[1]):
        print(override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
