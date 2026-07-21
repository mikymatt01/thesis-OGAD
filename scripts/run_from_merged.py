#!/usr/bin/env python3
"""Esplode experiments/test_merged.json in config scalari SIGL.
e per ciascuno lancia scripts/run_ogad_experiment.sh."""
import copy
import json
import subprocess
from pathlib import Path

from experiment_grid import expand_merged_experiment

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "experiments" / "test_merged.json"
RUN_SH = ROOT / "scripts" / "run_ogad_experiment.sh"

raw = json.loads(MERGED.read_text(encoding="utf-8"))

for index, test in enumerate(expand_merged_experiment(raw)):
    name = test.get("name") or f"test_nc{index}"
    path = MERGED.with_name(f"{name}.json")
    path.write_text(json.dumps(copy.deepcopy(test), indent=2), encoding="utf-8")
    subprocess.run([str(RUN_SH), str(path.relative_to(ROOT)), name], cwd=ROOT, check=True)
