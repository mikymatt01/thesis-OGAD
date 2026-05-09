#!/usr/bin/env python3
"""Esplode experiments/test_merged.json in N config scalari (uno per n_clusters)
e per ciascuno lancia scripts/run_ogad_experiment.sh."""
from itertools import product
import copy, json, subprocess
from pathlib import Path
import copy

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "experiments" / "test_merged.json"
RUN_SH = ROOT / "scripts" / "run_ogad_experiment.sh"

def expand_block(block: dict) -> list[dict]:
    keys = list(block.keys())
    values_lists = [block[k] for k in keys]  # ogni valore deve essere una lista
    combos = []
    for values in product(*values_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos

raw = json.loads(MERGED.read_text(encoding="utf-8"))


sigl_runs = [{"sigl": run} for run in expand_block(raw["sigl"])]

for index, sigl_run in enumerate(sigl_runs):
    test = copy.deepcopy(raw) | {"name": f"test_nc{index}"}
    test["sigl"] = sigl_run

    path = MERGED.with_name(f"test_nc{index}.json")

    path.write_text(json.dumps(test, indent=2), encoding="utf-8")
    subprocess.run([str(RUN_SH), str(path.relative_to(ROOT)), f"nc{index}"], cwd=ROOT, check=True)
