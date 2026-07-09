#!/usr/bin/env python3
"""Esplode experiments/test_merged.json in N config scalari (uno per n_clusters)
e per ciascuno lancia scripts/run_ogad_experiment.sh."""
import copy
import json
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "experiments" / "test_merged.json"
RUN_SH = ROOT / "scripts" / "run_ogad_experiment.sh"

def expand_block(block: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(block.keys())
    values_lists = [block[k] for k in keys]  # ogni valore deve essere una lista
    combos = []
    for values in product(*values_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos

raw = json.loads(MERGED.read_text(encoding="utf-8"))


sigl_runs = expand_block(raw["sigl"])

for index, sigl_run in enumerate(sigl_runs):
    test = copy.deepcopy(raw) | {"name": f"test_nc{index}"}
    test["sigl"] = sigl_run
    test["d3pm"]["image_size"] = [sigl_run['Res']]

    path = MERGED.with_name(f"test_nc{index}.json")
    path.write_text(json.dumps(test, indent=2), encoding="utf-8")
    subprocess.run([str(RUN_SH), str(path.relative_to(ROOT)), f"nc{index}"], cwd=ROOT, check=True)
