#!/usr/bin/env python3
"""Rank OGAD runs by Paper-normalized structural GGME score."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ggme"))

from selection_utils import DESCRIPTORS, PAPER_TARGET_MMD, structural_ratio_score  # noqa: E402

PROMISING_DEGREE_MMD_MAX = 0.18


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_ggme(records: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    mmd = {
        "OGAD": {descriptor: None for descriptor in DESCRIPTORS},
        "Paper": {descriptor: PAPER_TARGET_MMD[descriptor] for descriptor in DESCRIPTORS},
    }
    for record in records:
        if record.get("stage") != "ggme":
            continue
        descriptor = record.get("descriptor")
        if descriptor not in DESCRIPTORS:
            continue
        values = record.get("mmd") or {}
        mmd["OGAD"][descriptor] = values.get("OGAD")
        mmd["Paper"][descriptor] = values.get("Paper")
    return mmd


def manifest_sigl(run_dir: Path) -> dict[str, Any]:
    manifest = run_dir / "manifest.json"
    experiment = run_dir / "experiment.json"
    try:
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            return (payload.get("experiment") or {}).get("sigl") or {}
        if experiment.is_file():
            payload = json.loads(experiment.read_text(encoding="utf-8"))
            return payload.get("sigl") or {}
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def collect_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        values = extract_ggme(parse_jsonl(metrics_path))
        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "run_path": str(run_dir.relative_to(root)),
            "complete_ggme": all(values["OGAD"][descriptor] is not None for descriptor in DESCRIPTORS),
        }
        sigl = manifest_sigl(run_dir)
        for key in (
            "h_size",
            "n_clusters",
            "bootstrap_replicates",
            "include_original_clusters",
            "degree_correction",
            "degree_correction_iters",
        ):
            row[f"sigl_{key}"] = sigl.get(key)
        for descriptor in DESCRIPTORS:
            row[f"ogad_{descriptor}"] = values["OGAD"][descriptor]
            row[f"paper_{descriptor}"] = values["Paper"][descriptor]
        if row["complete_ggme"]:
            targets = {
                descriptor: float(values["Paper"][descriptor])
                for descriptor in DESCRIPTORS
            }
            scored = structural_ratio_score(values["OGAD"], targets=targets)
            row.update(scored)
            row["promising"] = (
                row["ogad_degree_distribution"] <= PROMISING_DEGREE_MMD_MAX
                and row["ogad_clustering_coefficient"] < 408.61717421124763
                and row["ogad_normalised_laplacian_spectrum"] < 6.1019142398373845
            )
        else:
            row["structural_ratios"] = {}
            row["structural_ratio_score"] = math.nan
            row["promising"] = False
        rows.append(row)
    return rows


def fmt(value: Any, precision: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], top: int | None) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            math.isnan(row["structural_ratio_score"]),
            row["structural_ratio_score"],
        ),
    )
    if top:
        ranked = ranked[:top]
    header = (
        f"{'rank':>4} {'score':>9} {'ok':>3} | {'h':>3} {'nc':>4} {'boot':>4} "
        f"{'orig':>5} {'degcorr':>12} | {'deg':>8} {'clust':>10} {'lap':>8} | "
        f"{'deg/P':>8} {'cl/P':>8} {'lap/P':>8} | run_id"
    )
    print(header)
    print("-" * len(header))
    for index, row in enumerate(ranked, 1):
        ratios = row["structural_ratios"]
        print(
            f"{index:>4} {fmt(row['structural_ratio_score']):>9} "
            f"{str(row['promising']):>3} | "
            f"{fmt(row.get('sigl_h_size')):>3} {fmt(row.get('sigl_n_clusters')):>4} "
            f"{fmt(row.get('sigl_bootstrap_replicates')):>4} "
            f"{fmt(row.get('sigl_include_original_clusters')):>5} "
            f"{fmt(row.get('sigl_degree_correction')):>12} | "
            f"{fmt(row.get('ogad_degree_distribution')):>8} "
            f"{fmt(row.get('ogad_clustering_coefficient')):>10} "
            f"{fmt(row.get('ogad_normalised_laplacian_spectrum')):>8} | "
            f"{fmt(ratios.get('degree_distribution')):>8} "
            f"{fmt(ratios.get('clustering_coefficient')):>8} "
            f"{fmt(ratios.get('normalised_laplacian_spectrum')):>8} | "
            f"{row['run_id']}"
        )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_id",
        "run_path",
        "complete_ggme",
        "promising",
        "structural_ratio_score",
        "sigl_h_size",
        "sigl_n_clusters",
        "sigl_bootstrap_replicates",
        "sigl_include_original_clusters",
        "sigl_degree_correction",
        "sigl_degree_correction_iters",
    ]
    for descriptor in DESCRIPTORS:
        fieldnames.extend(
            [
                f"ogad_{descriptor}",
                f"paper_{descriptor}",
                f"ratio_{descriptor}",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for descriptor in DESCRIPTORS:
                out[f"ratio_{descriptor}"] = row["structural_ratios"].get(descriptor)
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = collect_runs(args.root)
    if not rows:
        print(f"No metrics.jsonl files found under {args.root}")
        return 1
    print_table(rows, args.top)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
