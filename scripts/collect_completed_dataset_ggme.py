#!/usr/bin/env python3
"""Collect GGME metrics for datasets whose scheduled OGAD runs are complete.

The script treats queue entries as the set of scheduled runs. A dataset is
included only when every queued OGAD run for that dataset across the selected
parallel directories is marked complete and has exactly the three final
``stage == "ggme"`` metric rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DESCRIPTORS = (
    "degree_distribution",
    "clustering_coefficient",
    "normalised_laplacian_spectrum",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def experiment_path(root: Path, slug: str, status: dict[str, Any]) -> Path:
    if "experiment" in status:
        candidate = root / status["experiment"]
        if candidate.exists():
            return candidate
    return root / "experiments" / f"{slug}.json"


def dataset_from_experiment(path: Path) -> tuple[str, int | None]:
    experiment = load_json(path)
    return (
        experiment["dataset"]["name"],
        experiment.get("sigl", {}).get("n_clusters"),
    )


def run_dir_for_slug(root: Path, slug: str) -> Path | None:
    matches = sorted((root / "runs").glob(f"*_{slug}"))
    return matches[-1] if matches else None


def ggme_rows(metrics_path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with metrics_path.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("stage") == "ggme":
                rows[record["descriptor"]] = record["mmd"]
    return rows


def ranked_models(mmd: dict[str, float]) -> list[tuple[str, float, int]]:
    ordered = sorted(mmd.items(), key=lambda item: item[1])
    ranked: list[tuple[str, float, int]] = []
    previous_value: float | None = None
    current_rank = 0
    for index, (model, value) in enumerate(ordered, start=1):
        if previous_value is None or value != previous_value:
            current_rank = index
        ranked.append((model, value, current_rank))
        previous_value = value
    return ranked


def mean_ogad_run_ranks(runs: list[dict[str, Any]]) -> None:
    for descriptor in DESCRIPTORS:
        ordered = sorted(runs, key=lambda run: run["ggme"][descriptor]["OGAD"])
        previous_value: float | None = None
        current_rank = 0
        for index, run in enumerate(ordered, start=1):
            value = run["ggme"][descriptor]["OGAD"]
            if previous_value is None or value != previous_value:
                current_rank = index
            run.setdefault("ogad_run_ranks", {})[descriptor] = current_rank
            previous_value = value

    for run in runs:
        run["ogad_run_mean_rank"] = sum(
            run["ogad_run_ranks"][descriptor] for descriptor in DESCRIPTORS
        ) / len(DESCRIPTORS)
        run["ogad_model_mean_rank"] = sum(
            dict((model, rank) for model, _value, rank in ranked_models(run["ggme"][descriptor]))["OGAD"]
            for descriptor in DESCRIPTORS
        ) / len(DESCRIPTORS)


def format_value(value: float) -> str:
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.3e}"
    return f"{value:.4f}"


def collect(roots: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []

    for root in roots:
        for status_path in sorted((root / "queue").glob("*/status.json")):
            slug = status_path.parent.name
            status = load_json(status_path)
            exp_path = experiment_path(root, slug, status)
            dataset, n_clusters = dataset_from_experiment(exp_path)
            run_dir = run_dir_for_slug(root, slug)

            run_record: dict[str, Any] = {
                "batch": root.name,
                "slug": slug,
                "dataset": dataset,
                "n_clusters": n_clusters,
                "status": status.get("status"),
                "run_dir": str(run_dir) if run_dir else "",
            }

            if status.get("status") != "complete":
                exclusions.append({**run_record, "reason": "status is not complete"})
                by_dataset[dataset].append({**run_record, "complete_for_report": False})
                continue
            if run_dir is None:
                exclusions.append({**run_record, "reason": "missing run directory"})
                by_dataset[dataset].append({**run_record, "complete_for_report": False})
                continue
            metrics_path = run_dir / "metrics.jsonl"
            if not metrics_path.exists():
                exclusions.append({**run_record, "reason": "missing metrics.jsonl"})
                by_dataset[dataset].append({**run_record, "complete_for_report": False})
                continue

            ggme = ggme_rows(metrics_path)
            missing = [descriptor for descriptor in DESCRIPTORS if descriptor not in ggme]
            if missing:
                exclusions.append(
                    {**run_record, "reason": "missing ggme descriptors: " + ",".join(missing)}
                )
                by_dataset[dataset].append({**run_record, "complete_for_report": False})
                continue

            by_dataset[dataset].append(
                {
                    **run_record,
                    "run_id": run_dir.name,
                    "metrics_path": str(metrics_path),
                    "ggme": ggme,
                    "complete_for_report": True,
                }
            )

    return by_dataset, exclusions


def write_outputs(
    by_dataset: dict[str, list[dict[str, Any]]],
    exclusions: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    eligible: dict[str, list[dict[str, Any]]] = {
        dataset: runs
        for dataset, runs in by_dataset.items()
        if runs and all(run.get("complete_for_report") for run in runs)
    }
    for runs in eligible.values():
        mean_ogad_run_ranks(runs)

    best_runs = {
        dataset: sorted(
            runs,
            key=lambda run: (run["ogad_run_mean_rank"], run["ogad_model_mean_rank"], run["run_id"]),
        )[0]
        for dataset, runs in eligible.items()
    }

    metrics_csv = output_dir / "ggme_completed_datasets_best_ogad.csv"
    status_csv = output_dir / "ggme_completed_dataset_filter_status.csv"
    report_md = output_dir / "ggme_completed_datasets_best_ogad.md"

    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "best_ogad_run_id",
                "best_ogad_batch",
                "n_clusters",
                "descriptor",
                "model",
                "mmd",
                "rank_in_descriptor",
            ]
        )
        for dataset in sorted(best_runs):
            run = best_runs[dataset]
            for descriptor in DESCRIPTORS:
                for model, value, rank in ranked_models(run["ggme"][descriptor]):
                    writer.writerow(
                        [
                            dataset,
                            run["run_id"],
                            run["batch"],
                            run["n_clusters"],
                            descriptor,
                            model,
                            value,
                            rank,
                        ]
                    )

    with status_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "batch", "slug", "status", "included", "reason"])
        for dataset in sorted(by_dataset):
            for run in by_dataset[dataset]:
                reason = ""
                if not run.get("complete_for_report"):
                    reason = next(
                        (
                            item["reason"]
                            for item in exclusions
                            if item["dataset"] == dataset
                            and item["batch"] == run["batch"]
                            and item["slug"] == run["slug"]
                        ),
                        "not complete for report",
                    )
                writer.writerow(
                    [
                        dataset,
                        run["batch"],
                        run["slug"],
                        run.get("status", ""),
                        run.get("complete_for_report", False),
                        reason,
                    ]
                )

    lines: list[str] = []
    lines.append("# GGME metrics for completed datasets")
    lines.append("")
    lines.append(
        "Included datasets are those where every queued OGAD run across the selected parallel directories is complete and has all three final GGME descriptors."
    )
    lines.append("")
    lines.append(f"Eligible datasets: **{len(best_runs)}** / {len(by_dataset)}.")
    lines.append("")
    if best_runs:
        lines.append("## Best OGAD Runs")
        lines.append("")
        lines.append(
            "| Dataset | Best OGAD run | Batch | n_clusters | OGAD run mean rank | OGAD model mean rank |"
        )
        lines.append("|---|---|---|---:|---:|---:|")
        for dataset in sorted(best_runs):
            run = best_runs[dataset]
            lines.append(
                "| {} | `{}` | `{}` | {} | {:.3f} | {:.3f} |".format(
                    dataset,
                    run["run_id"],
                    run["batch"],
                    run["n_clusters"],
                    run["ogad_run_mean_rank"],
                    run["ogad_model_mean_rank"],
                )
            )
        lines.append("")

        for dataset in sorted(best_runs):
            run = best_runs[dataset]
            lines.append(f"## {dataset}")
            lines.append("")
            lines.append(
                f"Best OGAD run: `{run['run_id']}` (`{run['batch']}`, n_clusters={run['n_clusters']})."
            )
            lines.append("")
            lines.append("| Descriptor | Model | MMD | Rank |")
            lines.append("|---|---|---:|---:|")
            for descriptor in DESCRIPTORS:
                for model, value, rank in ranked_models(run["ggme"][descriptor]):
                    lines.append(
                        "| {} | {} | {} | {} |".format(
                            descriptor,
                            model,
                            format_value(value),
                            rank,
                        )
                    )
            lines.append("")

    lines.append("## Excluded Datasets")
    lines.append("")
    lines.append("| Dataset | Reason |")
    lines.append("|---|---|")
    for dataset in sorted(by_dataset):
        if dataset in best_runs:
            continue
        reasons = sorted(
            {
                item["reason"]
                for item in exclusions
                if item["dataset"] == dataset
            }
        )
        lines.append(f"| {dataset} | {'; '.join(reasons)} |")
    lines.append("")
    lines.append(f"CSV metrics: `{metrics_csv.name}`")
    lines.append(f"CSV filter status: `{status_csv.name}`")
    report_md.write_text("\n".join(lines) + "\n")

    return report_md, metrics_csv, status_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            Path("runs/parallel_20260729_144516"),
            Path("runs/parallel_20260803_063318"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    roots = [root.resolve() for root in args.roots]
    output_dir = args.output_dir.resolve()
    by_dataset, exclusions = collect(roots)
    report, metrics, status = write_outputs(by_dataset, exclusions, output_dir)
    print(report)
    print(metrics)
    print(status)


if __name__ == "__main__":
    main()
