#!/usr/bin/env python3
"""Run expanded OGAD experiments in parallel, with one worker per GPU."""

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from experiment_grid import assign_gpu, expand_merged_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED = ROOT / "experiments" / "test_merged.json"
RUN_SH = ROOT / "scripts" / "run_ogad_experiment.sh"


def parse_gpus(value: str) -> List[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id must be provided")
    return gpus


def write_status(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_jobs(merged_path: Path, work_dir: Path) -> List[Dict[str, Any]]:
    raw = json.loads(merged_path.read_text(encoding="utf-8"))
    jobs = []
    generated_dir = work_dir / "experiments"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for index, experiment in enumerate(expand_merged_experiment(raw)):
        name = "test_nc{}".format(index)
        experiment_path = generated_dir / "{}.json".format(name)
        experiment_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
        jobs.append(
            {
                "index": index,
                "slug": "nc{}".format(index),
                "experiment_path": experiment_path,
                "experiment_rel": experiment_path.relative_to(ROOT),
            }
        )
    return jobs


def run_job(
    job: Dict[str, Any],
    gpu_id: str,
    runs_root: Path,
    max_retries: int,
) -> Dict[str, Any]:
    job_dir = runs_root / "queue" / job["slug"]
    status_path = job_dir / "status.json"
    attempts = []
    for attempt in range(max_retries + 1):
        attempt_dir = job_dir / "attempt_{}".format(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "launcher.log"
        error_path = attempt_dir / "error.log"
        payload = {
            "status": "running",
            "slug": job["slug"],
            "gpu_id": gpu_id,
            "attempt": attempt,
            "max_retries": max_retries,
            "experiment": str(job["experiment_rel"]),
            "started_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        write_status(status_path, payload)
        env = os.environ.copy()
        env["OGAD_GPU_ID"] = gpu_id
        env["OGAD_CLEANUP"] = "0"
        env["OGAD_RUNS_ROOT"] = str(runs_root / "runs")
        cmd = [str(RUN_SH), str(job["experiment_rel"]), job["slug"]]
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        duration_seconds = round(time.time() - started, 3)
        attempt_result = {
            "attempt": attempt,
            "returncode": completed.returncode,
            "log": str(log_path),
            "duration_seconds": duration_seconds,
        }
        attempts.append(attempt_result)
        if completed.returncode == 0:
            result = {
                "status": "complete",
                "slug": job["slug"],
                "gpu_id": gpu_id,
                "attempts": attempts,
                "completed_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            write_status(status_path, result)
            return result

        error_path.write_text(
            "Command failed with return code {}. See {}.\n".format(
                completed.returncode, log_path
            ),
            encoding="utf-8",
        )

    result = {
        "status": "failed",
        "slug": job["slug"],
        "gpu_id": gpu_id,
        "attempts": attempts,
        "failed_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    write_status(status_path, result)
    return result


def worker_loop(
    gpu_id: str,
    pending: "queue.Queue[Dict[str, Any]]",
    runs_root: Path,
    max_retries: int,
) -> List[Dict[str, Any]]:
    results = []
    while True:
        try:
            job = pending.get_nowait()
        except queue.Empty:
            return results
        try:
            results.append(run_job(job, gpu_id, runs_root, max_retries))
        finally:
            pending.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root for this parallel batch; defaults to runs/parallel_<timestamp>.",
    )
    args = parser.parse_args()
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")

    gpus = parse_gpus(args.gpus)
    batch_id = dt.datetime.utcnow().strftime("parallel_%Y%m%d_%H%M%S")
    runs_root = args.runs_root or (ROOT / "runs" / batch_id)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    jobs = build_jobs(args.merged, runs_root)

    if args.dry_run:
        print("runs_root={}".format(runs_root))
        for index, job in enumerate(jobs):
            print(
                "{} gpu={} experiment={}".format(
                    job["slug"],
                    assign_gpu(index, gpus),
                    job["experiment_rel"],
                )
            )
        print("jobs={} workers={}".format(len(jobs), len(gpus)))
        return 0

    pending: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    for job in jobs:
        pending.put(job)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(worker_loop, gpu_id, pending, runs_root, args.max_retries)
            for gpu_id in gpus
        ]
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    summary = {
        "runs_root": str(runs_root),
        "jobs": len(jobs),
        "complete": sum(1 for row in all_results if row["status"] == "complete"),
        "failed": sum(1 for row in all_results if row["status"] == "failed"),
        "results": all_results,
    }
    write_status(runs_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
