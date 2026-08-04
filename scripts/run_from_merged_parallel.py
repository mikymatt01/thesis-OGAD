#!/usr/bin/env python3
"""Run expanded OGAD experiments in parallel, with one worker per GPU."""

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_grid import assign_gpu, expand_merged_experiment, slugify


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
        name = experiment.get("name") or "test_nc{}".format(index)
        experiment_path = generated_dir / "{}.json".format(name)
        experiment_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
        jobs.append(
            {
                "index": index,
                "slug": name,
                "experiment_path": experiment_path,
                "experiment_rel": experiment_path.relative_to(ROOT),
                "experiment": experiment,
                "digress_enabled": bool(
                    (experiment.get("digress") or {}).get("enabled", False)
                ),
                "baselines_enabled": bool(
                    (experiment.get("baselines") or {}).get("enabled", False)
                ),
            }
        )
    return jobs


def shared_digress_key(job: Dict[str, Any]) -> str:
    experiment = job["experiment"]
    digress = dict(experiment.get("digress") or {})
    digress.pop("samples_path", None)
    payload = {
        "dataset": experiment.get("dataset") or {},
        "digress": digress,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_shared_digress_plan(
    jobs: List[Dict[str, Any]],
    runs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        if not job["digress_enabled"]:
            continue
        key = shared_digress_key(job)
        if key in plan:
            continue
        dataset = slugify(
            str((job["experiment"].get("dataset") or {}).get("name", "dataset"))
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        slug = "{}_{}".format(dataset, digest)
        host_dir = runs_root / "shared" / "digress" / slug
        plan[key] = {
            "key": key,
            "slug": slug,
            "gpu_id": None,
            "experiment_rel": job["experiment_rel"],
            "host_dir": host_dir,
        }
    return plan


def shared_baselines_key(job: Dict[str, Any]) -> str:
    experiment = job["experiment"]
    dataset = experiment.get("dataset") or {}
    baselines = dict(experiment.get("baselines") or {})
    baselines.pop("output_dir", None)
    payload = {
        "dataset": {
            "name": dataset.get("name"),
            "train_graphs_path": dataset.get("train_graphs_path"),
            "test_graphs_path": dataset.get("test_graphs_path"),
            "generated_graph_count": dataset.get("generated_graph_count"),
        },
        "baselines": baselines,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_shared_baselines_plan(
    jobs: List[Dict[str, Any]],
    runs_root: Path,
) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        if not job["baselines_enabled"]:
            continue
        key = shared_baselines_key(job)
        if key in plan:
            continue
        dataset = slugify(
            str((job["experiment"].get("dataset") or {}).get("name", "dataset"))
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        slug = "{}_{}".format(dataset, digest)
        host_dir = runs_root / "shared" / "baselines" / slug
        plan[key] = {
            "key": key,
            "slug": slug,
            "gpu_id": None,
            "experiment_rel": job["experiment_rel"],
            "host_dir": host_dir,
        }
    return plan


def run_shared_digress(
    item: Dict[str, Any],
    gpu_id: str,
    runs_root: Path,
    max_retries: int,
) -> Dict[str, Any]:
    item["host_dir"].mkdir(parents=True, exist_ok=True)
    status_path = item["host_dir"] / "status.json"
    attempts = []
    for attempt in range(max_retries + 1):
        log_path = item["host_dir"] / "launcher_attempt_{}.log".format(attempt)
        payload = {
            "status": "running",
            "slug": item["slug"],
            "gpu_id": gpu_id,
            "attempt": attempt,
            "max_retries": max_retries,
            "experiment": str(item["experiment_rel"]),
            "started_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        write_status(status_path, payload)
        env = os.environ.copy()
        env["OGAD_GPU_ID"] = gpu_id
        env["OGAD_CLEANUP"] = "0"
        env["OGAD_ONLY_DIGRESS"] = "1"
        env["OGAD_DIGRESS_HOST_DIR"] = str(item["host_dir"])
        env["OGAD_RUNS_ROOT"] = str(runs_root / "shared_runs")
        cmd = [str(RUN_SH), str(item["experiment_rel"]), "digress_{}".format(item["slug"])]
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
        attempts.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "log": str(log_path),
                "duration_seconds": duration_seconds,
            }
        )
        if completed.returncode == 0:
            result = {
                "status": "complete",
                "slug": item["slug"],
                "gpu_id": gpu_id,
                "host_dir": str(item["host_dir"]),
                "attempts": attempts,
                "completed_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            write_status(status_path, result)
            return result

    result = {
        "status": "failed",
        "slug": item["slug"],
        "gpu_id": gpu_id,
        "host_dir": str(item["host_dir"]),
        "attempts": attempts,
        "failed_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    write_status(status_path, result)
    return result


def run_shared_baselines(
    item: Dict[str, Any],
    gpu_id: str,
    runs_root: Path,
    max_retries: int,
) -> Dict[str, Any]:
    item["host_dir"].mkdir(parents=True, exist_ok=True)
    status_path = item["host_dir"] / "status.json"
    attempts = []
    for attempt in range(max_retries + 1):
        log_path = item["host_dir"] / "launcher_attempt_{}.log".format(attempt)
        payload = {
            "status": "running",
            "slug": item["slug"],
            "gpu_id": gpu_id,
            "attempt": attempt,
            "max_retries": max_retries,
            "experiment": str(item["experiment_rel"]),
            "started_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        write_status(status_path, payload)
        env = os.environ.copy()
        env["OGAD_GPU_ID"] = gpu_id
        env["OGAD_CLEANUP"] = "0"
        env["OGAD_ONLY_BASELINES"] = "1"
        env["OGAD_BASELINES_HOST_DIR"] = str(item["host_dir"])
        env["OGAD_RUNS_ROOT"] = str(runs_root / "shared_runs")
        cmd = [str(RUN_SH), str(item["experiment_rel"]), "baselines_{}".format(item["slug"])]
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
        attempts.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "log": str(log_path),
                "duration_seconds": duration_seconds,
            }
        )
        if completed.returncode == 0:
            result = {
                "status": "complete",
                "slug": item["slug"],
                "gpu_id": gpu_id,
                "host_dir": str(item["host_dir"]),
                "attempts": attempts,
                "completed_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            write_status(status_path, result)
            return result

    result = {
        "status": "failed",
        "slug": item["slug"],
        "gpu_id": gpu_id,
        "host_dir": str(item["host_dir"]),
        "attempts": attempts,
        "failed_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    write_status(status_path, result)
    return result


def run_job(
    job: Dict[str, Any],
    gpu_id: str,
    runs_root: Path,
    max_retries: int,
    shared_digress_dir: Optional[Path] = None,
    shared_baselines_dir: Optional[Path] = None,
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
        if shared_digress_dir is not None and job["digress_enabled"]:
            env["OGAD_SKIP_DIGRESS"] = "1"
            env["OGAD_DIGRESS_HOST_DIR"] = str(shared_digress_dir)
        if shared_baselines_dir is not None and job["baselines_enabled"]:
            env["OGAD_SKIP_BASELINES"] = "1"
            env["OGAD_BASELINES_HOST_DIR"] = str(shared_baselines_dir)
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
    shared_digress_dirs: Dict[str, Path],
    shared_baselines_dirs: Dict[str, Path],
) -> List[Dict[str, Any]]:
    results = []
    while True:
        try:
            job = pending.get_nowait()
        except queue.Empty:
            return results
        try:
            shared_dir = None
            if job["digress_enabled"]:
                shared_dir = shared_digress_dirs.get(shared_digress_key(job))
            shared_baselines_dir = None
            if job["baselines_enabled"]:
                shared_baselines_dir = shared_baselines_dirs.get(shared_baselines_key(job))
            results.append(
                run_job(
                    job,
                    gpu_id,
                    runs_root,
                    max_retries,
                    shared_dir,
                    shared_baselines_dir,
                )
            )
        finally:
            pending.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-share-digress",
        action="store_true",
        help="Run DiGress inside every expanded job instead of sharing one run per dataset.",
    )
    parser.add_argument(
        "--no-share-baselines",
        action="store_true",
        help="Run classical baselines inside every expanded job instead of sharing one run per dataset.",
    )
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
    shared_digress_plan = (
        build_shared_digress_plan(jobs, runs_root)
        if not args.no_share_digress
        else {}
    )
    shared_digress_dirs = {
        key: item["host_dir"] for key, item in shared_digress_plan.items()
    }
    shared_baselines_plan = (
        build_shared_baselines_plan(jobs, runs_root)
        if not args.no_share_baselines
        else {}
    )
    shared_baselines_dirs = {
        key: item["host_dir"] for key, item in shared_baselines_plan.items()
    }

    if args.dry_run:
        print("runs_root={}".format(runs_root))
        if shared_digress_plan:
            for index, item in enumerate(shared_digress_plan.values()):
                print(
                    "shared_digress={} gpu={} experiment={} host_dir={}".format(
                        item["slug"],
                        assign_gpu(index, gpus),
                        item["experiment_rel"],
                        item["host_dir"],
                    )
                )
        if shared_baselines_plan:
            for index, item in enumerate(shared_baselines_plan.values()):
                print(
                    "shared_baselines={} gpu={} experiment={} host_dir={}".format(
                        item["slug"],
                        assign_gpu(index, gpus),
                        item["experiment_rel"],
                        item["host_dir"],
                    )
                )
        for index, job in enumerate(jobs):
            shared_dir = None
            if job["digress_enabled"]:
                shared_dir = shared_digress_dirs.get(shared_digress_key(job))
            shared_baselines_dir = None
            if job["baselines_enabled"]:
                shared_baselines_dir = shared_baselines_dirs.get(shared_baselines_key(job))
            print(
                "{} gpu={} experiment={} baselines={} shared_baselines={} shared_digress={}".format(
                    job["slug"],
                    assign_gpu(index, gpus),
                    job["experiment_rel"],
                    "on" if job["baselines_enabled"] else "off",
                    shared_baselines_dir or "<per-job>",
                    shared_dir or "<per-job>",
                )
            )
        print(
            "jobs={} workers={} shared_digress_runs={} shared_baselines_runs={}".format(
                len(jobs),
                len(gpus),
                len(shared_digress_plan),
                len(shared_baselines_plan),
            )
        )
        return 0

    shared_baselines_results = []
    for index, item in enumerate(shared_baselines_plan.values()):
        gpu_id = assign_gpu(index, gpus)
        item["gpu_id"] = gpu_id
        result = run_shared_baselines(item, gpu_id, runs_root, args.max_retries)
        shared_baselines_results.append(result)
    failed_baselines = [
        row for row in shared_baselines_results if row["status"] != "complete"
    ]
    if failed_baselines:
        summary = {
            "runs_root": str(runs_root),
            "jobs": len(jobs),
            "complete": 0,
            "failed": len(jobs),
            "shared_baselines": shared_baselines_results,
            "shared_digress": [],
            "results": [],
        }
        write_status(runs_root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 1

    shared_digress_results = []
    for index, item in enumerate(shared_digress_plan.values()):
        gpu_id = assign_gpu(index, gpus)
        item["gpu_id"] = gpu_id
        result = run_shared_digress(item, gpu_id, runs_root, args.max_retries)
        shared_digress_results.append(result)
    failed_shared = [
        row for row in shared_digress_results if row["status"] != "complete"
    ]
    if failed_shared:
        summary = {
            "runs_root": str(runs_root),
            "jobs": len(jobs),
            "complete": 0,
            "failed": len(jobs),
            "shared_baselines": shared_baselines_results,
            "shared_digress": shared_digress_results,
            "results": [],
        }
        write_status(runs_root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 1

    pending: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    for job in jobs:
        pending.put(job)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                worker_loop,
                gpu_id,
                pending,
                runs_root,
                args.max_retries,
                shared_digress_dirs,
                shared_baselines_dirs,
            )
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
        "shared_baselines": shared_baselines_results,
        "shared_digress": shared_digress_results,
        "results": all_results,
    }
    write_status(runs_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
