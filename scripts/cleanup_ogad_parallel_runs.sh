#!/usr/bin/env bash
# Remove per-run data/ dirs and graph pickle caches under parallel experiment folders.
set -euo pipefail

ROOT="${OGAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

usage() {
  echo "Usage: $0 <parallel_run_dir> [parallel_run_dir ...]" >&2
  echo "       $0 <run_dir> [run_dir ...]" >&2
  echo "  e.g. $0 runs/parallel_20260711_201704" >&2
  echo "  e.g. $0 runs/parallel_20260711_201704/runs/20260711_201704_nc0" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage

cleanup_run() {
  local rel="${1#./}"
  rel="${rel#"$ROOT"/}"

  if [[ ! -d "$rel" ]]; then
    echo "Not found: $rel" >&2
    exit 1
  fi

  docker run --rm -v "$ROOT:/repo" alpine sh -c \
    "rm -rf /repo/${rel}/data && rm -f /repo/${rel}/model_train_graphs.pickle /repo/${rel}/ggme_validation_graphs.pickle"

  echo "Cleaned $rel"
}

for arg in "$@"; do
  if [[ -d "$arg/runs" ]]; then
    shopt -s nullglob
    run_dirs=("$arg"/runs/*)
    shopt -u nullglob

    if [[ ${#run_dirs[@]} -eq 0 ]]; then
      echo "No runs found under $arg/runs" >&2
      exit 1
    fi

    for run_dir in "${run_dirs[@]}"; do
      cleanup_run "$run_dir"
    done
  else
    cleanup_run "$arg"
  fi
done

echo "Parallel run cleanup done."
