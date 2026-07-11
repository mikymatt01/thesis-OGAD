#!/usr/bin/env bash
# Run SIGL → D3PM → GGME (OGAD-only compose) with experiment JSON, logs, and run dir.
# Intended host: Linux with Docker Compose v2 and GNU coreutils (date, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP_REL="${1:-experiments/example.json}"
SLUG="${2:-}"
OGAD_CLEANUP="${OGAD_CLEANUP:-0}"
OGAD_GPU_ID="${OGAD_GPU_ID:-}"
RUNS_ROOT="${OGAD_RUNS_ROOT:-$ROOT/runs}"

EXP_ABS="$ROOT/$EXP_REL"
if [[ ! -f "$EXP_ABS" ]]; then
  echo "Experiment file not found: $EXP_ABS" >&2
  exit 1
fi

EXP_BASENAME="$(basename "$EXP_REL")"
EXP_DIR="$(dirname "$EXP_ABS")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
if [[ -n "$SLUG" ]]; then
  RUN_ID="${RUN_ID}_${SLUG}"
fi

RUN_DIR="$RUNS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
mkdir -p \
  "$RUN_DIR/data/train/graphons" \
  "$RUN_DIR/data/predictions/graphons" \
  "$RUN_DIR/data/models"

python3 "$ROOT/scripts/ogad_write_manifest.py" "$EXP_ABS" "$RUN_DIR" "$RUN_ID"

COMPOSE_PROJECT_NAME="${OGAD_COMPOSE_PROJECT_NAME:-ogad_${RUN_ID}}"
COMPOSE=(docker compose -p "$COMPOSE_PROJECT_NAME" -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.ogad-only.yml")
GPU_ENV=()
if [[ -n "$OGAD_GPU_ID" ]]; then
  GPU_ENV=(-e "NVIDIA_VISIBLE_DEVICES=$OGAD_GPU_ID")
fi

SIGL_CMD=(poetry run python Main.py
  --config "/app/experiments/$EXP_BASENAME"
)

D3PM_CMD=(bash -lc
  "poetry run python ogad_model_finetune.py --config /app/experiments/$EXP_BASENAME && poetry run python ogad_model_inference.py"
)

GGME_CMD=(poetry run python main.py
  --config "/app/experiments/$EXP_BASENAME"
  --skip-digress
)

run_stage() {
  local service="$1"
  shift
  local logfile="$1"
  shift
  "${COMPOSE[@]}" run --rm --no-deps \
    -e "OGAD_RUN_ID=$RUN_ID" \
    -e OGAD_RUN_DIR=/app/runs/current \
    "${GPU_ENV[@]}" \
    -v "$RUN_DIR:/app/runs/current" \
    -v "$EXP_DIR:/app/experiments:ro" \
    -v "$RUN_DIR/data/train/graphons:/app/data/train/graphons" \
    -v "$RUN_DIR/data/predictions/graphons:/app/data/predictions/graphons" \
    -v "$RUN_DIR/data/models:/app/data/models" \
    "$service" "$@" 2>&1 | tee "$logfile"
}

cleanup() {
  "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
  if [[ "$OGAD_CLEANUP" == "1" ]]; then
    OGAD_ROOT="$ROOT" bash "$ROOT/scripts/cleanup_ogad_run.sh" || true
  fi
}

trap cleanup EXIT

run_stage prep_data "$RUN_DIR/sigl.log" "${SIGL_CMD[@]}"
run_stage train_model_b "$RUN_DIR/d3pm.log" "${D3PM_CMD[@]}"
run_stage eval_metrics "$RUN_DIR/ggme.log" "${GGME_CMD[@]}"

echo "Run complete: $RUN_DIR"
