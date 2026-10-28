#!/usr/bin/env bash
# Run SIGL → optional DiGress → D3PM → GGME with experiment JSON, logs, and run dir.
# Intended host: Linux with Docker Compose v2 and GNU coreutils (date, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP_REL="${1:-experiments/example.json}"
SLUG="${2:-}"
OGAD_CLEANUP="${OGAD_CLEANUP:-0}"
OGAD_GPU_ID="${OGAD_GPU_ID:-}"
RUNS_ROOT="${OGAD_RUNS_ROOT:-$ROOT/runs}"
OGAD_ONLY_DIGRESS="${OGAD_ONLY_DIGRESS:-0}"
OGAD_ONLY_BASELINES="${OGAD_ONLY_BASELINES:-0}"
OGAD_SKIP_DIGRESS="${OGAD_SKIP_DIGRESS:-0}"
OGAD_SKIP_BASELINES="${OGAD_SKIP_BASELINES:-0}"
OGAD_DIGRESS_HOST_DIR="${OGAD_DIGRESS_HOST_DIR:-}"
OGAD_BASELINES_HOST_DIR="${OGAD_BASELINES_HOST_DIR:-}"

EXP_ABS="$ROOT/$EXP_REL"
if [[ ! -f "$EXP_ABS" ]]; then
  echo "Experiment file not found: $EXP_ABS" >&2
  exit 1
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
if [[ -n "$SLUG" ]]; then
  RUN_ID="${RUN_ID}_${SLUG}"
fi

RUN_DIR="$RUNS_ROOT/$RUN_ID"
DIGRESS_HOST_DIR="${OGAD_DIGRESS_HOST_DIR:-$RUN_DIR/data/predictions/graphs/digress}"
BASELINES_HOST_DIR="${OGAD_BASELINES_HOST_DIR:-$RUN_DIR/data/predictions/graphs/classical}"
mkdir -p "$RUN_DIR"
mkdir -p \
  "$RUN_DIR/data/train/graphons" \
  "$RUN_DIR/data/predictions/graphons" \
  "$BASELINES_HOST_DIR" \
  "$RUN_DIR/data/models" \
  "$DIGRESS_HOST_DIR"

python3 "$ROOT/scripts/ogad_write_manifest.py" "$EXP_ABS" "$RUN_DIR" "$RUN_ID"

CONTAINER_CONFIG="/app/runs/current/experiment.json"
DIGRESS_ENABLED="$(python3 -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print("1" if (cfg.get("digress") or {}).get("enabled", False) else "0")' "$RUN_DIR/experiment.json")"
BASELINES_ENABLED="$(python3 -c 'import json, sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print("1" if (cfg.get("baselines") or {}).get("enabled", False) else "0")' "$RUN_DIR/experiment.json")"

COMPOSE_PROJECT_NAME="${OGAD_COMPOSE_PROJECT_NAME:-ogad_${RUN_ID}}"
COMPOSE=(docker compose -p "$COMPOSE_PROJECT_NAME" -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.ogad-only.yml")
GPU_ENV=()
if [[ -n "$OGAD_GPU_ID" ]]; then
  GPU_ENV=(-e "NVIDIA_VISIBLE_DEVICES=$OGAD_GPU_ID" -e "CUDA_VISIBLE_DEVICES=0")
fi

SIGL_CMD=(poetry run python Main.py
  --config "$CONTAINER_CONFIG"
)

D3PM_CMD=(bash -lc
  "poetry run python ogad_model_finetune.py --config $CONTAINER_CONFIG && poetry run python ogad_model_inference.py --config $CONTAINER_CONFIG"
)

DIGRESS_CMD=(python src/main.py
  +experiment=community_graphs
)

BASELINES_CMD=(bash -lc
  "cd /app && python -m src.baselines.classical_graphs --config $CONTAINER_CONFIG"
)

GGME_CMD=(poetry run python main.py
  --config "$CONTAINER_CONFIG"
)
if [[ "$DIGRESS_ENABLED" != "1" ]]; then
  GGME_CMD+=(--skip-digress)
fi

run_stage() {
  local service="$1"
  shift
  local logfile="$1"
  shift
  "${COMPOSE[@]}" run --rm --no-deps \
    -e "OGAD_RUN_ID=$RUN_ID" \
    -e OGAD_RUN_DIR=/app/runs/current \
    -e "OGAD_EXPERIMENT_JSON=$CONTAINER_CONFIG" \
    "${GPU_ENV[@]}" \
    -v "$RUN_DIR:/app/runs/current" \
    -v "$RUN_DIR/data/train/graphons:/app/data/train/graphons" \
    -v "$RUN_DIR/data/predictions/graphons:/app/data/predictions/graphons" \
    -v "$DIGRESS_HOST_DIR:/app/data/predictions/graphs/digress" \
    -v "$BASELINES_HOST_DIR:/app/data/predictions/graphs/classical" \
    -v "$RUN_DIR/data/models:/app/data/models" \
    -v "$ROOT/src/baselines:/app/src/baselines:ro" \
    "$service" "$@" 2>&1 | tee "$logfile"
}

cleanup() {
  "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
  if [[ "$OGAD_CLEANUP" == "1" ]]; then
    OGAD_ROOT="$ROOT" bash "$ROOT/scripts/cleanup_ogad_run.sh" || true
  fi
}

trap cleanup EXIT

if [[ "$OGAD_ONLY_DIGRESS" == "1" ]]; then
  if [[ "$DIGRESS_ENABLED" == "1" ]]; then
    run_stage train_model_a "$RUN_DIR/digress.log" "${DIGRESS_CMD[@]}"
  else
    echo "DiGress disabled by experiment config; no shared DiGress run needed."
  fi
  echo "Run complete: $RUN_DIR"
  exit 0
fi

if [[ "$OGAD_ONLY_BASELINES" == "1" ]]; then
  if [[ "$BASELINES_ENABLED" == "1" ]]; then
    run_stage eval_metrics "$RUN_DIR/baselines.log" "${BASELINES_CMD[@]}"
  else
    echo "Classical baselines disabled by experiment config; no baseline run needed."
  fi
  echo "Run complete: $RUN_DIR"
  exit 0
fi

run_stage prep_data "$RUN_DIR/sigl.log" "${SIGL_CMD[@]}"
if [[ "$DIGRESS_ENABLED" == "1" && "$OGAD_SKIP_DIGRESS" != "1" ]]; then
  run_stage train_model_a "$RUN_DIR/digress.log" "${DIGRESS_CMD[@]}"
elif [[ "$DIGRESS_ENABLED" == "1" ]]; then
  echo "Skipping DiGress stage; using shared samples from $DIGRESS_HOST_DIR"
fi
if [[ "$BASELINES_ENABLED" == "1" && "$OGAD_SKIP_BASELINES" != "1" ]]; then
  run_stage eval_metrics "$RUN_DIR/baselines.log" "${BASELINES_CMD[@]}"
elif [[ "$BASELINES_ENABLED" == "1" ]]; then
  echo "Skipping classical baselines stage; using samples from $BASELINES_HOST_DIR"
fi
run_stage train_model_b "$RUN_DIR/d3pm.log" "${D3PM_CMD[@]}"
run_stage eval_metrics "$RUN_DIR/ggme.log" "${GGME_CMD[@]}"

echo "Run complete: $RUN_DIR"
