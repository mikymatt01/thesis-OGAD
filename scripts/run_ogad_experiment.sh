#!/usr/bin/env bash
# Run SIGL → D3PM → GGME (OGAD-only compose) with experiment JSON, logs, and run dir.
# Intended host: Linux with Docker Compose v2 and GNU coreutils (date, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP_REL="${1:-experiments/example.json}"
SLUG="${2:-}"

EXP_ABS="$ROOT/$EXP_REL"
if [[ ! -f "$EXP_ABS" ]]; then
  echo "Experiment file not found: $EXP_ABS" >&2
  exit 1
fi

EXP_BASENAME="$(basename "$EXP_REL")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
if [[ -n "$SLUG" ]]; then
  RUN_ID="${RUN_ID}_${SLUG}"
fi

RUN_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

python3 "$ROOT/scripts/ogad_write_manifest.py" "$EXP_ABS" "$RUN_DIR" "$RUN_ID"

COMPOSE=(docker compose -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.ogad-only.yml")

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
  "${COMPOSE[@]}" run --rm \
    -e "OGAD_RUN_ID=$RUN_ID" \
    -e OGAD_RUN_DIR=/app/runs/current \
    -v "$RUN_DIR:/app/runs/current" \
    -v "$ROOT/experiments:/app/experiments:ro" \
    "$service" "$@" 2>&1 | tee "$logfile"
}

run_stage prep_data "$RUN_DIR/sigl.log" "${SIGL_CMD[@]}"
run_stage train_model_b "$RUN_DIR/d3pm.log" "${D3PM_CMD[@]}"
run_stage eval_metrics "$RUN_DIR/ggme.log" "${GGME_CMD[@]}"

echo "Run complete: $RUN_DIR"
