#!/usr/bin/env bash
# Re-run only the GGME evaluation against an existing run directory.
#
# The generator stages are the expensive ones; when GGME fails the graphons and
# candidates are already on disk, so there is no reason to train again.
#
#   ./scripts/rerun_ggme.sh runs/parallel_20260718_145730/runs/20260718_145730_nc0
#
# Rebuild eval_metrics first if src/ggme changed:
#   docker compose -f docker-compose.yml -f docker-compose.ogad-only.yml build eval_metrics
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR_REL="${1:-}"
if [[ -z "$RUN_DIR_REL" ]]; then
  echo "Usage: rerun_ggme.sh RUN_DIR" >&2
  exit 2
fi
RUN_DIR="$(cd "$RUN_DIR_REL" && pwd)"

if [[ ! -f "$RUN_DIR/experiment.json" ]]; then
  echo "Not a run directory (no experiment.json): $RUN_DIR" >&2
  exit 1
fi

RUN_ID="$(basename "$RUN_DIR")"
GRAPHON_GENERATOR="${OGAD_GRAPHON_GENERATOR:-d3pm}"
OGAD_GPU_ID="${OGAD_GPU_ID:-}"
CONTAINER_CONFIG="/app/runs/current/experiment.json"

COMPOSE_PROJECT_NAME="${OGAD_COMPOSE_PROJECT_NAME:-ogad_rerun_${RUN_ID}}"
COMPOSE=(docker compose -p "$COMPOSE_PROJECT_NAME"
  -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.ogad-only.yml")
if [[ "$GRAPHON_GENERATOR" == "digress" ]]; then
  COMPOSE+=(-f "$ROOT/docker-compose.digress-graphons.yml")
fi

GGME_CMD=(poetry run python main.py --config "$CONTAINER_CONFIG" --skip-digress)
if [[ "$GRAPHON_GENERATOR" == "digress" ]]; then
  GGME_CMD+=(--graphons_pt /app/data/predictions/graphons/digress/tensors.pt)
  GGME_CMD+=(--candidate_manifest /app/data/predictions/graphons/digress/candidates.json)
fi

GPU_ENV=()
if [[ -n "$OGAD_GPU_ID" ]]; then
  GPU_ENV=(-e "NVIDIA_VISIBLE_DEVICES=$OGAD_GPU_ID" -e "CUDA_VISIBLE_DEVICES=0")
fi

# GGME appends to metrics.jsonl, so keep the failed attempt's rows out of the way.
if [[ -f "$RUN_DIR/metrics.jsonl" ]]; then
  cp "$RUN_DIR/metrics.jsonl" "$RUN_DIR/metrics.jsonl.before_rerun"
  grep -v '"stage": "ggme' "$RUN_DIR/metrics.jsonl" > "$RUN_DIR/metrics.jsonl.tmp" || true
  mv "$RUN_DIR/metrics.jsonl.tmp" "$RUN_DIR/metrics.jsonl"
  echo "Previous metrics.jsonl kept as metrics.jsonl.before_rerun"
fi

"${COMPOSE[@]}" run --rm --no-deps \
  -e "OGAD_RUN_ID=$RUN_ID" \
  -e OGAD_RUN_DIR=/app/runs/current \
  -e "OGAD_GRAPHON_GENERATOR=$GRAPHON_GENERATOR" \
  "${GPU_ENV[@]}" \
  -v "$RUN_DIR:/app/runs/current" \
  -v "$RUN_DIR/data/train/graphons:/app/data/train/graphons" \
  -v "$RUN_DIR/data/predictions/graphons:/app/data/predictions/graphons" \
  -v "$RUN_DIR/data/models:/app/data/models" \
  eval_metrics "${GGME_CMD[@]}" 2>&1 | tee "$RUN_DIR/ggme_rerun.log"

"${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
