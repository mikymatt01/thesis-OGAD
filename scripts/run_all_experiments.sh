#!/usr/bin/env bash
# Run run_ogad_experiment.sh once per experiments/*.json (this directory only, no subfolders).
# Intended host: Linux, bash >= 4.1 (uses mapfile), Docker Compose v2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

shopt -s nullglob
files=(experiments/*.json)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No JSON files in experiments/ (top level only)." >&2
  exit 1
fi

mapfile -t sorted_files < <(printf '%s\n' "${files[@]}" | sort)

for f in "${sorted_files[@]}"; do
  slug="$(basename "$f" .json)"
  echo "=== Running experiment: $f (slug=$slug) ===" >&2
  "$ROOT/scripts/run_ogad_experiment.sh" "$f" "$slug"
done

echo "All ${#sorted_files[@]} experiment(s) finished." >&2
