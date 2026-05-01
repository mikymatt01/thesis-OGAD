#!/usr/bin/env bash
# Cleanup artifacts that can leak across experiments (generated samples, cached datasets, etc.).
set -euo pipefail

ROOT="${OGAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

rm_in_container() {
  # Run rm inside a tiny container to avoid host OS rm differences and permission issues.
  docker run --rm -v "$ROOT:/repo" alpine rm -rf "$@"
}

# Predictions / generations
rm_in_container /repo/data/predictions/graphons
rm_in_container /repo/data/predictions/graphs/digress/generated_samples1.txt
rm_in_container /repo/data/predictions/graphs/digress/generated_samples2.txt

# Training data caches that you want regenerated per experiment
rm_in_container /repo/data/train/graphons

echo "Cleanup done."
