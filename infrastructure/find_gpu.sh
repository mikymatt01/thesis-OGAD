#!/usr/bin/env bash
set -u

# Try creating a VM with:
# - Machine: N1 "N8" (`n1-standard-8`)
# - GPU: NVIDIA T4 (`nvidia-tesla-t4`) x1
#
# Zone list strategy:
# - Prefer dynamically enumerating zones where `nvidia-tesla-t4` exists (fast + accurate).
# - Fall back to a curated list from Google's GPU locations doc if gcloud isn't usable.
#
# Doc: https://cloud.google.com/compute/docs/gpus/gpu-regions-zones

machine_type="${MACHINE_TYPE:-n1-standard-8}"
gpu_type="${GPU_TYPE:-nvidia-tesla-t4}"
gpu_count="${GPU_COUNT:-1}"

zones=()
project="${GOOGLE_CLOUD_PROJECT:-${GCLOUD_PROJECT:-}}"

if zones_json="$(gcloud compute accelerator-types list \
  --filter="name=${gpu_type}" \
  --format="json(name,zone)" 2>/dev/null)"; then
  zone_lines="$(echo "$zones_json" | python3 -c 'import json,sys; rows=json.load(sys.stdin); zones=sorted({r.get("zone","").split("/")[-1] for r in rows if r.get("zone")}); print("\n".join(zones))' 2>/dev/null || true)"
  if [[ -n "$zone_lines" ]]; then
    while IFS= read -r z; do
      [[ -n "$z" ]] && zones+=("$z")
    done <<<"$zone_lines"
  fi
fi

if [[ "${#zones[@]}" -eq 0 ]]; then
  echo "WARN: couldn't enumerate T4 zones via gcloud; using a curated fallback zone list."
  zones=(
    us-central1-a us-central1-b us-central1-c us-central1-f
    us-east1-b us-east1-c us-east1-d
    us-east4-a us-east4-b us-east4-c
    us-west1-a us-west1-b
    us-west2-b us-west2-c
    us-west3-b
    us-west4-a us-west4-b us-west4-c
    us-east5-a
    europe-west1-b europe-west1-c europe-west1-d
    europe-west2-a europe-west2-b europe-west2-c
    europe-west3-b
    europe-west4-a europe-west4-b europe-west4-c
    europe-central2-b europe-central2-c
    asia-east1-a asia-east1-c
    asia-east2-a asia-east2-c
    asia-northeast1-a asia-northeast1-b asia-northeast1-c
    asia-northeast3-b asia-northeast3-c
    asia-south1-a asia-south1-b
    asia-southeast1-a asia-southeast1-b asia-southeast1-c
    australia-southeast1-a australia-southeast1-c
    me-west1-b
    southamerica-east1-a southamerica-east1-c
  )
fi

echo "Machine type: $machine_type"
echo "GPU:          ${gpu_type} x${gpu_count}"
echo "Zones to try: ${#zones[@]}"
echo

for zone in "${zones[@]}"; do
  echo "=============================="
  echo "Trying zone: $zone"
  echo "=============================="

  # Optional fast-fail (requires listing accelerator types).
  if [[ -n "$project" ]]; then
    if ! gcloud compute accelerator-types list \
      --project="$project" \
      --filter="zone:(${zone}) AND name=${gpu_type}" \
      --limit=1 \
      --format="value(name)" \
      >/dev/null 2>&1; then
      echo "Skipping $zone (${gpu_type} not advertised in this zone, per gcloud)."
      continue
    fi
  fi

  tf_log="$(mktemp -t terraform-apply.XXXXXX)"
  terraform apply -auto-approve \
    -var="zone=${zone}" \
    -var="machine_type=${machine_type}" \
    -var="gpu_type=${gpu_type}" \
    -var="gpu_count=${gpu_count}" \
    >"$tf_log" 2>&1
  ec=$?

  if [[ "$ec" -eq 0 ]]; then
    rm -f "$tf_log"
    echo "SUCCESS in $zone"
    exit 0
  fi

  echo "Failed in $zone"
  tail -n 40 "$tf_log"

  if grep -qiE "GPUS_ALL_REGIONS|gpus_all_regions|GPUS-ALL-REGIONS-per-project" "$tf_log"; then
    echo
    echo "Stopping: project global GPU quota (GPUS_ALL_REGIONS) is exhausted."
    echo "Retrying zones won't help until you delete other GPU VMs/reservations or request a quota increase."
    rm -f "$tf_log"
    exit 2
  fi

  rm -f "$tf_log"
  sleep 5
done

echo "No zone worked."
exit 1
