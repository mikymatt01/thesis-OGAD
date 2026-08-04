#!/usr/bin/env bash
#
# Rigenera in modo deterministico l'INTERA suite di dataset di validazione della
# tesi dentro datasets/. La directory datasets/ e' in .gitignore: questo script
# e' il deliverable riproducibile (il "manifest" versionato dei dati).
#
# La cartella datasets/ e' tipicamente di proprieta' di root (generata in Docker),
# quindi va eseguito come root, preferibilmente dentro il container prep-data:
#
#   docker run --rm -v "$PWD":/app -w /app \
#     -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
#     mikymatt/ogad-prep-data bash scripts/build_validation_datasets.sh
#
# VINCOLO: questo script tocca ESCLUSIVAMENTE il contenuto di datasets/
# (cancellazione degli elementi esistenti + generazione dei nuovi).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-datasets}"
GEN="python generate_community_graphs.py"
BASELINE_GEN="python -m src.baselines.classical_graphs"

# Iperparametri comuni della suite (seed fissi -> riproducibilita').
INSTANCES=300
RATIO=0.8
SEED=42

echo ">> Pulizia: rimuovo tutto il contenuto attuale di $DATA_ROOT/ (solo validazione d'ora in poi)"
mkdir -p "$DATA_ROOT"
rm -rf "${DATA_ROOT:?}"/*

gen() {  # gen <nome> <famiglia> [args extra...]
  local name="$1"; local family="$2"; shift 2
  local out="$DATA_ROOT/$name"
  echo ">> genero $name ($family)"
  $GEN --family "$family" --instances "$INSTANCES" --train-ratio "$RATIO" \
       --seed "$SEED" --output-dir "$out" --overwrite "$@"
  echo ">> genero baseline classiche per $name"
  $BASELINE_GEN --model all \
       --train-graphs "$out/CommunityGraphs_train.pkl" \
       --test-graphs "$out/CommunityGraphs_test.pkl" \
       --output-dir "$out/baselines" \
       --num-graphs 100 \
       --seed "$SEED"
}

# Suite ridotta a 5 dataset: copre i 3 tipi di graphon (costante, a blocchi,
# liscio) + 2 casi che stressano i descrittori degree e clustering.
gen er_n100_p005    erdos-renyi            --nodes 100 --params p=0.05                        # graphon costante (ground-truth)
gen sbm_n100_4c     stochastic-block-model --nodes 100 --params sizes='[25,25,25,25]' \
    probabilities='[[0.10,0.01,0.01,0.01],[0.01,0.10,0.01,0.01],[0.01,0.01,0.10,0.01],[0.01,0.01,0.01,0.10]]'  # graphon a blocchi
gen expdeg_n100_lin expected-degree        --nodes 100 --params min_weight=1 max_weight=20   # graphon liscio (rank-1)
gen ba_n100_m3      barabasi-albert        --nodes 100 --params m=3                           # scale-free (stress degree)
gen ws_n100_k8_p01  watts-strogatz         --nodes 100 --params k=8 p=0.1                     # small-world (clustering)

# --- Manifest leggibile accanto ai dati ---------------------------------------
cat > "$DATA_ROOT/README.md" <<'MD'
# Suite di validazione (tesi) — generata da scripts/build_validation_datasets.sh

5 dataset: ~300 grafi, split 80/20, seed 42, n=100, formato CommunityGraphs
(list[csr_matrix]) leggibile da SIGL via community_graphs.sigl_loader.

Ogni dataset include anche `baselines/{erdos_renyi,configuration_model,sbm,degree_corrected_sbm}.pickle`.

| dataset | struttura | tipo graphon / scopo | metrica esercitata |
|---|---|---|---|
| er_n100_p005    | Erdos-Renyi          | graphon costante (ground-truth)   | degree |
| sbm_n100_4c     | 4 comunita'          | graphon a blocchi (ground-truth)  | spettro / comunita' |
| expdeg_n100_lin | expected-degree      | graphon liscio rank-1 (non a blocchi) | struttura continua |
| ba_n100_m3      | Barabasi-Albert      | scale-free (caso di stress)       | degree heavy-tail |
| ws_n100_k8_p01  | Watts-Strogatz       | small-world                       | clustering |
MD

# --- Permessi: rendi i dati di proprieta' dell'utente host, se noto -----------
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
  chown -R "$HOST_UID:$HOST_GID" "$DATA_ROOT"
fi
chmod -R a+rX "$DATA_ROOT" || true

echo ">> Fatto. Contenuto di $DATA_ROOT/:"
ls -1 "$DATA_ROOT"
