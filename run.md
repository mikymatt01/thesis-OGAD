# Esecuzione pipeline OGAD

Requisiti sulla macchina GPU: Linux, Docker Compose v2, driver NVIDIA, `python3`.

## Build immagini

```bash
docker compose -f docker-compose.yml -f docker-compose.ogad-only.yml build
```

## Esecuzione seriale

Singola configurazione:

```bash
./scripts/run_ogad_experiment.sh experiments/test_nc0.json nc0
```

Per usare DiGress graphon al posto di D3PM nello stage generatore:

```bash
OGAD_GRAPHON_GENERATOR=digress ./scripts/run_ogad_experiment.sh experiments/test_merged_digress.json digress
```

`OGAD_GRAPHON_GENERATOR` accetta `d3pm` o `digress`; il default resta `d3pm`.
Entrambi producono tensori graphon compatibili con GGME, con shape `[N,1,128,128]`.
Gli output sono separati:

```text
D3PM:    data/predictions/graphons/tensors.pt
DiGress: data/predictions/graphons/digress/tensors.pt
```

Con DiGress viene salvata anche la diagnostica density-matching:

```text
data/predictions/graphons/digress/stats.json
```

Per DiGress si può aggiungere al JSON esperimento una sezione opzionale:

```json
{
  "digress_graphons": {
    "num_bins": 10,
    "resolution": 128,
    "batch_size": 2,
    "accumulate_grad_batches": 16,
    "final_samples": 100,
    "max_epochs": 200,
    "match_training_density": true
  }
}
```

Espansione ed esecuzione seriale di `experiments/test_merged.json`:

```bash
python3 scripts/run_from_merged.py
```

Esecuzione parallela DiGress usando il merged dedicato:

```bash
OGAD_GRAPHON_GENERATOR=digress python3 scripts/run_from_merged_parallel.py \
  --merged experiments/test_merged_digress.json \
  --gpus 0,1 \
  --max-retries 1
```

## Esecuzione parallela su più GPU

Dry-run, senza lanciare Docker:

```bash
python3 scripts/run_from_merged_parallel.py --dry-run --gpus 0,1
```

Esecuzione reale su due GPU, con un retry massimo per run fallita:

```bash
python3 scripts/run_from_merged_parallel.py --gpus 0,1 --max-retries 1
```

Gli output vengono isolati in:

```text
runs/parallel_<timestamp>/
├── experiments/
├── queue/
└── runs/
```

Ogni run usa artifact separati per `graphons`, `models` e `predictions`, quindi due run parallele non dovrebbero sovrascriversi.
Ogni run usa anche un project name Docker Compose dedicato (`ogad_<run_id>`), così i network Docker non collidono tra run seriali/parallele.
La configurazione attiva viene copiata nella run dir e letta nel container da `/app/runs/current/experiment.json`, evitando collisioni con il mount Compose di `/app/experiments`.
Il compose pinna la GPU fisica con `OGAD_GPU_ID` (`device_ids`), mentre dentro al container PyTorch vede quella singola GPU come `cuda:0`.

Se su una vecchia checkout compare:

```text
network ogad_default is ambiguous
```

ci sono network Docker residui con lo stesso nome. Con questa versione le nuove run non usano più il project name Compose globale `ogad`.

## tmux

Avvio:

```bash
tmux
```

Detach:

```text
Ctrl+B, poi D
```

Riattacco:

```bash
tmux a
```

Lista/kill sessioni:

```bash
tmux ls
tmux kill-session -t <id>
```

## Cleanup

Il cleanup globale è disabilitato di default nelle run isolate.

Pulizia completa di container, volumi, immagini e orfani Docker:

```bash
docker compose down -v --rmi all --remove-orphans
```

Se serve pulire manualmente gli artifact globali vecchi:

```bash
bash scripts/cleanup_ogad_run.sh
```

Pulizia manuale delle directory globali con Docker:

```bash
docker run --rm -v "$(pwd):/app" alpine rm -rf /app/data/predictions/graphons
docker run --rm -v "$(pwd):/app" alpine rm -f /app/data/predictions/graphs/digress/generated_samples1.txt
docker run --rm -v "$(pwd):/app" alpine rm -f /app/data/predictions/graphs/digress/generated_samples2.txt
docker run --rm -v "$(pwd):/app" alpine rm -rf /app/data/train/graphons
docker run --rm -v "$(pwd):/app" alpine rm -rf /app/runs

bash scripts/cleanup_ogad_parallel_runs.sh runs/parallel_20260711_201704

# one-liner equivalente (senza script):
# P=runs/parallel_20260711_201704; for d in "$P"/runs/*; do docker run --rm -v "$(pwd):/app" alpine sh -c "rm -rf /app/$d/data && rm -f /app/$d/model_train_graphs.pickle /app/$d/ggme_validation_graphs.pickle"; done
```

Per forzare il cleanup a fine singola run seriale:

```bash
OGAD_CLEANUP=1 ./scripts/run_ogad_experiment.sh experiments/test_nc0.json nc0
```

## Copia risultati dal server

```bash
scp -r thesis_unipi:/raid/m.mattiello/OGAD/runs ./
```

## DiGress

```
docker compose \
  -f docker-compose.yml \
  -f docker-compose.ogad-only.yml \
  -f docker-compose.digress-graphons.yml \
  build train_model_b
  
OGAD_GRAPHON_GENERATOR=digress python3 scripts/run_from_merged_parallel.py \
  --merged experiments/test_merged_digress.json \
  --gpus 0,2 \
  --max-retries 1
```
