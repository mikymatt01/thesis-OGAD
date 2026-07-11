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

Espansione ed esecuzione seriale di `experiments/test_merged.json`:

```bash
python3 scripts/run_from_merged.py
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
```

Per forzare il cleanup a fine singola run seriale:

```bash
OGAD_CLEANUP=1 ./scripts/run_ogad_experiment.sh experiments/test_nc0.json nc0
```

## Copia risultati dal server

```bash
scp -r thesis_unipi:/raid/m.mattiello/OGAD/runs ./
```
