### Delete images
- docker compose down -v --rmi all --remove-orphans

### How to execute
- tmux
- docker compose up
- control + B and D

### OGAD-only pipeline (SIGL → D3PM → GGME, no DiGress)

Gli script in `scripts/` sono pensati per **Linux**: bash recente (>= 4.1 per `run_all_experiments.sh`), Docker Compose plugin v2, `python3` sul PATH.

```bash
docker compose -f docker-compose.yml -f docker-compose.ogad-only.yml build
```

End-to-end run with unified experiment JSON, local logs under `runs/<run_id>/`, and DiGress skipped in GGME:

```bash
./scripts/run_ogad_experiment.sh experiments/example.json
```

Optional second argument sets a slug in the run id (e.g. `mytrial`).

Run every top-level `experiments/*.json` in sorted order (one full pipeline per file; stops on first error):

```bash
t
```

### Delete tmux session
- tmux ls
- tmux kill-session -t <id>

### Delete data dir
docker run --rm -v $(pwd):/app alpine rm -rf /app/data/predictions/graphons
docker run --rm -v $(pwd):/app alpine rm /app/data/predictions/graphs/digress/generated_samples1.txt 
cd data/train/graphons && docker run --rm -v $(pwd):/app alpine rm -rf /app

### Cleanup automatico a fine esperimento
Lo script `scripts/run_ogad_experiment.sh` esegue automaticamente la pulizia (generazioni + cache dataset) a fine run.

Per disabilitarla temporaneamente:

```bash
OGAD_CLEANUP=0 ./scripts/run_ogad_experiment.sh experiments/example.json
```

### Copy results from server
```bash
scp -r thesis_unipi:/raid/m.mattiello/OGAD/runs ./
```