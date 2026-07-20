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
    "num_bins": 16,
    "quantization": "codebook",
    "resolution": 128,
    "batch_size": 32,
    "accumulate_grad_batches": 1,
    "final_samples": 100,
    "max_epochs": 600,
    "transition": "marginal",
    "extra_features": "all",
    "match_training_density": false
  }
}
```

Ogni campo è opzionale e sovrascrive il default di `configs/experiment/graphons.yaml`.
Un campo non riconosciuto ora fa fallire il run invece di essere ignorato in silenzio.

`quantization` sceglie come le classi discrete di DiGress mappano sulle probabilità del graphon:

- `codebook` (default): livelli stimati sulla distribuzione dei valori dello split di training (Lloyd-Max).
- `uniform`: livelli equispaziati in `[0,1]`, il comportamento originale.

Con `uniform` e `num_bins=10` la probabilità inter-comunità (~0.01) viene arrotondata
esattamente a 0, quindi ogni grafo generato si spezza in una componente connessa per
comunità. Il codebook stimato mantiene entrambi i modi con lo stesso numero di classi.
I livelli usati vengono salvati in `data/predictions/graphons/digress/codebook.json`
e ripetuti in `stats.json`.

Per le ablazioni basta cambiare un campo alla volta: `quantization`, `transition`
(`uniform`/`marginal`) ed `extra_features` (`all`/`cycles`/`eigenvalues`/`null`)
sono tutti sovrascrivibili da qui.

### Selezione del candidato

Come lo stage D3PM, DiGress ora emette più set di grafoni candidati e lascia che
GGME scelga il migliore sullo split di validazione, copiandolo in `tensors.pt`.
Serve la sezione `graph_selection` nel JSON dell'esperimento, altrimenti SIGL non
crea `ggme_validation_graphs.pickle` e la selezione resta inerte:

```json
{
  "graph_selection": {
    "validation_fraction": 0.2,
    "split_seed": 0,
    "selection_metric": "mean_rank"
  }
}
```

I candidati vengono dagli snapshot salvati ogni `candidate_checkpoint_every`
epoche: si prendono gli ultimi `candidate_checkpoints`, ciascuno campionato con
`candidate_seeds` seed. Non serve un nuovo training, solo campionamento: il costo
è `candidate_checkpoints × candidate_seeds × candidate_samples` grafoni a circa
11 s l'uno, quindi 15 × 1 × 24 sono ~65 minuti in più per run.

Il budget va su checkpoint distinti e non su seed: sul run `20260718_145730` tre
seed dello stesso checkpoint coprivano ~20% di MMD, mentre checkpoint diversi
coprivano un fattore 4. Per lo stesso motivo gli snapshot non sono i top-k di
`val/E_CE`: su quel run la metrica monitorata era anti-correlata con la qualità
dei grafi, e metteva al primo posto il checkpoint peggiore. Se non esiste nessuno
snapshot (`candidate_checkpoint_every: 0`) si ricade sui top-k del monitor.

`lr_schedule: "cosine"` (con `lr_min_ratio`, default 0.05) fa decadere il
learning rate lungo il run. Serve a ridurre proprio quell'oscillazione tra
checkpoint: `lambda_train` moltiplica la loss per 5, quindi il learning rate
effettivo è 0.001 e tenuto costante per 600 epoche non converge mai. Il default
resta `constant` per non toccare gli altri dataset.

Se GGME fallisce, i grafoni e i candidati sono già su disco: non serve
riaddestrare, basta rieseguire il solo stage di valutazione.

```bash
docker compose -f docker-compose.yml -f docker-compose.ogad-only.yml build eval_metrics

OGAD_GRAPHON_GENERATOR=digress ./scripts/rerun_ggme.sh \
  runs/parallel_<timestamp>/runs/<run_id>
```

Le righe `ggme` del tentativo fallito vengono tolte da `metrics.jsonl`, che viene
comunque salvato come `metrics.jsonl.before_rerun`.

Quando la selezione è attiva, `candidate_samples` sostituisce `final_samples`.
Con `candidate_checkpoints: 0` si torna al singolo campionamento di prima.
Gli artifact finiscono in `data/predictions/graphons/digress/candidates/` e il
manifest in `candidates.json`; GGME scrive una riga `ggme_validation_candidate`
per candidato in `metrics.jsonl`, utile per vedere la dispersione.

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

## D3PM

```
docker compose \
  -f docker-compose.yml \
  -f docker-compose.ogad-only.yml \
  build prep_data train_model_b eval_metrics
```

```
  OGAD_GRAPHON_GENERATOR=d3pm python3 scripts/run_from_merged_parallel.py \
  --merged experiments/10_sigl_h2_degree_corrected_d3pm_full.json \
  --gpus 0 \
  --max-retries 1
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
