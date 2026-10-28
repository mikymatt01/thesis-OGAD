# Log delle run esportati per il repo

Sottoinsieme versionato di `runs/`, prodotto da `scripts/export_run_logs.py`.
`runs/` resta locale e ignorato da git: qui ci sono solo i log e le metriche
dei batch che girano sui **dataset baseline generati da noi**
(`datasets/README.md`). I batch su soli CommunityGraphs sono esclusi.

## Batch inclusi

| batch | dataset | file | dimensione |
|---|---|---:|---:|
| `parallel_20260728_155720` | `ba_n100_m3`, `er_n100_p005`, `expdeg_n100_lin`, `sbm_n100_4c`, `ws_n100_k8_p01` | 153 | 9.28 MB |
| `parallel_20260729_144516` | `ba_n100_m3`, `er_n100_p005`, `expdeg_n100_lin`, `sbm_n100_4c`, `ws_n100_k8_p01` | 326 | 36.09 MB |
| `parallel_20260803_063318` | `ba_n100_m3`, `er_n100_p005`, `ws_n100_k8_p01` | 71 | 9.67 MB |
| `parallel_20260803_154001` | `ba_n100_m3`, `ws_n100_k8_p01` | 83 | 12.53 MB |

## Cosa e' incluso

- `*.log` con i frame intermedi delle progress bar tqdm rimossi (l'ultimo
  frame di ogni barra resta, per conservare iterazioni e tempi);
- `*.json` / `*.jsonl` / `*.yaml` di metriche e configurazione
  (`metrics.jsonl`, `manifest.json`, `experiment.json`, `status.json`, hydra);
- i report aggregati CSV/MD al livello di `runs/`:
  `ggme_completed_dataset_filter_status.csv`, `ggme_completed_datasets_best_ogad.csv`, `ggme_completed_datasets_best_ogad.md`, `ogad_best_runs_model_comparison.csv`, `ogad_completed_run_ranking.csv`, `ogad_incomplete_runs_parallel_20260729_20260803.csv`, `ogad_report_parallel_20260729_20260803.md`.

## Cosa e' escluso (resta solo in `runs/`)

- pickle, PNG, GIF, `.npz`, `generated_samples*.txt`: artefatti binari pesanti;
- `sigl_workers/job_*.log`: decine di migliaia di file da ~600 byte;
- directory `wandb/`: logging interno dell'SDK;
- `data/train/graphons/bootstrap_metadata.json`: input rigenerabile da decine di MB.

Per rigenerare questa directory: `python scripts/export_run_logs.py`.
