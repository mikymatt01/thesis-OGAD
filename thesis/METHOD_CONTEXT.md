# METHOD_CONTEXT: execution evidence and thesis-run parameters

This note reconstructs the method and experiment parameters from the exported execution evidence in:

- `runs_export/parallel_20260728_155720`
- `runs_export/parallel_20260729_144516`
- `runs_export/parallel_20260803_063318`
- `runs_export/parallel_20260803_154001`

The user-facing prompt referred to `experiments/parallel_*`; in this repository the inspected artifacts are under `runs_export/parallel_*`. Each of the four listed export directories was inspected.

Evidence priority used here:

1. Successful execution logs, queue status, `manifest.json` runtime fields, `metrics.jsonl`.
2. Exported experiment JSON files under each `runs_export/parallel_*/experiments`.
3. Source behavior in `scripts/`, `src/`, and `community_graphs/`.
4. Code defaults.
5. Existing reports in `runs_export/*.md` and `runs_export/*.csv`.

## Executive Summary

The thesis executions were not one homogeneous run. There are at least four execution groups with different SIGL cluster sweeps and D3PM learning-rate grids:

| Group | Datasets in exported configs | Queue status | Main resolved settings | Interpretation |
|---|---|---|---|---|
| `parallel_20260728_155720` | ER, SBM, expdeg, BA, WS | 4 failed, 1 running | `n_clusters=1`; D3PM `lr=[5e-5]`, `num_classes=[14]`; all baseline stages enabled | Partial/failed exploratory run. Per-run D3PM completed for 8 manifests, but GGME usually crashed on NaNs after only 2 descriptors. Do not treat as final. |
| `parallel_20260729_144516` | ER, SBM, expdeg, BA, WS | 22 complete with full GGME rows, 8 incomplete/running/no metrics | `n_clusters=[1,1,4,4,8,8]` per dataset; D3PM `lr=[2e-5,3e-5,5e-5]`, `num_classes=[10,14]`, `hybrid_loss_coeff=[0.006,0.01]` | Main broad thesis sweep. Existing best-run reports select final candidates from this group for all five datasets, but some jobs were retried or incomplete. |
| `parallel_20260803_063318` | ER, BA, WS | queue statuses `running`; no full D3PM/GGME per-run completion | `n_clusters` includes 8/16/32; D3PM `lr=[1e-5,2e-5,3e-5]`, `num_classes=[10,14]` | Partial follow-up sweep. Shared DiGress/baselines completed, per-run OGAD did not complete in export. |
| `parallel_20260803_154001` | BA, WS | 6 complete | BA/WS `n_clusters=[4,8,16]`; D3PM `lr=[1e-5,2e-5,3e-5]`, `num_classes=[10,14]` | Successful follow-up sweep for BA and WS only. It supersedes some incomplete BA/WS attempts, but repository reports do not explicitly declare it the final thesis selection. |

The most defensible final-run statement is:

- Fully successful ranked OGAD runs selected by `runs_export/ogad_report_parallel_20260729_20260803.md`: BA `20260802_011527_ba_n100_m3_nc19`, ER `20260730_045033_er_n100_p005_nc5`, expdeg `20260801_171551_expdeg_n100_lin_nc17`, SBM `20260731_055853_sbm_n100_4c_nc10`, WS `20260803_064123_ws_n100_k8_p01_nc24`.
- Stricter completed-dataset report `runs_export/ggme_completed_datasets_best_ogad.md` includes only expdeg and SBM, because BA/ER/WS have incomplete queued jobs across selected batches.
- BA and WS have later successful follow-up runs in `parallel_20260803_154001`; final selection among those and the 20260729 report is not explicitly resolved by repository evidence.

## Source Defaults vs Config vs Runtime

| Parameter | Code default | Experiment config value | Runtime value | Source |
|---|---:|---:|---:|---|
| SIGL `Res` | 1000 | 128 | 128 | `src/sigl/Main.py:147`; per-run `experiment.json`; SIGL `Parameters: Namespace(... Res=128 ...)` |
| SIGL `n_epoch` | 100 | 100 | 100 | `src/sigl/Main.py:148`; per-run config/log |
| SIGL `lr` | 0.06 | 0.06 | 0.06 | `src/sigl/Main.py:154`; per-run config/log |
| SIGL hidden GNN dims | `8,8,8` | omitted | `8,8,8` | SIGL log `Parameters` |
| SIGL INR hidden dims | `20,20` | omitted | `20,20` | SIGL log `Parameters` |
| SIGL INR batch size | 1024 | omitted | 1024 | `src/sigl/Main.py:152`; SIGL log |
| SIGL `n_clusters` | `None` | group-dependent | group-dependent; runtime graphon count can be lower than `n_clusters*bootstrap_replicates` when clusters are empty | `src/sigl/Main.py:160`; SIGL logs |
| SIGL `bootstrap_replicates` | 1 | 500 | 500 | `src/sigl/Main.py:161`; SIGL logs |
| SIGL `bootstrap_seed` | 0 | 0 | 0 | `src/sigl/Main.py:162`; SIGL logs |
| SIGL workers | 1 | 4 | 4 | `src/sigl/Main.py:163`; SIGL logs |
| Include original clusters | false | false | false | `src/sigl/Main.py:164-168`; SIGL logs |
| D3PM dim | `[128,256]` | `[512]` | 512 in all selected checkpoints | `src/d3pm/training_utils.py:17-25`; manifests |
| D3PM image size | `[128]` | `[128]` from SIGL `Res` | 128 | `scripts/experiment_grid.py`; manifests |
| D3PM batch size | `[16,32]` | `[32]` | 32 | `src/d3pm/training_utils.py:17-25`; D3PM logs |
| D3PM LR grid | `[1e-5,2e-5,5e-5]` | 20260728 `[5e-5]`; 20260729 `[2e-5,3e-5,5e-5]`; 20260803 groups `[1e-5,2e-5,3e-5]` | selected per run | configs/manifests |
| D3PM classes | `[8]` | 20260728 `[14]`; later `[10,14]` | selected per run; GGME-selected 20260729 final checkpoints are mostly 14, with WS selected 10 | configs/manifests |
| D3PM quantization rule | source rule only | not separately config-controlled; depends on `num_classes=K` | `q = clamp(round((K-1)W), 0, K-1)` at D3PM batch preparation time | `src/d3pm/ogad_model_finetune.py:73-80` |
| D3PM dequantization rule | source rule only | not separately config-controlled; depends on selected `num_classes=K` | generated class `k -> k/(K-1)`, followed by symmetrize, clamp, diagonal zero | `src/d3pm/ogad_model_finetune.py:246-273` |
| D3PM forward transition | uniform categorical transition with cosine beta schedule | not config-controlled | same for all selected D3PM runs | `src/d3pm/d3pm_runner.py:159-211` |
| D3PM reverse target/loss | predicts `x_0` logits; objective `KL + lambda * CE` | lambda from `hybrid_loss_coeff` grid | same form for all D3PM candidates | `src/d3pm/d3pm_runner.py:291-326`; `src/d3pm/training_utils.py:49-51` |
| D3PM hybrid loss coeff | `[0.01,0,0.001]` | 20260728 and 20260803 groups `[0.006]`; 20260729 `[0.006,0.01]` | selected per run; GGME-selected 20260729 finals often use 0.01 | `src/d3pm/training_utils.py:17-25`; manifests |
| D3PM diffusion steps | hard-coded 1000 | not config-controlled | 1000 | `src/d3pm/ogad_model_finetune.py:159-164` and `:253-258` |
| D3PM optimizer | AdamW in code | not config-controlled | AdamW | `src/d3pm/ogad_model_finetune.py` |
| D3PM max epochs | 200 | 300 | up to 300 with early stopping | `src/d3pm/training_utils.py:27-33`; manifests |
| D3PM patience | 10 | 25 | 25 | `src/d3pm/training_utils.py:27-33`; manifests |
| D3PM min delta | 0.0001 | 0.00005 | 0.00005 | `src/d3pm/training_utils.py:27-33`; manifests |
| D3PM split | 20% validation, seed 0 | 20%, seed 0 | deterministic 80/20 split of generated graphon images | `src/d3pm/training_utils.py:54-68`; manifests |
| D3PM checkpoint behavior | best validation loss | best validation loss plus optional GGME selection | training first selects best validation-loss checkpoint; GGME validation can replace final `best_model`/`best_params`/`tensors.pt` by selected candidate | `src/d3pm/ogad_model_finetune.py:216-227`; `src/ggme/main.py` |
| Generated graphon tensors per candidate | 16 | not config-controlled | 16 tensors per candidate | `src/d3pm/ogad_model_finetune.py:261-266` |
| GGME generated graph count | 100 | 100 | 100 OGAD and 100 DiGress graphs in successful GGME logs | `src/ggme/main.py:404`; logs |
| GGME seed | 0 | 0 | 0 | configs/logs |
| GGME descriptors | code-defined | not config-controlled | degree distribution, clustering coefficient, normalized Laplacian spectrum | `src/ggme/main.py`; `src/ggme/metrics/descriptor_functions.py` |
| GGME kernels | Gaussian, linear | not config-controlled | per descriptor selected from Gaussian/linear and sigmas `[0.1,1,10]` | `src/ggme/main.py:110-118`; GGME logs |
| GGME histogram bins/ranges | degree: variable histogram; clustering: 200 bins `[0,1]`; spectrum: 200 bins `[0,2]` | not config-controlled | same | `src/ggme/metrics/descriptor_functions.py:15-45` |
| MMD | exact quadratic MMD, density descriptors | not config-controlled | exact MMD, `density=True`, no linear approximation | `src/ggme/main.py:135-142` |
| Classical baselines | CLI default all models, seed fallback 0 | ER, configuration, SBM, DC-SBM; seed 42 | shared per dataset, 100 graphs each, seed 42 | `src/baselines/classical_graphs/cli.py`; baseline logs/status |
| DiGress enabled | controlled by config | true in all inspected configs | shared per dataset where possible; mounted into per-run GGME | `scripts/run_ogad_experiment.sh`; shared status/logs |

## Cross-Run Comparison

| Parameter / behavior | `parallel_20260728_155720` | `parallel_20260729_144516` | `parallel_20260803_063318` | `parallel_20260803_154001` | Interpretation |
|---|---|---|---|---|---|
| Datasets executed/configured | ER, SBM, expdeg, BA, WS | ER, SBM, expdeg, BA, WS | ER, BA, WS | BA, WS | Not identical; later follow-ups narrow to problematic datasets. |
| Per-run configs | 5 | 30 | 9 configs, 4 queued/runs exported | 6 | 20260729 is the broad sweep. |
| SIGL resolution | 128 | 128 | 128 | 128 | Fixed across exports; differs from SIGL default 1000. |
| SIGL epochs/LR | 100, 0.06 | 100, 0.06 | 100, 0.06 | 100, 0.06 | Fixed. |
| SIGL bootstrap | 500, seed 0 | 500, seed 0 | 500, seed 0 | 500, seed 0 | Fixed, but actual count depends on cluster output. |
| SIGL workers | 4 | 4 | 4 | 4 | Fixed. |
| `n_clusters` | 1 | per dataset `[1,1,4,4,8,8]` | ER 8/16/32; BA 8/16/32; WS 8/16/32 configs; only ER 8/16/32 and BA 8 queue/runs present | BA/WS `[4,8,16]` | Core experiment variable. |
| D3PM LR grid | `[5e-5]` | `[2e-5,3e-5,5e-5]` | `[1e-5,2e-5,3e-5]` | `[1e-5,2e-5,3e-5]` | Later sweep lowers top LR. |
| D3PM num classes | `[14]` | `[10,14]` | `[10,14]` | `[10,14]` | 20260728 is narrower. |
| D3PM candidates/run | 1 | 12 | not completed | 6 | Derived from LR/classes/hybrid-loss grid with fixed batch/dim. |
| D3PM dim | 512 | 512 | 512 | 512 | Fixed experiment override. |
| D3PM batch | 32 | 32 | 32 | 32 | Fixed experiment override. |
| D3PM max epochs | 300 | 300 | 300 | 300 | Fixed experiment override. |
| D3PM early stopping | patience 25, min_delta 5e-5 | same | same if reached | same | Runtime best epochs vary. |
| Generated graphons/candidate | 16 tensors | 16 tensors | unknown for incomplete runs | 16 tensors | Hard-coded. |
| Generated graphs for GGME | 100 | 100 | no successful per-run GGME | 100 | Fixed in config/logs. |
| Random seeds | SIGL bootstrap 0; D3PM split 0; GGME 0; baselines 42 | same | same | same | Fixed across inspected configs. |
| Shared DiGress | complete for 5 datasets | complete for 5 datasets | complete for ER/BA/WS | complete for BA/WS | Shared stages often succeed even when OGAD per-run stages do not. |
| Shared classical baselines | complete for 5 datasets | complete for 5 datasets | complete for ER/BA/WS | complete for BA/WS | Same four model families. |
| Final status | failed/partial | mixed; 22 full GGME rows | partial/running | complete for all 6 | Final tables should prefer 20260729 successful rows and 20260803_154001 follow-ups. |

## Execution Group Details

### `parallel_20260728_155720`

- Config files: `runs_export/parallel_20260728_155720/experiments/*.json`.
- Datasets: `er_n100_p005`, `sbm_n100_4c`, `expdeg_n100_lin`, `ba_n100_m3`, `ws_n100_k8_p01`.
- Enabled stages: SIGL, DiGress, classical baselines, D3PM, GGME. DiGress and baselines were run as shared stages; per-run logs show SIGL/D3PM/GGME.
- Resolved SIGL config: `Res=128`, `n_epoch=100`, `lr=0.06`, `n_clusters=1`, `bootstrap_replicates=500`, `bootstrap_seed=0`, `sigl_workers=4`, `include_original_clusters=false`.
- Resolved D3PM config: `batch_size=[32]`, `dim=[512]`, `lr=[5e-5]`, `num_classes=[14]`, `image_size=[128]`, `hybrid_loss_coeff=[0.006]`, `num_workers=[0]`.
- D3PM training: `validation_fraction=0.2`, `split_seed=0`, `patience=25`, `min_delta=5e-5`, `max_epochs=300`.
- Queue status: BA, ER, expdeg, SBM failed after two attempts; WS status remained running.
- Failure mode: GGME crashed with `ValueError: Input contains NaN` during normalized Laplacian spectrum in failed attempts. Successful-looking per-run D3PM manifests therefore do not imply full experiment completion.

### `parallel_20260729_144516`

- Config files: `runs_export/parallel_20260729_144516/experiments/*.json`.
- Datasets: all five synthetic datasets.
- SIGL cluster sweep: for each dataset, configs use duplicate seeds/replicates at `n_clusters=1`, `4`, and `8`:
  - ER: `nc0,nc1=1`, `nc2,nc3=4`, `nc4,nc5=8`
  - SBM: `nc6,nc7=1`, `nc8,nc9=4`, `nc10,nc11=8`
  - expdeg: `nc12,nc13=1`, `nc14,nc15=4`, `nc16,nc17=8`
  - BA: `nc18,nc19=1`, `nc20,nc21=4`, `nc22,nc23=8`
  - WS: `nc24,nc25=1`, `nc26,nc27=4`, `nc28,nc29=8`, but only `nc24,nc25` were queued/exported as queue status entries.
- Resolved common SIGL: `Res=128`, `n_epoch=100`, `lr=0.06`, `bootstrap_replicates=500`, seed 0, 4 workers.
- Resolved D3PM grid: `batch_size=[32]`, `dim=[512]`, `lr=[2e-5,3e-5,5e-5]`, `num_classes=[10,14]`, `image_size=[128]`, `hybrid_loss_coeff=[0.006,0.01]`, twelve candidates/run.
- Completion: 22 runs have three final `stage="ggme"` metrics rows; 8 are incomplete, retried, running, or missing final metrics.
- NaN warnings: several D3PM logs contain `nan` candidate losses, but queue-complete runs still selected finite best candidates and completed GGME. Treat the affected candidate histories as noisy evidence; use selected runtime metadata from manifests/logs.

### `parallel_20260803_063318`

- Config files: `runs_export/parallel_20260803_063318/experiments/*.json`.
- Datasets configured: ER, BA, WS.
- Queue/run artifacts present for ER `n_clusters=8,16,32` and BA `n_clusters=8`; all queue statuses read `running` and only `sigl.log` exists in per-run directories.
- Additional configs exist for BA `n_clusters=16,32` and WS `n_clusters=8,16,32`, but there are no matching per-run completion artifacts in this export.
- Shared DiGress and shared classical baselines completed for ER, BA, WS.
- Do not use per-run D3PM/GGME hyperparameters from this group as final execution values; runtime never reached those stages in the exported per-run manifests.

### `parallel_20260803_154001`

- Config files: `runs_export/parallel_20260803_154001/experiments/*.json`.
- Datasets: BA and WS.
- SIGL sweep: BA and WS each at `n_clusters=4,8,16`.
- Resolved D3PM grid: `batch_size=[32]`, `dim=[512]`, `lr=[1e-5,2e-5,3e-5]`, `num_classes=[10,14]`, `image_size=[128]`, `hybrid_loss_coeff=[0.006]`.
- Queue status: all 6 jobs complete, one attempt each.
- Runtime: each run completed SIGL, D3PM, GGME; each has 3 GGME rows and 100 OGAD/DiGress/classical graphs loaded.
- Interpretation: successful follow-up for BA and WS. The repository does not explicitly state whether these replace the BA/WS selections in `ogad_report_parallel_20260729_20260803.md`.

## Final Successful Run Candidates

The following table uses the existing ranking report as the primary repository-level selection when it names a best run. It also records stricter-status caveats.

| Dataset | Candidate final run | Group | Runtime `n_clusters` | SIGL graphons observed | D3PM train/val graphons | Validation-loss best -> GGME-selected final params | Final selected epoch | Status |
|---|---|---|---:|---:|---:|---|---:|---|
| `ba_n100_m3` | `20260802_011527_ba_n100_m3_nc19` | 20260729 | 1 | 500 | 400/100 | val-best `lr=2e-5, classes=10, hybrid=0.006` -> final `candidate_009: lr=3e-5, classes=14, hybrid=0.01` | 97 | Complete with 3 GGME rows, but stricter report excludes BA because other BA jobs remained incomplete. Later 20260803_154001 BA follow-ups complete but not explicitly selected as final. |
| `er_n100_p005` | `20260730_045033_er_n100_p005_nc5` | 20260729 | 8 | 4000 | 3200/800 | val-best `lr=3e-5, classes=14, hybrid=0.006` -> final `candidate_003: same` | 60 | Complete with 3 GGME rows; no later complete ER follow-up in inspected exports. |
| `expdeg_n100_lin` | `20260801_171551_expdeg_n100_lin_nc17` | 20260729 | 8 | 4000 | 3200/800 | val-best `lr=2e-5, classes=10, hybrid=0.006` -> final `candidate_007: lr=2e-5, classes=14, hybrid=0.01` | 115 | Complete; also included by stricter completed-dataset report. |
| `sbm_n100_4c` | `20260731_055853_sbm_n100_4c_nc10` | 20260729 | 8 | 2000 | 1600/400 | val-best `lr=2e-5, classes=10, hybrid=0.006` -> final `candidate_011: lr=5e-5, classes=14, hybrid=0.01` | 25 | Complete; also included by stricter completed-dataset report. Runtime graphon count is 2000 despite `n_clusters=8`. |
| `ws_n100_k8_p01` | `20260803_064123_ws_n100_k8_p01_nc24` | 20260729 | 1 | 250 | 200/50 | val-best `lr=2e-5, classes=10, hybrid=0.006` -> final `candidate_008: lr=3e-5, classes=10, hybrid=0.01` | 62 | Complete with 3 GGME rows, but stricter report excludes WS because other WS jobs remained incomplete. Later 20260803_154001 WS follow-ups complete but not explicitly selected as final. |

Successful BA/WS follow-up runs in `parallel_20260803_154001`:

| Run | Dataset | `n_clusters` | SIGL graphons | Best params | Best epoch | GGME-selected candidate |
|---|---|---:|---:|---|---:|---|
| `20260803_162715_ba_n100_m3_nc0` | BA | 4 | 2000 | `lr=1e-5`, `classes=14`, `hybrid=0.006` | 102 | `candidate_001` |
| `20260803_162715_ba_n100_m3_nc1` | BA | 8 | 4000 | `lr=1e-5`, `classes=14`, `hybrid=0.006` | 60 | `candidate_001` |
| `20260804_100803_ba_n100_m3_nc2` | BA | 16 | 8000 | `lr=2e-5`, `classes=14`, `hybrid=0.006` | 43 | `candidate_003` |
| `20260804_150351_ws_n100_k8_p01_nc3` | WS | 4 | 2000 | `lr=3e-5`, `classes=10`, `hybrid=0.006` | 81 | `candidate_004` |
| `20260805_104945_ws_n100_k8_p01_nc4` | WS | 8 | 4000 | `lr=3e-5`, `classes=14`, `hybrid=0.006` | 77 | `candidate_005` |
| `20260805_185229_ws_n100_k8_p01_nc5` | WS | 16 | 8000 | `lr=1e-5`, `classes=14`, `hybrid=0.006` | 47 | `candidate_001` |

## Core Method Parameters

| Parameter | Value | Fixed or config-dependent | Runs | Source | Thesis classification |
|---|---|---|---|---|---|
| Input graph train/test datasets | `CommunityGraphs_train.pkl`, `CommunityGraphs_test.pkl`; logs show 240 training graphs and 60 test graphs per selected dataset | dataset-dependent paths, fixed counts in inspected selected runs | all successful selected runs | experiment JSON and SIGL/GGME logs | Experiments |
| GGME validation holdout before SIGL bootstrap | 20% of training graphs: 192 model-train, 48 validation | fixed | successful selected runs | SIGL logs `GGME validation split` | Method |
| SIGL graphon resolution | finite graphon grid; runtime `Res=128` | grid representation is Method; value 128 is fixed experiment/implementation setting | all inspected runs | configs/logs | Method concept + Experiments value |
| SIGL training | 100 epochs, LR 0.06, INR batch 1024, GNN hidden `8,8,8`, INR hidden `20,20`, `w0=10` | fixed | all inspected runs | SIGL logs | Experiments/reproducibility |
| SIGL clustering | `n_clusters` varied by sweep | config-dependent | 1, 4, 8, 16, 32 depending on group | experiment JSON and logs | Experiments |
| SIGL bootstrap | cluster-bootstrap mapping is Method; 500 replicates, seed 0, no original cluster graphons, 4 workers are runtime settings | fixed settings in inspected runs | all inspected runs | configs/logs | Method concept + Experiments value |
| D3PM discretization | quantization formula is Method; class count `K` is swept/selected (`10` or `14`, 20260728 only `14`) | config/runtime-dependent | selected runs | manifests/source | Method formula + Experiments value |
| D3PM tensor/image size | one-channel square graphon tensor; runtime `128 x 128` | representation is Method; 128 is fixed experiment/implementation setting | all selected runs | configs/logs | Method concept + Experiments value |
| D3PM architecture | `DiT_Llama(1, num_classes, dim=512)` | fixed dim; classes config-dependent | selected runs | manifests/source | Method |
| D3PM diffusion steps | 1000 | fixed code runtime | selected runs | source | Method |
| D3PM candidate generation | 16 graphon tensors per candidate; final GGME samples cycle through selected tensors | fixed code runtime | selected runs | source | Method |
| Graph conversion | generated graphon tensors are dequantized, symmetrized, clamped, diagonal-zeroed; GGME sanitizes again, samples latent grid indices uniformly, and samples undirected edges by Bernoulli probabilities | fixed | selected runs | `src/d3pm/ogad_model_finetune.py`; `src/ggme/main.py`; `src/ggme/graphon_utils.py` | Method |

## Training Hyperparameters

| Parameter | Value | Fixed or config-dependent | Runs | Source | Thesis classification |
|---|---|---|---|---|---|
| D3PM batch size | 32 | fixed override | all successful selected runs | manifests/logs | Experiments |
| D3PM learning-rate grid | 20260729: `[2e-5,3e-5,5e-5]`; 20260803 follow-up: `[1e-5,2e-5,3e-5]`; 20260728: `[5e-5]` | config-dependent | group-specific | experiment JSON | Experiments |
| D3PM hybrid-loss grid | 20260729: `[0.006,0.01]`; 20260728 and 20260803 groups: `[0.006]` | config-dependent | group-specific | experiment JSON/manifests | Experiments |
| D3PM optimizer | AdamW | fixed code | all D3PM runs | source | Method |
| D3PM max epochs | 300 | fixed override | all successful selected runs | configs/manifests | Experiments |
| D3PM early stopping | patience 25, min_delta 5e-5, best validation loss | fixed override | all successful selected runs | configs/manifests/logs | Experiments |
| D3PM train/validation split | 80/20, seed 0 | fixed | all successful selected runs | manifests | Experiments |
| Checkpoint/model selection | training stores all candidates and selects lowest validation loss; GGME validation selection then copies selected candidate checkpoint/params/graphons into final output | fixed when `graph_selection` present | all successful selected runs | source/manifests | Method |

## Dataset-Specific Parameters

| Dataset | Train graphs | Test graphs | Selected/important `n_clusters` | Runtime SIGL graphons in selected 20260729 run | Graph size in GGME logs | Source |
|---|---:|---:|---|---:|---|---|
| `ba_n100_m3` | 240 | 60 | selected report: 1; follow-up successful: 4/8/16 | 500 | 100 nodes | SIGL/GGME logs |
| `er_n100_p005` | 240 | 60 | selected report: 8 | 4000 | 100 nodes | SIGL/GGME logs |
| `expdeg_n100_lin` | 240 | 60 | selected report: 8 | 4000 | 100 nodes | SIGL/GGME logs |
| `sbm_n100_4c` | 240 | 60 | selected report: 8 | 2000 | 100 nodes | SIGL/GGME logs |
| `ws_n100_k8_p01` | 240 | 60 | selected report: 1; follow-up successful: 4/8/16 | 250 | 100 nodes | SIGL/GGME logs |

## Evaluation Parameters

| Parameter | Value | Fixed/config-dependent | Runs | Source | Thesis classification |
|---|---|---|---|---|---|
| Evaluation graph count | 100 OGAD, 100 DiGress, 100 per classical baseline | fixed config | successful GGME runs | GGME logs | Experiments |
| GGME seed | 0 | fixed | successful GGME runs | configs/logs | Experiments |
| Descriptors | degree distribution; clustering coefficient; normalized Laplacian spectrum | fixed code | successful GGME runs | source/logs | Method |
| Descriptor histogram settings | degree: NetworkX degree histogram, density normalized; clustering: 200 bins range `[0,1]`; spectrum: 200 bins range `[0,2]` | fixed code | successful GGME runs | source | Method |
| Perturbation | `AddEdges`, values `0.15..0.95` step 0.05, random seed 42 | fixed code | successful GGME runs | source | Method |
| Kernel search | Gaussian and linear kernels; sigmas 0.1, 1, 10; Pearson correlation | fixed code | successful GGME runs | source/logs | Method |
| MMD computation | exact MMD, no linear approximation, descriptors density-normalized | fixed code | successful GGME runs | source | Method |
| Candidate selection metric | `digress_target_ratio` | config override from default `mean_rank` | inspected thesis configs | configs/manifests | Experiments |

## Baseline Parameters

| Baseline | Runtime setting | Source | Thesis classification |
|---|---|---|---|
| DiGress | Enabled in all inspected experiment JSONs; run once per dataset/shared key and mounted into per-run GGME. Logs show `GraphTransformer`, 7.1M params, input feature sizes `7 2 12`, 100 loaded samples in GGME. Hydra config `community_graphs` has 200 epochs, batch 32, gradient accumulation 16, diffusion steps 1000, 8 layers, hidden dims dx 256/de 64/dy 64, 8 heads, final samples 160. | `runs_export/*/shared_runs/*digress*/digress.log`; `src/baselines/DiGress/configs/experiment/community_graphs.yaml` | Baseline/Experiments |
| Classical ER | Fit global edge probability from training graphs; generate 100 graphs with test-size sampling; seed 42 | `src/baselines/classical_graphs/models.py`; baseline logs/config | Baseline |
| Classical configuration model | Sample degree sequence from pooled training degrees; sanitize simple graph; 100 graphs; seed 42 | source/config | Baseline |
| Classical SBM | Greedy modularity communities per training graph, fit block probabilities, sample templates; 100 graphs; seed 42 | source/config | Baseline |
| Classical DC-SBM | Same SBM templates with sampled degree weights clipped 0.05 to 5.0; 100 graphs; seed 42 | source/config | Baseline |
| Paper baseline | Config points to degree-corrected SBM pickle; when baselines enabled, `paper_graphs` is overwritten to generated classical DC-SBM path | `scripts/experiment_grid.py`; experiment JSON | Baseline/Discrepancy |

## Log-Based Pipeline Validation

Expected shell flow in `scripts/run_ogad_experiment.sh` is:

`SIGL -> optional DiGress -> optional classical baselines -> D3PM finetune + inference -> GGME`.

Observed parallel runtime flow is different when shared stages are enabled:

1. `scripts/run_from_merged_parallel.py` runs shared classical baselines once per dataset/config key.
2. It runs shared DiGress once per dataset/config key.
3. Per-candidate OGAD jobs run SIGL, skip/mount the shared DiGress and baseline outputs, then run D3PM and GGME.

Evidence:

- Shared directories contain complete status JSONs, launcher logs, and `shared_runs/*/baselines.log` or `digress.log`.
- Per-run logs in the successful groups generally include `sigl.log`, `d3pm.log`, and `ggme.log`, not per-run `digress.log`/`baselines.log`.
- Per-run `ggme.log` confirms 100 DiGress graphs and 100 graphs from each classical baseline were loaded.
- D3PM manifests confirm split sizes and candidate selections, for example selected 20260729 runs have train/validation graphon splits of 400/100, 3200/800, 1600/400, or 200/50 depending on actual SIGL output count.

## Implementation Trace: Graphs -> SIGL Graphons -> D3PM Dataset

Verified fact: raw OGAD training graphs are loaded as adjacency matrices for SIGL, not as NetworkX graphs. `src/sigl/tools.py::load_graphs` first tries `community_graphs.sigl_loader.load_community_graphs_for_sigl`, which validates CSR matrices and densifies them to C-contiguous `np.float32` arrays. SIGL then converts each dense adjacency into a PyTorch Geometric `Data` object with random scalar node features, `edge_index` from nonzero adjacency entries, and `y` equal to the flattened adjacency values.

Verified fact: when `graph_selection` is present, SIGL first creates a deterministic train/validation split over the original training graphs. With the thesis config this is 80/20 using seed 0; successful logs show 240 input training graphs split into 192 model-training graphs and 48 GGME-validation graphs. Only the model-training subset is clustered and bootstrapped for SIGL graphon estimation.

Verified fact: one saved SIGL graphon image is one graphon-estimation job, not one original graph except in the unclustered/default `n_clusters=None` branch. For clustered thesis runs, `cluster_graphs_from_arrays` builds descriptor vectors from degree distribution, clustering coefficient, and normalized Laplacian spectrum, pads them, standardizes them, and runs `KMeans(n_clusters=n_clusters, random_state=0)`. The returned clusters include only labels that receive at least one graph. Empty labels are not emitted, so the effective cluster count can be smaller than requested `n_clusters`.

The clustered bootstrap mapping implemented by `build_cluster_bootstrap_plan` is:

```text
Input clusters C_0..C_(C_eff-1), bootstrap_replicates R, include_original_clusters flag.
If R == 1:
    one original job per non-empty cluster
If R > 1 and include_original_clusters == false:
    R jobs per non-empty cluster; each job samples |C_c| graph indices with replacement from C_c
If R > 1 and include_original_clusters == true:
    one original job plus R bootstrap jobs per non-empty cluster
```

Thus for thesis-style clustered runs with `R=500` and `include_original_clusters=false`, `N_graphons = C_eff * 500`, where `C_eff` is the number of non-empty KMeans labels returned by the implementation. This explains cases such as SBM selected run `n_clusters=8` but 2000 graphons: `C_eff=4`. When `n_clusters=None`, the code instead initializes one graphon job per model-training graph.

Each graphon job trains SIGL on the graphs listed in that job. The learned GCN predicts one scalar latent coordinate per node by applying GCN layers with ReLU, a final linear projection, sigmoid, and min-max normalization. Nodes are sorted by these predicted coordinates. Sorted adjacency matrices are average-pooled with window size `h=int(log(N))` for the default `h_size=4`; pooled cell centers `(i+0.5)/k, (j+0.5)/k` become INR inputs, pooled adjacency values become targets, and graph-level weights are proportional to `graph_node_count / total_nodes_in_job`. The final INR is a SIREN MLP with hidden dimensions `20,20`, sine activations, and sigmoid output. It is trained with weighted MSE against the pooled histograms. The saved graphon grid evaluates the INR on a `Res x Res` midpoint grid, symmetrizes it by averaging with its transpose, zeroes the diagonal, and saves it as `sample_<sample_id>.png` via `torchvision.utils.save_image`.

Strong inference: because the final INR uses sigmoid output and `get_graphon` symmetrizes/zeroes the grid before saving, the graphon PNG is intended to encode probabilities in `[0,1]`. The source does not add an explicit clamp before `save_image`; any image-format quantization is handled by `torchvision.save_image` and later PIL loading.

Sources:
- `community_graphs/sigl_loader.py :: load_community_graphs_for_sigl`, `densify_graph`
- `src/sigl/tools.py :: load_graphs`, `nx2torch`, `GCN`, `SIGL`, `coords_prediction`, `graph2XY`, `train_graphon`, `get_graphon`, `save_graphon_for_diff`
- `src/sigl/Main.py :: build_graphon_payload`, `learn_and_save_graphon_job`, graph-selection and graphon-job construction
- `src/sigl/clusterize.py :: cluster_graphs_from_arrays`
- `src/sigl/bootstrap_utils.py :: build_cluster_bootstrap_plan`, `expected_graphon_count`
- Representative runtime: `runs_export/parallel_20260729_144516/runs/20260730_045033_er_n100_p005_nc5/sigl.log`

## Exact Graphon Quantization and D3PM State Semantics

Verified fact: D3PM loads SIGL outputs with HuggingFace `imagefolder`. Each `sample_*.png` is opened as an image, converted to grayscale (`"L"`), transformed by `torchvision.transforms.ToTensor()`, and stored as `example["data"]`. A single D3PM dataset example is therefore one saved SIGL graphon image as a float tensor with shape `[1, image_size, image_size]`; successful thesis runs use `image_size=128`.

Verified fact: quantization happens inside the D3PM training/validation batch path, not during SIGL output generation and not inside `DiT_Llama`. For every pixel value `W_ij` loaded as a float tensor, and for selected class count `K = num_classes`, the implemented operation is:

```text
q_ij = clamp(round((K - 1) * W_ij), min=0, max=K - 1)
```

The output dtype is `torch.long`; the class index range is `{0, ..., K-1}`; batch shape is `[B, 1, 128, 128]` for thesis runs. Boundary behavior follows the code directly: `W=0` maps to class 0; `W=1` maps to class `K-1`; values outside `[0,1]`, if any arrive from image loading, are clipped after rounding by `clamp(0, K-1)`. The rounding operator is PyTorch tensor `.round()`. The repository does not define a custom half-tie rule beyond PyTorch's operator.

### Quantization formula

For continuous loaded tensor value `W_ij` and `K` classes:

```text
Q_K(W_ij) = min(K - 1, max(0, round((K - 1) W_ij)))
```

Worked example, verified from the formula:

```text
W_ij = 0.73, K = 14
(K - 1) W_ij = 13 * 0.73 = 9.49
round(9.49) = 9
Q_14(0.73) = 9
```

Sources:
- `src/d3pm/ogad_model_finetune.py :: apply_transform`, `load_graphons_dataset`, `_prepare_batch`
- Representative runtime: `runs_export/parallel_20260729_144516/runs/20260730_045033_er_n100_p005_nc5/d3pm.log` shows 4000 imagefolder examples for 4000 SIGL graphons

## Exact Dequantization and Generated Tensor Sanitization

Verified fact: D3PM sampling returns categorical class tensors. OGAD dequantizes only after sampling the final image sequence element:

```text
D_K(k) = k / (K - 1)
```

where `k` is the generated class index and `K` is the selected candidate's `num_classes`. The resulting tensor is moved to CPU and represented as `float`. The immediate post-processing order in D3PM generation is:

1. divide final categorical tensor by `K-1`;
2. symmetrize each channel matrix as `(W + W^T)/2`;
3. clamp to `[0,1]`;
4. zero the diagonal;
5. save a tensor with shape `[16, 1, image_size, image_size]` to the candidate `.pt` file.

Quantization and dequantization are not exact inverses for arbitrary continuous graphon values. Quantization collapses intervals to class indices, and dequantization maps each class to an endpoint grid value `k/(K-1)`, not to bin centers under a floor/binning scheme. Post-sampling symmetrization can also create off-grid averaged probabilities such as `(k_a + k_b) / (2(K-1))`.

Sources:
- `src/d3pm/ogad_model_finetune.py :: generate_graphons_from_checkpoint`
- `src/ggme/graphon_utils.py :: sanitize_graphon_matrix`

## D3PM Forward Process, Reverse Model, and Loss

Verified fact: OGAD uses `D3PM(DiT_Llama(1, num_classes, dim=512), 1000, num_classes=K, hybrid_loss_coeff=lambda)` with the D3PM default `forward_type="uniform"`.

The forward transition schedule is code-defined. For `T=1000`, the code sets `steps_t = t/T` for `t=0..T`, then:

```text
a_bar(t) = cos(((steps_t + 0.008) / 1.008) * pi/2)
beta_t = min(1 - a_bar(t) / a_bar(t-1), 0.999),  t=1..T
```

For each `beta_t`, the one-step categorical transition matrix is row-stochastic and shared across all tensor locations:

```text
Q_t[i,j] = beta_t / K                         if i != j
Q_t[i,i] = 1 - ((K - 1) * beta_t / K)
```

This is a uniform discrete corruption transition. There is no absorbing state, no mask token, no graph-specific transition, and no location-specific transition matrix. Cumulative matrices are stored as `Qbar_t = Q_1 Q_2 ... Q_t`.

Training samples a timestep uniformly with `torch.randint(1, n_T, (B,))`, so for `n_T=1000` the sampled training timesteps are integers `1..999`. The corrupted tensor `x_t` is sampled from `q(x_t | x_0)` by indexing `Qbar_t[x_0, :]` and applying Gumbel-max categorical sampling.

Verified fact: the reverse network predicts logits for the clean categorical tensor `x_0`, not direct noise and not direct `x_{t-1}` logits. The D3PM wrapper converts those predicted `x_0` logits into predicted posterior logits for `q(x_{t-1} | x_t, x_0)` by combining:

```text
fact1 = Q_t^T[x_t, :]
fact2 = softmax(predicted_x0_logits) @ Qbar_(t-1)
posterior_logits = log(fact1 + eps) + log(fact2 + eps)
```

For `t=1`, the posterior branch is replaced with the `x_0` logits themselves. The true posterior uses the same computation with one-hot/log one-hot `x_0`.

The implemented loss is:

```text
L = L_vb + lambda * L_ce
```

where `lambda = hybrid_loss_coeff`. `L_vb` is the mean KL divergence between the true posterior categorical distribution and the predicted posterior categorical distribution after flattening all batch and pixel positions. `L_ce` is PyTorch `CrossEntropyLoss` between flattened predicted `x_0` logits and flattened clean class indices. No additional timestep weighting is implemented. Training uses AdamW on `d3pm.x0_model.parameters()`, mixed precision autocast/scaler, and gradient clipping at norm 5.0.

Sources:
- `src/d3pm/d3pm_runner.py :: D3PM.__init__`, `_at`, `q_sample`, `q_posterior_logits`, `vb`, `forward`, `p_sample`
- `src/d3pm/training_utils.py :: combine_d3pm_losses`
- `src/d3pm/ogad_model_finetune.py :: train_epoch`, `validate_epoch`, `finetune_model`

## DiT_Llama Architecture Used by OGAD

Verified fact: the executed image model is `src/d3pm/dit.py::DiT_Llama`, not `DummyX0Model` and not `DDiT_Llama`. For thesis runs the constructor is `DiT_Llama(in_channels=1, N=K, dim=512)`. Default architecture values left in force are `input_size=32` (stored but not used to check OGAD's 128 input), `patch_size=2`, `n_layers=5`, `n_heads=16`, `multiple_of=256`, and `norm_eps=1e-5`.

Main shape trace for an OGAD batch:

```text
x categorical input:       [B, 1, 128, 128]
one-hot residual tensor:   [B, 1, 128, 128, K]
scaled float image:        [B, 1, 128, 128] in [-1, 1]
conv feature map:          [B, 256, 128, 128]      # dim/2 channels
patchified tokens:         [B, 4096, 1024]         # 64*64 patches, each 2*2*256
token embeddings:          [B, 4096, 512]
after 5 transformer blocks:[B, 4096, 512]
final patch projection:    [B, 4096, 2*2*(2*K)]
unpatchified output:       [B, 2*K, 128, 128]
reshaped/chunked logits:   [B, 1, 128, 128, K]
```

The model uses a two-layer convolutional stem with SiLU and GroupNorm, a linear patch embedder, sinusoidal timestep embeddings passed through an MLP, rotary-position self-attention inside transformer blocks, adaptive layer-norm modulation from the timestep embedding, gated SiLU feed-forward layers, and a final adaptive layer-norm projection. Because OGAD passes `cond=None`, the label/class conditioning embedder is not used in thesis OGAD runs.

The final return value is not a plain projection: after unpatchifying, the output is split into logits and a gate, and the model returns `logits + one_hot(x_t) * abs(1 + gate)`. This biases the predicted `x_0` logits with a gated residual on the current noisy class at each tensor location.

Sources:
- `src/d3pm/dit.py :: DiT_Llama`, `TransformerBlock`, `Attention`, `FeedForward`, `FinalLayer`, `TimestepEmbedder`
- `src/d3pm/ogad_model_finetune.py :: finetune_model`, `generate_graphons_from_checkpoint`

## D3PM Sampling Procedure

Verified fact: candidate graphon generation initializes `x_T` directly as independent uniform categorical noise:

```text
x = randint(0, K, shape=[16, 1, image_size, image_size])
```

The reverse loop in `sample_with_image_sequence` runs `t=999,998,...,1` for `n_T=1000`. At every step it calls `p_sample`, which predicts `x_0` logits, converts them to posterior logits over `x_{t-1}`, and uses Gumbel-max categorical sampling. At `t=1`, the multiplier `not_first_step` is zero, so no Gumbel noise is added and the update becomes an argmax over the final logits. There is no temperature parameter, no argmax before the final step, and no mask/clipping in categorical state space.

The D3PM generation seed is `training_config["split_seed"]` (0 in inspected thesis configs) passed into `set_reproducible_seed`, which seeds Python `random`, NumPy, Torch CPU, and Torch CUDA if available. Device is `cuda` when available, otherwise CPU. `stride=40` only controls which intermediate tensors are retained; OGAD uses the last retained tensor as the final sample.

Exactly 16 generated graphon-like tensors are produced per D3PM candidate because the initial noise batch size is hard-coded to 16. GGME later creates 100 finite graphs by cycling through the selected candidate's 16 tensors: graph `s` uses tensor index `s mod 16`.

Sources:
- `src/d3pm/ogad_model_finetune.py :: set_reproducible_seed`, `generate_graphons_from_checkpoint`
- `src/d3pm/d3pm_runner.py :: p_sample`, `sample_with_image_sequence`
- `src/ggme/graphon_utils.py :: cyclic_graphon_index`
- `src/ggme/main.py :: generate_graphs`

## Exact Tensor -> Graph Conversion

Verified fact: the edge rule is Bernoulli sampling from graphon probabilities, not thresholding. The complete executed path for an OGAD candidate is:

```text
generated categorical tensor [16,1,128,128]
-> dequantize: k/(K-1)
-> symmetrize, clamp to [0,1], diagonal zero in D3PM generation
-> save candidate .pt
-> optional GGME candidate selection copies one candidate .pt to final tensors.pt
-> GGME loads tensors.pt
-> for each generated graph, select graphon index sample_index mod 16
-> remove channel dimension by graphon[0]
-> sanitize again: symmetrize, clamp, diagonal zero
-> sample node count M from empirical test graph sizes with rejection if M > 128
-> sample latent grid indices idx_i = floor(U_i * 128), clipped to [0,127]
-> probability matrix P_ab = W[idx_a, idx_b]
-> sample A_ab ~ Bernoulli(P_ab)
-> keep strict upper triangle and mirror it: A = triu(A,1) + triu(A,1)^T
-> nx.from_numpy_array(A)
```

Mathematically, for a sanitized finite grid `W` of size `R x R`, a generated graph with `M` nodes uses IID `U_i ~ Uniform(0,1)`, grid indices `z_i = min(R-1, floor(R U_i))`, and for `i < j`:

```text
A_ij ~ Bernoulli(W[z_i, z_j])
A_ji = A_ij
A_ii = 0
```

The implementation samples a full `M x M` Bernoulli matrix first, then discards the diagonal and lower triangle; effectively only the sampled upper-triangular entries survive. There is no interpolation/resizing, no top-left `M x M` crop, and no random subset of rows/columns without replacement. Latent coordinates are not continuous values used directly in the probability lookup; they are discretized to grid indices by `floor(U_i * R)`. If repeated latent indices occur, multiple generated nodes can share the same graphon grid row/column. If a requested test size exceeds graphon resolution, `_sample_M` rejects and resamples up to 64 times, then clamps the last draw to `[1,R]`.

Post-processing not performed: no largest connected component extraction, no connectivity enforcement, no isolated-node removal, no invalid-output rejection, no retry/resampling based on graph structure, and no additional node relabeling beyond NetworkX's default integer labels. Self loops are removed by the upper-triangle operation; undirected symmetry is enforced by mirroring.

Sources:
- `src/d3pm/ogad_model_finetune.py :: generate_graphons_from_checkpoint`
- `src/ggme/main.py :: generate_graphs`, `_sample_M`, `sample_graph_from_graphon`
- `src/ggme/graphon_utils.py :: sanitize_graphon_matrix`, `probability_matrix`, `cyclic_graphon_index`, `validate_generation_request`

## End-to-End Data Shapes and Semantics

| Stage | Object | Python type | Shape | Value range | Semantic meaning | Produced by | Consumed by |
|---|---|---|---|---|---|---|---|
| Raw persisted graph | CSR adjacency list entry | `scipy.sparse.csr_matrix` | `[n,n]` | binary | one finite training/test graph | dataset pickle | `community_graphs.sigl_loader` / `src/ggme/utils.py` |
| Sparse adjacency matrix | validated CSR | `scipy.sparse.csr_matrix` | `[n,n]` | binary, symmetric per metadata when available | memory-efficient finite graph | `load_community_graphs_sparse` | `densify_graph` |
| Dense SIGL input | adjacency matrix | `np.ndarray` | `[n,n]` | `0/1` float32 | finite graph for SIGL | `load_community_graphs_for_sigl` | `src/sigl/tools.py::nx2torch` |
| Node-order/alignment representation | latent node coordinate | `torch.Tensor` | `[n,1]` | sigmoid then min-max normalized | scalar ordering coordinate per node | `GCN.forward` in SIGL | `graph2XY` sorting |
| Continuous SIGL graphon | estimated grid | `np.ndarray` | `[Res,Res]` | intended `[0,1]` | INR graphon estimate on midpoint grid | `get_graphon` | `save_graphon_for_diff` |
| Saved graphon grid | image file | PNG grayscale | `128 x 128` in thesis runs | image-encoded probability | one graphon-estimation job | `save_image` | D3PM `imagefolder` loader |
| D3PM dataset sample before quantization | grayscale tensor | `torch.Tensor` | `[1,128,128]` | `[0,1]` after `ToTensor()` | one SIGL graphon image | `apply_transform` | `_prepare_batch` |
| Quantized D3PM input | class tensor | `torch.Tensor` long | `[1,128,128]` | integers `0..K-1` | categorical graphon state | `_prepare_batch` | D3PM dataloader batch |
| Batched D3PM tensor | class tensor | `torch.Tensor` long | `[B,1,128,128]` | integers `0..K-1` | batch of categorical graphons | DataLoader + `_prepare_batch` | `D3PM.forward` |
| Model embedding/token representation | patch tokens | `torch.Tensor` | `[B,4096,512]` | learned features | transformer input sequence | `DiT_Llama.patchify` + `x_embedder` | 5 transformer blocks |
| Model logits | clean-state logits | `torch.Tensor` | `[B,1,128,128,K]` | real logits | predicted `x_0` categorical logits | `DiT_Llama.forward` | `D3PM.q_posterior_logits`, CE |
| Sampled categorical `x_0` | final class tensor | `torch.Tensor` long | `[16,1,128,128]` | integers `0..K-1` | generated categorical graphon states | `sample_with_image_sequence` | dequantization |
| Dequantized graphon-like matrix | float tensor | `torch.Tensor` | `[16,1,128,128]` | `{k/(K-1)}` before symmetrization | generated probability-like grid | `generate_graphons_from_checkpoint` | symmetrization |
| Symmetrized probability matrix | float tensor | `torch.Tensor` | `[128,128]` per graphon | `[0,1]`, diagonal zero | finite graphon grid for graph sampling | D3PM generation and GGME sanitize | `sample_graph_from_graphon` |
| Selected `n x n` probability representation | probability matrix | `torch.Tensor` | `[M,M]` | `[0,1]` | probabilities for sampled latent grid indices | `probability_matrix` | `torch.bernoulli` |
| Generated adjacency matrix | binary matrix | `torch.Tensor` then `np.ndarray` | `[M,M]` | `0/1`, symmetric, diagonal zero | sampled finite graph | `sample_graph_from_graphon` | NetworkX conversion |
| NetworkX graph | generated graph | `networkx.Graph` | `M` nodes | unweighted undirected | OGAD output graph | `nx.from_numpy_array` | GGME descriptors |
| GGME descriptor vectors/histograms | descriptor array | `np.ndarray` | variable degree length; 200 bins for clustering/spectrum | density-normalized when requested | graph summary statistic | descriptor functions | `evaluate_mmd` |
| MMD inputs | descriptor distributions | list/array of arrays | `#graphs x descriptor_length` after padding | real-valued | empirical descriptor samples | `evaluate_mmd` | kernel wrapper and MMD |

Sources:
- `community_graphs/sigl_loader.py`; `src/sigl/tools.py`; `src/d3pm/ogad_model_finetune.py`; `src/d3pm/dit.py`; `src/d3pm/d3pm_runner.py`; `src/ggme/main.py`; `src/ggme/evaluate.py`; `src/ggme/metrics/descriptor_functions.py`

## End-to-End OGAD Method Pseudocode

Repository-grounded pseudocode for one configured OGAD run:

```text
Input:
    training graph pickle, test graph pickle
    SIGL params: Res, n_epoch, lr, GNN dims, INR dims, batch_size, h_size
    clustering/bootstrap params: n_clusters, bootstrap_replicates, seed, include_original_clusters
    D3PM params: K grid, dim, batch_size, lr grid, hybrid_loss_coeff grid, max_epochs, patience
    GGME params: num_generated_graphs, seed, descriptors, kernel/sigma search

If graph_selection is enabled:
    deterministically split training graphs into model_train and GGME-validation graphs
Else:
    use all training graphs for model_train and no GGME candidate-validation set

Load model_train graphs as dense adjacency matrices for SIGL.

If n_clusters is not None:
    compute descriptor features for each model_train graph
    run standardized KMeans
    keep only non-empty clusters
    for each non-empty cluster:
        create bootstrap_replicates graphon jobs by sampling graph indices with replacement
        optionally add the original whole-cluster job
Else:
    create one graphon job per model_train graph

For each graphon job:
    train SIGL GCN+INR to predict adjacency entries from learned node coordinates
    sort job graphs by learned node coordinates
    average-pool sorted adjacency matrices into histogram observations
    train final SIREN INR on pooled observations with graph-size weights
    evaluate INR on a Res x Res midpoint grid
    symmetrize the grid and zero the diagonal
    save sample_<id>.png

Load sample_*.png with D3PM imagefolder.
Create deterministic D3PM train/validation split over graphon images.

For each D3PM hyperparameter candidate:
    for each epoch until max_epochs or early stopping:
        load graphon image batch as float tensors
        quantize each value with q = clamp(round((K-1)W), 0, K-1)
        sample timestep t in 1..999
        sample x_t from the uniform-transition D3PM forward process
        predict x_0 logits with DiT_Llama
        compute KL posterior loss plus lambda * cross-entropy-to-x_0
        update with AdamW
        evaluate validation loss
        retain best validation-loss state dict
    save candidate checkpoint, params, and 16 generated dequantized graphon tensors

Select the lowest validation-loss D3PM candidate as the D3PM best candidate.

If GGME candidate selection is enabled:
    for each saved D3PM candidate tensor file:
        generate num_generated_graphs graphs from its 16 graphon tensors using validation graph sizes
        compute validation GGME descriptors and MMDs
    rank candidates by configured selection metric
    copy selected candidate checkpoint, params, and tensor file to final output paths

Load final graphon tensor file.
For sample_index in 0..num_generated_graphs-1:
    choose graphon sample_index mod 16
    sanitize graphon
    sample node count from empirical test graph sizes
    sample latent grid indices uniformly
    sample undirected Bernoulli edges from W[idx_i, idx_j]
    convert adjacency to NetworkX graph

Load test graphs, DiGress graphs if enabled, paper/classical baselines if configured.
For each descriptor:
    perturb test graphs with AddEdges
    choose kernel and sigma by Pearson correlation against perturbation level
    compute exact MMD between test graphs and each generated/baseline graph set
    append GGME metric rows
```

Sources:
- `src/sigl/Main.py`; `src/sigl/bootstrap_utils.py`; `src/sigl/clusterize.py`; `src/sigl/tools.py`
- `src/d3pm/ogad_model_finetune.py`; `src/d3pm/d3pm_runner.py`; `src/d3pm/dit.py`; `src/d3pm/training_utils.py`
- `src/ggme/main.py`; `src/ggme/evaluate.py`; `src/ggme/correlation.py`; `src/ggme/selection_utils.py`

## Static Validation of Mandatory Transitions

- A real saved SIGL graphon can be followed into D3PM: `src/sigl/Main.py::build_graphon_payload` writes `sample_<id>.png`; `save_graphon_for_diff` uses `save_image`; `src/d3pm/training_utils.py::load_d3pm_paths_config` resolves `dataset.sigl_graphons_dir`; `src/d3pm/ogad_model_finetune.py::load_graphons_dataset` reads that directory with `imagefolder`. Representative ER run `20260730_045033_er_n100_p005_nc5` logs `bootstrap graphons: 4000` and D3PM logs `Generating train split: 4000 examples`.
- A D3PM training sample can be followed through quantization into the model: `apply_transform` creates `[1,128,128]` float data; `_prepare_batch` converts it to long categorical classes; `train_epoch` passes `x_cat` into `D3PM.forward`; `DiT_Llama.forward` embeds and patchifies it.
- A generated categorical sample can be followed through dequantization into graph conversion: `sample_with_image_sequence` returns class tensors; `generate_graphons_from_checkpoint` divides by `K-1`, sanitizes, and saves `.pt`; `src/ggme/main.py` loads `graphons_pt` and calls `generate_graphs`.
- One generated graph can be followed into GGME: `generate_graphs` returns `networkx.Graph` objects; `compute_ggme` passes them to `evaluate_mmd`; descriptor functions convert each graph to arrays/histograms; `metrics.mmd.mmd` computes exact MMD. Representative ER `ggme.log` confirms `Loaded 100 OGAD graphs from /app/data/predictions/graphons/tensors.pt` and three descriptor rows.

Sources:
- Representative config/logs: `runs_export/parallel_20260729_144516/experiments/er_n100_p005_nc5.json`; `runs_export/parallel_20260729_144516/runs/20260730_045033_er_n100_p005_nc5/sigl.log`; `d3pm.log`; `ggme.log`; `manifest.json`; `metrics.jsonl`

## Failed, Partial, or Superseded Runs

| Group | Evidence | Classification |
|---|---|---|
| `parallel_20260728_155720` | Queue statuses failed for BA/ER/expdeg/SBM after two attempts; WS remained running. GGME logs crash with `ValueError: Input contains NaN`; only two final GGME descriptor rows. | Failed/partial, not final. |
| `parallel_20260729_144516` early partial attempts | `expdeg_nc15/nc16/nc17` and `ba_nc18` have early incomplete run directories before retry-complete run directories. | Superseded by later retry directories with same config names and full `Run complete`. |
| `parallel_20260729_144516` BA `nc21,nc22,nc23` | Some run manifests contain D3PM results but queue status remains running/no launcher completion and no final GGME rows. | Partial, not final. |
| `parallel_20260729_144516` WS `nc25` | running/no final metrics; configs for WS `nc26-29` have no queue status entries in export. | Partial/not executed in export. |
| `parallel_20260803_063318` | Per-run manifests only reach SIGL; queue statuses running; no D3PM/GGME completion. | Partial follow-up, not final. |
| `parallel_20260803_154001` | All 6 queue jobs complete; all have three GGME rows. | Successful follow-up for BA/WS. Final thesis use unresolved unless explicitly selected elsewhere. |

## Documentation-Implementation Discrepancies

- The prompt path `experiments/parallel_*` does not exist in the repo root; exported evidence lives under `runs_export/parallel_*`.
- `scripts/run_ogad_experiment.sh` suggests a serial stage order with optional DiGress/baselines inside each run, but parallel execution precomputes those stages as shared outputs.
- `paper_graphs` in configs is derived from the generated classical DC-SBM output when baselines are enabled, not necessarily the dataset's original `paper_baseline_graphs_path`.
- Code defaults are not thesis runtime defaults: SIGL `Res` defaults to 1000, but runtime uses 128; D3PM defaults include dim 128/256 and classes 8, but runtime uses dim 512 and classes 10/14.
- Some queue-complete D3PM logs contain NaN losses for unsuccessful candidates. Runtime manifests and GGME selections still identify finite selected candidates, but this should be mentioned in an appendix or limitations note.

## Unknowns / Missing Information

- Scheduler logs: no exported Slurm or external scheduler logs were found beyond queue launcher/status logs. Inspected `runs_export/parallel_*/queue`, run logs, manifests, and status JSONs. Blocks: Appendix/reproducibility only.
- DiGress fully resolved runtime YAML snapshot: no per-run resolved Hydra YAML snapshot was exported. The best available evidence remains DiGress logs plus `src/baselines/DiGress/configs/experiment/community_graphs.yaml`. Blocks: Baseline/Appendix, not OGAD Method.
- BA/WS final supersession: the repository does not explicitly declare whether successful `parallel_20260803_154001` BA/WS follow-ups supersede the 20260729 BA/WS selections in `runs_export/ogad_report_parallel_20260729_20260803.md`. Blocks: Experiments final-table wording, not Method.
- Exact root cause of every NaN D3PM candidate: logs identify where NaNs appeared and whether the enclosing job completed, but the repository evidence does not diagnose each candidate-level numerical failure. Inspected D3PM logs and manifests. Blocks: Appendix/limitations only.
- Original dataset-generation runtime flags: the four inspected run exports contain train/test pickle paths and downstream loading behavior, but not the original dataset-generation logs. Source defaults exist in `community_graphs/config.py`, but runtime generation evidence was not exported. Blocks: Experiments reproducibility, not OGAD Method.
- External image serialization details between SIGL and D3PM: repository code shows `torchvision.utils.save_image` writes SIGL graphon PNGs and `PIL -> ToTensor()` reloads them, but the repository does not vendor the exact torchvision/PIL implementation or version-specific float-to-PNG rounding behavior. The downstream D3PM categorical formula on the loaded tensor is known exactly. Inspected `src/sigl/tools.py::save_graphon_for_diff` and `src/d3pm/ogad_model_finetune.py::apply_transform`. Blocks: Appendix/reproducibility nuance; Method can describe the persisted image intermediate and exact loaded-tensor quantization.
- Half-tie behavior of `torch.round`: OGAD code calls PyTorch `.round()` without a custom tie convention. The implemented operator is known; a version-specific half-tie statement is not encoded in repository source or exports. Blocks: Appendix/reproducibility nuance only.

## Method Chapter Readiness

- [x] Exact graph input representation known
- [x] Exact SIGL output semantics known
- [x] Exact SIGL bootstrap/cluster -> graphon dataset mapping known
- [x] Exact graphon quantization formula known
- [x] Exact D3PM input shape and categorical semantics known
- [x] Exact D3PM transition/noise process known
- [x] Exact D3PM reverse target known
- [x] Exact D3PM loss known
- [x] Relevant `DiT_Llama` architecture known
- [x] Exact D3PM sampling process known
- [x] Exact dequantization formula known
- [x] Exact tensor -> graph edge-sampling rule known
- [x] Graph structural post-processing known
- [x] GGME evaluation procedure known
- [x] Method vs Experiments parameters separated
- [x] End-to-end shapes documented
- [x] End-to-end pseudocode documented

Checklist note: the code-level Method is ready. The remaining unknowns above affect final experiment selection wording or appendix-level reproducibility details, not the core OGAD method flow.
