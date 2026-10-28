# OGAD good/bad examples for thesis
Selection is based on `runs_export/ogad_completed_run_ranking.csv` and final `ggme` rows in each run `metrics.jsonl`. Lower MMD is better. The sampled graph panels are regenerated from the stored `data/predictions/graphons/tensors.pt` with seed 0, following the same graphon sampling idea used by GGME; the original generated samples are not persisted as pickle files in these runs.
Main figure: `ogad_good_bad_contact_sheet.png`.
| Type | Dataset | Run | n_clusters | Mean rank | degree MMD | clustering MMD | spectrum MMD | Why useful |
|---|---|---|---:|---:|---:|---:|---:|---|
| good | sbm_n100_4c | `20260731_055853_sbm_n100_4c_nc10` | 8 | 1 | 0.01636 | 0.2031 | 4.035e-07 | Best SBM run: all three OGAD run ranks are 1; clustering MMD is especially low. |
| bad | sbm_n100_4c | `20260730_072137_sbm_n100_4c_nc6` | 1 | 5.67 | 0.278 | 54.01 | 3.232e-06 | Worst SBM run: large degradation in degree and spectral MMD relative to the best run. |
| good | expdeg_n100_lin | `20260801_171551_expdeg_n100_lin_nc17` | 8 | 1.67 | 0.02228 | 20.87 | 2.591e-05 | Best ExpDeg run: strong degree fit and best clustering among OGAD runs. |
| bad | expdeg_n100_lin | `20260731_135854_expdeg_n100_lin_nc12` | 1 | 5.67 | 0.5041 | 1494 | 4.683e-05 | Worst ExpDeg run: graphon collapses toward an almost uniform dense surface; clustering MMD explodes. |
| good | er_n100_p005 | `20260730_045033_er_n100_p005_nc5` | 8 | 2 | 0.1257 | 171.4 | 5.394e-06 | Balanced ER run: ranks 2/2/2 across degree, clustering, and spectrum. |
| bad | er_n100_p005 | `20260729_164206_er_n100_p005_nc0` | 1 | 6 | 0.7651 | 6272 | 5.963e-05 | Worst ER run: all three descriptor ranks are last among ER OGAD runs. |

## Interpretation notes
- **good sbm_n100_4c / `20260731_055853_sbm_n100_4c_nc10`**: sampled graph density 0.031 vs target 0.032; clustering 0.037 vs target 0.048; graphon mean/std 0.031/0.038. Best SBM run: all three OGAD run ranks are 1; clustering MMD is especially low.
- **bad sbm_n100_4c / `20260730_072137_sbm_n100_4c_nc6`**: sampled graph density 0.024 vs target 0.032; clustering 0.055 vs target 0.048; graphon mean/std 0.024/0.044. Worst SBM run: large degradation in degree and spectral MMD relative to the best run.
- **good expdeg_n100_lin / `20260801_171551_expdeg_n100_lin_nc17`**: sampled graph density 0.097 vs target 0.105; clustering 0.166 vs target 0.160; graphon mean/std 0.099/0.091. Best ExpDeg run: strong degree fit and best clustering among OGAD runs.
- **bad expdeg_n100_lin / `20260731_135854_expdeg_n100_lin_nc12`**: sampled graph density 0.110 vs target 0.105; clustering 0.110 vs target 0.160; graphon mean/std 0.110/0.010. Worst ExpDeg run: graphon collapses toward an almost uniform dense surface; clustering MMD explodes.
- **good er_n100_p005 / `20260730_045033_er_n100_p005_nc5`**: sampled graph density 0.049 vs target 0.049; clustering 0.057 vs target 0.048; graphon mean/std 0.051/0.037. Balanced ER run: ranks 2/2/2 across degree, clustering, and spectrum.
- **bad er_n100_p005 / `20260729_164206_er_n100_p005_nc0`**: sampled graph density 0.077 vs target 0.049; clustering 0.076 vs target 0.048; graphon mean/std 0.076/0.007. Worst ER run: all three descriptor ranks are last among ER OGAD runs.
