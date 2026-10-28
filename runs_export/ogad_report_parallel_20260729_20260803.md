# OGAD best run report

Batch analizzati: `parallel_20260729_144516`, `parallel_20260803_063318`. MMD piu basso = migliore.

Run complete con tre righe `stage=ggme`: **22**. Run incomplete/senza metriche: **12**.

Criterio usato: per ogni dataset ho ordinato le run OGAD per rango medio sui tre MMD; a parita ho scelto la run con miglior rango medio di OGAD rispetto agli altri modelli nella stessa valutazione. I rank dei modelli usano pari merito quando due MMD coincidono, frequente per `Paper` e `DC-SBM`.

## Migliori run OGAD per dataset

| Dataset | Best run | Batch | n_clusters | OGAD rank medio tra run | OGAD rank medio vs modelli | MMD degree | MMD clustering | MMD spectrum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ba_n100_m3 | `20260802_011527_ba_n100_m3_nc19` | `parallel_20260729_144516` | 1 | 1.000 | 6.000 | 1.0550 | 1177.3255 | 3.543e-05 |
| er_n100_p005 | `20260730_045033_er_n100_p005_nc5` | `parallel_20260729_144516` | 8 | 2.000 | 4.667 | 0.1257 | 171.3671 | 5.394e-06 |
| expdeg_n100_lin | `20260801_171551_expdeg_n100_lin_nc17` | `parallel_20260729_144516` | 8 | 1.667 | 3.000 | 0.0223 | 20.8710 | 2.591e-05 |
| sbm_n100_4c | `20260731_055853_sbm_n100_4c_nc10` | `parallel_20260729_144516` | 8 | 1.000 | 3.000 | 0.0164 | 0.2031 | 4.035e-07 |
| ws_n100_k8_p01 | `20260803_064123_ws_n100_k8_p01_nc24` | `parallel_20260729_144516` | 1 | 1.000 | 4.667 | 0.3333 | 5239.3725 | 1.845e-04 |

## Confronto sintetico contro gli altri modelli

### ba_n100_m3
Best run: `20260802_011527_ba_n100_m3_nc19` (`parallel_20260729_144516`, n_clusters=1).

| Descriptor | Winner | OGAD MMD | OGAD rank / modelli |
|---|---|---:|---:|
| degree | Configuration | 1.0550 | 5/7 |
| clustering | DiGress | 1177.3255 | 6/7 |
| spectrum | DiGress | 3.543e-05 | 7/7 |

| Descriptor | Modello | MMD | Rank |
|---|---|---:|---:|
| degree | Configuration | 0.0084 | 1 |
| degree | DiGress | 0.3254 | 2 |
| degree | Paper | 0.8827 | 3 |
| degree | DC-SBM | 0.8827 | 3 |
| degree | OGAD | 1.0550 | 5 |
| degree | SBM | 1.1357 | 6 |
| degree | ER | 1.1371 | 7 |
| clustering | DiGress | 81.2121 | 1 |
| clustering | Paper | 381.5987 | 2 |
| clustering | DC-SBM | 381.5987 | 2 |
| clustering | Configuration | 900.1712 | 4 |
| clustering | SBM | 968.0170 | 5 |
| clustering | OGAD | 1177.3255 | 6 |
| clustering | ER | 1821.8838 | 7 |
| spectrum | DiGress | 3.741e-06 | 1 |
| spectrum | Configuration | 1.302e-05 | 2 |
| spectrum | ER | 1.337e-05 | 3 |
| spectrum | SBM | 1.524e-05 | 4 |
| spectrum | Paper | 1.777e-05 | 5 |
| spectrum | DC-SBM | 1.777e-05 | 5 |
| spectrum | OGAD | 3.543e-05 | 7 |

### er_n100_p005
Best run: `20260730_045033_er_n100_p005_nc5` (`parallel_20260729_144516`, n_clusters=8).

| Descriptor | Winner | OGAD MMD | OGAD rank / modelli |
|---|---|---:|---:|
| degree | ER | 0.1257 | 5/7 |
| clustering | ER | 171.3671 | 4/7 |
| spectrum | Configuration | 5.394e-06 | 5/7 |

| Descriptor | Modello | MMD | Rank |
|---|---|---:|---:|
| degree | ER | 0.0012 | 1 |
| degree | Configuration | 0.0013 | 2 |
| degree | SBM | 0.0041 | 3 |
| degree | DiGress | 0.0134 | 4 |
| degree | OGAD | 0.1257 | 5 |
| degree | Paper | 0.1611 | 6 |
| degree | DC-SBM | 0.1611 | 6 |
| clustering | ER | -4.5622 | 1 |
| clustering | Configuration | 7.3729 | 2 |
| clustering | DiGress | 21.5956 | 3 |
| clustering | OGAD | 171.3671 | 4 |
| clustering | SBM | 927.7251 | 5 |
| clustering | Paper | 1291.4952 | 6 |
| clustering | DC-SBM | 1291.4952 | 6 |
| spectrum | Configuration | -6.987e-07 | 1 |
| spectrum | DiGress | 5.292e-07 | 2 |
| spectrum | ER | 5.754e-07 | 3 |
| spectrum | SBM | 2.563e-06 | 4 |
| spectrum | OGAD | 5.394e-06 | 5 |
| spectrum | Paper | 1.178e-05 | 6 |
| spectrum | DC-SBM | 1.178e-05 | 6 |

### expdeg_n100_lin
Best run: `20260801_171551_expdeg_n100_lin_nc17` (`parallel_20260729_144516`, n_clusters=8).

| Descriptor | Winner | OGAD MMD | OGAD rank / modelli |
|---|---|---:|---:|
| degree | Paper/DC-SBM | 0.0223 | 4/7 |
| clustering | OGAD | 20.8710 | 1/7 |
| spectrum | DiGress | 2.591e-05 | 4/7 |

| Descriptor | Modello | MMD | Rank |
|---|---|---:|---:|
| degree | Paper | 0.0128 | 1 |
| degree | DC-SBM | 0.0128 | 1 |
| degree | DiGress | 0.0209 | 3 |
| degree | OGAD | 0.0223 | 4 |
| degree | Configuration | 0.0223 | 5 |
| degree | SBM | 0.3552 | 6 |
| degree | ER | 0.5193 | 7 |
| clustering | OGAD | 20.8710 | 1 |
| clustering | DiGress | 58.5847 | 2 |
| clustering | Configuration | 87.0578 | 3 |
| clustering | Paper | 127.2859 | 4 |
| clustering | DC-SBM | 127.2859 | 4 |
| clustering | SBM | 736.2066 | 6 |
| clustering | ER | 1397.2955 | 7 |
| spectrum | DiGress | 6.623e-06 | 1 |
| spectrum | SBM | 1.326e-05 | 2 |
| spectrum | Configuration | 1.614e-05 | 3 |
| spectrum | OGAD | 2.591e-05 | 4 |
| spectrum | ER | 5.101e-05 | 5 |
| spectrum | Paper | 5.929e-05 | 6 |
| spectrum | DC-SBM | 5.929e-05 | 6 |

### sbm_n100_4c
Best run: `20260731_055853_sbm_n100_4c_nc10` (`parallel_20260729_144516`, n_clusters=8).

| Descriptor | Winner | OGAD MMD | OGAD rank / modelli |
|---|---|---:|---:|
| degree | DiGress | 0.0164 | 5/7 |
| clustering | OGAD | 0.2031 | 1/7 |
| spectrum | ER | 4.035e-07 | 3/7 |

| Descriptor | Modello | MMD | Rank |
|---|---|---:|---:|
| degree | DiGress | 0.0022 | 1 |
| degree | Configuration | 0.0037 | 2 |
| degree | ER | 0.0038 | 3 |
| degree | SBM | 0.0147 | 4 |
| degree | OGAD | 0.0164 | 5 |
| degree | Paper | 0.2171 | 6 |
| degree | DC-SBM | 0.2171 | 6 |
| clustering | OGAD | 0.2031 | 1 |
| clustering | DiGress | 190.0348 | 2 |
| clustering | ER | 430.7631 | 3 |
| clustering | Configuration | 614.5343 | 4 |
| clustering | SBM | 695.4746 | 5 |
| clustering | Paper | 1394.3205 | 6 |
| clustering | DC-SBM | 1394.3205 | 6 |
| spectrum | ER | -1.774e-07 | 1 |
| spectrum | Configuration | 1.609e-08 | 2 |
| spectrum | OGAD | 4.035e-07 | 3 |
| spectrum | DiGress | 1.005e-06 | 4 |
| spectrum | SBM | 1.962e-06 | 5 |
| spectrum | Paper | 3.628e-06 | 6 |
| spectrum | DC-SBM | 3.628e-06 | 6 |

### ws_n100_k8_p01
Best run: `20260803_064123_ws_n100_k8_p01_nc24` (`parallel_20260729_144516`, n_clusters=1).

| Descriptor | Winner | OGAD MMD | OGAD rank / modelli |
|---|---|---:|---:|
| degree | Configuration | 0.3333 | 6/7 |
| clustering | DiGress | 5239.3725 | 6/7 |
| spectrum | DiGress | 1.845e-04 | 2/7 |

| Descriptor | Modello | MMD | Rank |
|---|---|---:|---:|
| degree | Configuration | 0.0408 | 1 |
| degree | SBM | 0.2970 | 2 |
| degree | Paper | 0.3143 | 3 |
| degree | DC-SBM | 0.3143 | 3 |
| degree | ER | 0.3227 | 5 |
| degree | OGAD | 0.3333 | 6 |
| degree | DiGress | 0.3411 | 7 |
| clustering | DiGress | 1505.2381 | 1 |
| clustering | Paper | 2758.5512 | 2 |
| clustering | DC-SBM | 2758.5512 | 2 |
| clustering | SBM | 2783.3540 | 4 |
| clustering | ER | 4532.9251 | 5 |
| clustering | OGAD | 5239.3725 | 6 |
| clustering | Configuration | 6003.3401 | 7 |
| spectrum | DiGress | 9.620e-05 | 1 |
| spectrum | OGAD | 1.845e-04 | 2 |
| spectrum | Paper | 1.862e-04 | 3 |
| spectrum | DC-SBM | 1.862e-04 | 3 |
| spectrum | Configuration | 1.880e-04 | 5 |
| spectrum | ER | 1.885e-04 | 6 |
| spectrum | SBM | 1.914e-04 | 7 |

## Run incomplete o ancora senza metriche finali

| Batch | Run | Dataset | n_clusters | Status | Motivo |
|---|---|---|---:|---|---|
| `parallel_20260729_144516` | `20260801_001426_expdeg_n100_lin_nc15` | expdeg_n100_lin | 4 | complete | only 0 ggme rows |
| `parallel_20260729_144516` | `20260801_075813_expdeg_n100_lin_nc16` | expdeg_n100_lin | 8 | complete | missing metrics.jsonl |
| `parallel_20260729_144516` | `20260801_113746_expdeg_n100_lin_nc17` | expdeg_n100_lin | 8 | complete | missing metrics.jsonl |
| `parallel_20260729_144516` | `20260801_121326_ba_n100_m3_nc18` | ba_n100_m3 | 1 | complete | only 0 ggme rows |
| `parallel_20260729_144516` | `20260802_153533_ba_n100_m3_nc21` | ba_n100_m3 | 4 | running | only 0 ggme rows |
| `parallel_20260729_144516` | `20260802_200413_ba_n100_m3_nc22` | ba_n100_m3 | 8 | running | only 0 ggme rows |
| `parallel_20260729_144516` | `20260803_053901_ba_n100_m3_nc23` | ba_n100_m3 | 8 | running | missing metrics.jsonl |
| `parallel_20260729_144516` | `20260803_150028_ws_n100_k8_p01_nc25` | ws_n100_k8_p01 | 1 | running | missing metrics.jsonl |
| `parallel_20260803_063318` | `20260803_074421_ba_n100_m3_nc3` | ba_n100_m3 | 8 | running | missing metrics.jsonl |
| `parallel_20260803_063318` | `20260803_074421_er_n100_p005_nc0` | er_n100_p005 | 8 | running | missing metrics.jsonl |
| `parallel_20260803_063318` | `20260803_074421_er_n100_p005_nc1` | er_n100_p005 | 16 | running | missing metrics.jsonl |
| `parallel_20260803_063318` | `20260803_074421_er_n100_p005_nc2` | er_n100_p005 | 32 | running | missing metrics.jsonl |

File CSV generati: `ogad_best_runs_model_comparison.csv`, `ogad_completed_run_ranking.csv`, `ogad_incomplete_runs_parallel_20260729_20260803.csv`.
