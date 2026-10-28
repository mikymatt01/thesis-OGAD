# GGME metrics for completed datasets

Included datasets are those where every queued OGAD run across the selected parallel directories is complete and has all three final GGME descriptors.

Eligible datasets: **2** / 5.

## Best OGAD Runs

| Dataset | Best OGAD run | Batch | n_clusters | OGAD run mean rank | OGAD model mean rank |
|---|---|---|---:|---:|---:|
| expdeg_n100_lin | `20260801_171551_expdeg_n100_lin_nc17` | `parallel_20260729_144516` | 8 | 1.667 | 3.000 |
| sbm_n100_4c | `20260731_055853_sbm_n100_4c_nc10` | `parallel_20260729_144516` | 8 | 1.000 | 3.000 |

## expdeg_n100_lin

Best OGAD run: `20260801_171551_expdeg_n100_lin_nc17` (`parallel_20260729_144516`, n_clusters=8).

| Descriptor | Model | MMD | Rank |
|---|---|---:|---:|
| degree_distribution | Paper | 0.0128 | 1 |
| degree_distribution | DC-SBM | 0.0128 | 1 |
| degree_distribution | DiGress | 0.0209 | 3 |
| degree_distribution | OGAD | 0.0223 | 4 |
| degree_distribution | Configuration | 0.0223 | 5 |
| degree_distribution | SBM | 0.3552 | 6 |
| degree_distribution | ER | 0.5193 | 7 |
| clustering_coefficient | OGAD | 20.8710 | 1 |
| clustering_coefficient | DiGress | 58.5847 | 2 |
| clustering_coefficient | Configuration | 87.0578 | 3 |
| clustering_coefficient | Paper | 127.2859 | 4 |
| clustering_coefficient | DC-SBM | 127.2859 | 4 |
| clustering_coefficient | SBM | 736.2066 | 6 |
| clustering_coefficient | ER | 1397.2955 | 7 |
| normalised_laplacian_spectrum | DiGress | 6.623e-06 | 1 |
| normalised_laplacian_spectrum | SBM | 1.326e-05 | 2 |
| normalised_laplacian_spectrum | Configuration | 1.614e-05 | 3 |
| normalised_laplacian_spectrum | OGAD | 2.591e-05 | 4 |
| normalised_laplacian_spectrum | ER | 5.101e-05 | 5 |
| normalised_laplacian_spectrum | Paper | 5.929e-05 | 6 |
| normalised_laplacian_spectrum | DC-SBM | 5.929e-05 | 6 |

## sbm_n100_4c

Best OGAD run: `20260731_055853_sbm_n100_4c_nc10` (`parallel_20260729_144516`, n_clusters=8).

| Descriptor | Model | MMD | Rank |
|---|---|---:|---:|
| degree_distribution | DiGress | 0.0022 | 1 |
| degree_distribution | Configuration | 0.0037 | 2 |
| degree_distribution | ER | 0.0038 | 3 |
| degree_distribution | SBM | 0.0147 | 4 |
| degree_distribution | OGAD | 0.0164 | 5 |
| degree_distribution | Paper | 0.2171 | 6 |
| degree_distribution | DC-SBM | 0.2171 | 6 |
| clustering_coefficient | OGAD | 0.2031 | 1 |
| clustering_coefficient | DiGress | 190.0348 | 2 |
| clustering_coefficient | ER | 430.7631 | 3 |
| clustering_coefficient | Configuration | 614.5343 | 4 |
| clustering_coefficient | SBM | 695.4746 | 5 |
| clustering_coefficient | Paper | 1394.3205 | 6 |
| clustering_coefficient | DC-SBM | 1394.3205 | 6 |
| normalised_laplacian_spectrum | ER | -1.774e-07 | 1 |
| normalised_laplacian_spectrum | Configuration | 1.609e-08 | 2 |
| normalised_laplacian_spectrum | OGAD | 4.035e-07 | 3 |
| normalised_laplacian_spectrum | DiGress | 1.005e-06 | 4 |
| normalised_laplacian_spectrum | SBM | 1.962e-06 | 5 |
| normalised_laplacian_spectrum | Paper | 3.628e-06 | 6 |
| normalised_laplacian_spectrum | DC-SBM | 3.628e-06 | 6 |

## Excluded Datasets

| Dataset | Reason |
|---|---|
| ba_n100_m3 | status is not complete |
| er_n100_p005 | status is not complete |
| ws_n100_k8_p01 | status is not complete |

CSV metrics: `ggme_completed_datasets_best_ogad.csv`
CSV filter status: `ggme_completed_dataset_filter_status.csv`
