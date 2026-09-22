# Migliori risultati per dataset e modello

Dati riestratti dagli artefatti locali il 2026-09-08. Non sono valori inseriti a mano: per ogni cella e riportato il run/file sorgente nella sezione Fonti.

Criterio: per ogni dataset, modello e descrittore ho preso il minimo MMD osservato tra i run completati con tutte e tre le righe finali `stage=ggme`. MMD piu basso = migliore. `EDG` corrisponde a `expdeg_n100_lin`.

Run completati usati per dataset:

| Dataset | Nome config | Run completi con GGME |
|---|---|---:|
| ER | `er_n100_p005` | 7 |
| SBM | `sbm_n100_4c` | 6 |
| BA | `ba_n100_m3` | 6 |
| WS | `ws_n100_k8_p01` | 4 |
| EDG | `expdeg_n100_lin` | 6 |

## Tabella risultati

| Dataset | Modello | Degree | Clustering | Spectrum |
|---|---|---:|---:|---:|
| ER | OGAD | 0.080129 | 69.4112 | 4.288e-06 |
| ER | DiGress | 0.007058 | 21.5956 | **-1.101e-06** |
| ER | ER | **0.001205** | **-4.56225** | 5.754e-07 |
| ER | Configuration | 0.001266 | 7.37291 | -6.987e-07 |
| ER | SBM | 0.004082 | 927.725 | 2.563e-06 |
| ER | DC-SBM/Paper | 0.161066 | 1291.50 | 1.178e-05 |
| SBM | OGAD | 0.016365 | **0.203109** | 4.035e-07 |
| SBM | DiGress | **0.002235** | 190.035 | 1.005e-06 |
| SBM | ER | 0.003837 | 430.763 | **-1.774e-07** |
| SBM | Configuration | 0.00371 | 614.534 | 1.609e-08 |
| SBM | SBM | 0.014688 | 695.475 | 1.962e-06 |
| SBM | DC-SBM/Paper | 0.217106 | 1394.32 | 3.628e-06 |
| BA | OGAD | 0.980363 | 508.541 | 3.543e-05 |
| BA | DiGress | 0.204514 | **78.0936** | **1.979e-06** |
| BA | ER | 1.1371 | 1821.88 | 1.337e-05 |
| BA | Configuration | **0.008423** | 900.171 | 1.302e-05 |
| BA | SBM | 1.13568 | 968.017 | 1.524e-05 |
| BA | DC-SBM/Paper | 0.882699 | 381.599 | 1.777e-05 |
| WS | OGAD | 0.333326 | 2221.89 | 0.000129 |
| WS | DiGress | 0.290017 | **1505.24** | **9.620e-05** |
| WS | ER | 0.322729 | 4532.93 | 0.000189 |
| WS | Configuration | **0.04079** | 6003.34 | 0.000188 |
| WS | SBM | 0.297014 | 2783.35 | 0.000191 |
| WS | DC-SBM/Paper | 0.314288 | 2758.55 | 0.000186 |
| EDG | OGAD | 0.022283 | **20.871** | 1.696e-05 |
| EDG | DiGress | 0.020852 | 58.5847 | **6.623e-06** |
| EDG | ER | 0.519254 | 1397.30 | 5.101e-05 |
| EDG | Configuration | 0.022298 | 87.0578 | 1.614e-05 |
| EDG | SBM | 0.355204 | 736.207 | 1.326e-05 |
| EDG | DC-SBM/Paper | **0.012818** | 127.286 | 5.929e-05 |

## Fonti per cella

Ogni riga indica da quale `metrics.jsonl` e da quale riga JSONL arriva il valore della tabella sopra. Le righe `DC-SBM/Paper` combinano due chiavi presenti nello stesso record `mmd`; nei run ispezionati hanno lo stesso valore.

| Dataset | Modello | Descrittore | MMD | Run | Fonte |
|---|---|---|---:|---|---|
| ER | OGAD | `degree_distribution` | 0.080129 | `20260901_173519_er_n100_p005_nc0` | `OGAD/runs/parallel_20260901_171156/runs/20260901_173519_er_n100_p005_nc0/metrics.jsonl:28` |
| ER | OGAD | `clustering_coefficient` | 69.4112 | `20260729_164206_er_n100_p005_nc3` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc3/metrics.jsonl:29` |
| ER | OGAD | `normalised_laplacian_spectrum` | 4.288e-06 | `20260730_002832_er_n100_p005_nc4` | `OGAD/runs/parallel_20260729_144516/runs/20260730_002832_er_n100_p005_nc4/metrics.jsonl:30` |
| ER | DiGress | `degree_distribution` | 0.007058 | `20260901_173519_er_n100_p005_nc0` | `OGAD/runs/parallel_20260901_171156/runs/20260901_173519_er_n100_p005_nc0/metrics.jsonl:28` |
| ER | DiGress | `clustering_coefficient` | 21.5956 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:29` |
| ER | DiGress | `normalised_laplacian_spectrum` | -1.101e-06 | `20260901_173519_er_n100_p005_nc0` | `OGAD/runs/parallel_20260901_171156/runs/20260901_173519_er_n100_p005_nc0/metrics.jsonl:30` |
| ER | ER | `degree_distribution` | 0.001205 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:28` |
| ER | ER | `clustering_coefficient` | -4.56225 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:29` |
| ER | ER | `normalised_laplacian_spectrum` | 5.754e-07 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:30` |
| ER | Configuration | `degree_distribution` | 0.001266 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:28` |
| ER | Configuration | `clustering_coefficient` | 7.37291 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:29` |
| ER | Configuration | `normalised_laplacian_spectrum` | -6.987e-07 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:30` |
| ER | SBM | `degree_distribution` | 0.004082 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:28` |
| ER | SBM | `clustering_coefficient` | 927.725 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:29` |
| ER | SBM | `normalised_laplacian_spectrum` | 2.563e-06 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:30` |
| ER | DC-SBM/Paper | `degree_distribution` | 0.161066 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:28` |
| ER | DC-SBM/Paper | `clustering_coefficient` | 1291.50 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:29` |
| ER | DC-SBM/Paper | `normalised_laplacian_spectrum` | 1.178e-05 | `20260729_164206_er_n100_p005_nc0` | `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl:30` |
| SBM | OGAD | `degree_distribution` | 0.016365 | `20260731_055853_sbm_n100_4c_nc10` | `OGAD/runs/parallel_20260729_144516/runs/20260731_055853_sbm_n100_4c_nc10/metrics.jsonl:28` |
| SBM | OGAD | `clustering_coefficient` | 0.203109 | `20260731_055853_sbm_n100_4c_nc10` | `OGAD/runs/parallel_20260729_144516/runs/20260731_055853_sbm_n100_4c_nc10/metrics.jsonl:29` |
| SBM | OGAD | `normalised_laplacian_spectrum` | 4.035e-07 | `20260731_055853_sbm_n100_4c_nc10` | `OGAD/runs/parallel_20260729_144516/runs/20260731_055853_sbm_n100_4c_nc10/metrics.jsonl:30` |
| SBM | DiGress | `degree_distribution` | 0.002235 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:28` |
| SBM | DiGress | `clustering_coefficient` | 190.035 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:29` |
| SBM | DiGress | `normalised_laplacian_spectrum` | 1.005e-06 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:30` |
| SBM | ER | `degree_distribution` | 0.003837 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:28` |
| SBM | ER | `clustering_coefficient` | 430.763 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:29` |
| SBM | ER | `normalised_laplacian_spectrum` | -1.774e-07 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:30` |
| SBM | Configuration | `degree_distribution` | 0.00371 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:28` |
| SBM | Configuration | `clustering_coefficient` | 614.534 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:29` |
| SBM | Configuration | `normalised_laplacian_spectrum` | 1.609e-08 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:30` |
| SBM | SBM | `degree_distribution` | 0.014688 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:28` |
| SBM | SBM | `clustering_coefficient` | 695.475 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:29` |
| SBM | SBM | `normalised_laplacian_spectrum` | 1.962e-06 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:30` |
| SBM | DC-SBM/Paper | `degree_distribution` | 0.217106 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:28` |
| SBM | DC-SBM/Paper | `clustering_coefficient` | 1394.32 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:29` |
| SBM | DC-SBM/Paper | `normalised_laplacian_spectrum` | 3.628e-06 | `20260730_072137_sbm_n100_4c_nc6` | `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl:30` |
| BA | OGAD | `degree_distribution` | 0.980363 | `20260803_162715_ba_n100_m3_nc1` | `OGAD/runs/parallel_20260803_154001/runs/20260803_162715_ba_n100_m3_nc1/metrics.jsonl:16` |
| BA | OGAD | `clustering_coefficient` | 508.541 | `20260804_100803_ba_n100_m3_nc2` | `OGAD/runs/parallel_20260803_154001/runs/20260804_100803_ba_n100_m3_nc2/metrics.jsonl:17` |
| BA | OGAD | `normalised_laplacian_spectrum` | 3.543e-05 | `20260802_011527_ba_n100_m3_nc19` | `OGAD/runs/parallel_20260729_144516/runs/20260802_011527_ba_n100_m3_nc19/metrics.jsonl:30` |
| BA | DiGress | `degree_distribution` | 0.204514 | `20260803_162715_ba_n100_m3_nc0` | `OGAD/runs/parallel_20260803_154001/runs/20260803_162715_ba_n100_m3_nc0/metrics.jsonl:16` |
| BA | DiGress | `clustering_coefficient` | 78.0936 | `20260803_162715_ba_n100_m3_nc0` | `OGAD/runs/parallel_20260803_154001/runs/20260803_162715_ba_n100_m3_nc0/metrics.jsonl:17` |
| BA | DiGress | `normalised_laplacian_spectrum` | 1.979e-06 | `20260803_162715_ba_n100_m3_nc0` | `OGAD/runs/parallel_20260803_154001/runs/20260803_162715_ba_n100_m3_nc0/metrics.jsonl:18` |
| BA | ER | `degree_distribution` | 1.1371 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:28` |
| BA | ER | `clustering_coefficient` | 1821.88 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:29` |
| BA | ER | `normalised_laplacian_spectrum` | 1.337e-05 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:30` |
| BA | Configuration | `degree_distribution` | 0.008423 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:28` |
| BA | Configuration | `clustering_coefficient` | 900.171 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:29` |
| BA | Configuration | `normalised_laplacian_spectrum` | 1.302e-05 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:30` |
| BA | SBM | `degree_distribution` | 1.13568 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:28` |
| BA | SBM | `clustering_coefficient` | 968.017 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:29` |
| BA | SBM | `normalised_laplacian_spectrum` | 1.524e-05 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:30` |
| BA | DC-SBM/Paper | `degree_distribution` | 0.882699 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:28` |
| BA | DC-SBM/Paper | `clustering_coefficient` | 381.599 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:29` |
| BA | DC-SBM/Paper | `normalised_laplacian_spectrum` | 1.777e-05 | `20260801_171552_ba_n100_m3_nc18` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl:30` |
| WS | OGAD | `degree_distribution` | 0.333326 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:28` |
| WS | OGAD | `clustering_coefficient` | 2221.89 | `20260804_150351_ws_n100_k8_p01_nc3` | `OGAD/runs/parallel_20260803_154001/runs/20260804_150351_ws_n100_k8_p01_nc3/metrics.jsonl:17` |
| WS | OGAD | `normalised_laplacian_spectrum` | 0.000129 | `20260804_150351_ws_n100_k8_p01_nc3` | `OGAD/runs/parallel_20260803_154001/runs/20260804_150351_ws_n100_k8_p01_nc3/metrics.jsonl:18` |
| WS | DiGress | `degree_distribution` | 0.290017 | `20260804_150351_ws_n100_k8_p01_nc3` | `OGAD/runs/parallel_20260803_154001/runs/20260804_150351_ws_n100_k8_p01_nc3/metrics.jsonl:16` |
| WS | DiGress | `clustering_coefficient` | 1505.24 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:29` |
| WS | DiGress | `normalised_laplacian_spectrum` | 9.620e-05 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:30` |
| WS | ER | `degree_distribution` | 0.322729 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:28` |
| WS | ER | `clustering_coefficient` | 4532.93 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:29` |
| WS | ER | `normalised_laplacian_spectrum` | 0.000189 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:30` |
| WS | Configuration | `degree_distribution` | 0.04079 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:28` |
| WS | Configuration | `clustering_coefficient` | 6003.34 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:29` |
| WS | Configuration | `normalised_laplacian_spectrum` | 0.000188 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:30` |
| WS | SBM | `degree_distribution` | 0.297014 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:28` |
| WS | SBM | `clustering_coefficient` | 2783.35 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:29` |
| WS | SBM | `normalised_laplacian_spectrum` | 0.000191 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:30` |
| WS | DC-SBM/Paper | `degree_distribution` | 0.314288 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:28` |
| WS | DC-SBM/Paper | `clustering_coefficient` | 2758.55 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:29` |
| WS | DC-SBM/Paper | `normalised_laplacian_spectrum` | 0.000186 | `20260803_064123_ws_n100_k8_p01_nc24` | `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl:30` |
| EDG | OGAD | `degree_distribution` | 0.022283 | `20260801_171551_expdeg_n100_lin_nc17` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171551_expdeg_n100_lin_nc17/metrics.jsonl:28` |
| EDG | OGAD | `clustering_coefficient` | 20.871 | `20260801_171551_expdeg_n100_lin_nc17` | `OGAD/runs/parallel_20260729_144516/runs/20260801_171551_expdeg_n100_lin_nc17/metrics.jsonl:29` |
| EDG | OGAD | `normalised_laplacian_spectrum` | 1.696e-05 | `20260731_213135_expdeg_n100_lin_nc14` | `OGAD/runs/parallel_20260729_144516/runs/20260731_213135_expdeg_n100_lin_nc14/metrics.jsonl:30` |
| EDG | DiGress | `degree_distribution` | 0.020852 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:28` |
| EDG | DiGress | `clustering_coefficient` | 58.5847 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:29` |
| EDG | DiGress | `normalised_laplacian_spectrum` | 6.623e-06 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:30` |
| EDG | ER | `degree_distribution` | 0.519254 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:28` |
| EDG | ER | `clustering_coefficient` | 1397.30 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:29` |
| EDG | ER | `normalised_laplacian_spectrum` | 5.101e-05 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:30` |
| EDG | Configuration | `degree_distribution` | 0.022298 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:28` |
| EDG | Configuration | `clustering_coefficient` | 87.0578 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:29` |
| EDG | Configuration | `normalised_laplacian_spectrum` | 1.614e-05 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:30` |
| EDG | SBM | `degree_distribution` | 0.355204 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:28` |
| EDG | SBM | `clustering_coefficient` | 736.207 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:29` |
| EDG | SBM | `normalised_laplacian_spectrum` | 1.326e-05 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:30` |
| EDG | DC-SBM/Paper | `degree_distribution` | 0.012818 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:28` |
| EDG | DC-SBM/Paper | `clustering_coefficient` | 127.286 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:29` |
| EDG | DC-SBM/Paper | `normalised_laplacian_spectrum` | 5.929e-05 | `20260731_135854_expdeg_n100_lin_nc12` | `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl:30` |

## Run inclusi

### ER (`er_n100_p005`)

- `20260729_164206_er_n100_p005_nc0`: `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc0/metrics.jsonl`
- `20260729_164206_er_n100_p005_nc1`: `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc1/metrics.jsonl`
- `20260729_164206_er_n100_p005_nc2`: `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc2/metrics.jsonl`
- `20260729_164206_er_n100_p005_nc3`: `OGAD/runs/parallel_20260729_144516/runs/20260729_164206_er_n100_p005_nc3/metrics.jsonl`
- `20260730_002832_er_n100_p005_nc4`: `OGAD/runs/parallel_20260729_144516/runs/20260730_002832_er_n100_p005_nc4/metrics.jsonl`
- `20260730_045033_er_n100_p005_nc5`: `OGAD/runs/parallel_20260729_144516/runs/20260730_045033_er_n100_p005_nc5/metrics.jsonl`
- `20260901_173519_er_n100_p005_nc0`: `OGAD/runs/parallel_20260901_171156/runs/20260901_173519_er_n100_p005_nc0/metrics.jsonl`

### SBM (`sbm_n100_4c`)

- `20260730_072137_sbm_n100_4c_nc6`: `OGAD/runs/parallel_20260729_144516/runs/20260730_072137_sbm_n100_4c_nc6/metrics.jsonl`
- `20260730_155509_sbm_n100_4c_nc7`: `OGAD/runs/parallel_20260729_144516/runs/20260730_155509_sbm_n100_4c_nc7/metrics.jsonl`
- `20260730_155545_sbm_n100_4c_nc8`: `OGAD/runs/parallel_20260729_144516/runs/20260730_155545_sbm_n100_4c_nc8/metrics.jsonl`
- `20260730_202546_sbm_n100_4c_nc9`: `OGAD/runs/parallel_20260729_144516/runs/20260730_202546_sbm_n100_4c_nc9/metrics.jsonl`
- `20260731_055853_sbm_n100_4c_nc10`: `OGAD/runs/parallel_20260729_144516/runs/20260731_055853_sbm_n100_4c_nc10/metrics.jsonl`
- `20260731_061406_sbm_n100_4c_nc11`: `OGAD/runs/parallel_20260729_144516/runs/20260731_061406_sbm_n100_4c_nc11/metrics.jsonl`

### BA (`ba_n100_m3`)

- `20260801_171552_ba_n100_m3_nc18`: `OGAD/runs/parallel_20260729_144516/runs/20260801_171552_ba_n100_m3_nc18/metrics.jsonl`
- `20260802_011527_ba_n100_m3_nc19`: `OGAD/runs/parallel_20260729_144516/runs/20260802_011527_ba_n100_m3_nc19/metrics.jsonl`
- `20260802_145225_ba_n100_m3_nc20`: `OGAD/runs/parallel_20260729_144516/runs/20260802_145225_ba_n100_m3_nc20/metrics.jsonl`
- `20260803_162715_ba_n100_m3_nc0`: `OGAD/runs/parallel_20260803_154001/runs/20260803_162715_ba_n100_m3_nc0/metrics.jsonl`
- `20260803_162715_ba_n100_m3_nc1`: `OGAD/runs/parallel_20260803_154001/runs/20260803_162715_ba_n100_m3_nc1/metrics.jsonl`
- `20260804_100803_ba_n100_m3_nc2`: `OGAD/runs/parallel_20260803_154001/runs/20260804_100803_ba_n100_m3_nc2/metrics.jsonl`

### WS (`ws_n100_k8_p01`)

- `20260803_064123_ws_n100_k8_p01_nc24`: `OGAD/runs/parallel_20260729_144516/runs/20260803_064123_ws_n100_k8_p01_nc24/metrics.jsonl`
- `20260804_150351_ws_n100_k8_p01_nc3`: `OGAD/runs/parallel_20260803_154001/runs/20260804_150351_ws_n100_k8_p01_nc3/metrics.jsonl`
- `20260805_104945_ws_n100_k8_p01_nc4`: `OGAD/runs/parallel_20260803_154001/runs/20260805_104945_ws_n100_k8_p01_nc4/metrics.jsonl`
- `20260805_185229_ws_n100_k8_p01_nc5`: `OGAD/runs/parallel_20260803_154001/runs/20260805_185229_ws_n100_k8_p01_nc5/metrics.jsonl`

### EDG (`expdeg_n100_lin`)

- `20260731_135854_expdeg_n100_lin_nc12`: `OGAD/runs/parallel_20260729_144516/runs/20260731_135854_expdeg_n100_lin_nc12/metrics.jsonl`
- `20260731_193803_expdeg_n100_lin_nc13`: `OGAD/runs/parallel_20260729_144516/runs/20260731_193803_expdeg_n100_lin_nc13/metrics.jsonl`
- `20260731_213135_expdeg_n100_lin_nc14`: `OGAD/runs/parallel_20260729_144516/runs/20260731_213135_expdeg_n100_lin_nc14/metrics.jsonl`
- `20260801_171550_expdeg_n100_lin_nc16`: `OGAD/runs/parallel_20260729_144516/runs/20260801_171550_expdeg_n100_lin_nc16/metrics.jsonl`
- `20260801_171551_expdeg_n100_lin_nc15`: `OGAD/runs/parallel_20260729_144516/runs/20260801_171551_expdeg_n100_lin_nc15/metrics.jsonl`
- `20260801_171551_expdeg_n100_lin_nc17`: `OGAD/runs/parallel_20260729_144516/runs/20260801_171551_expdeg_n100_lin_nc17/metrics.jsonl`
