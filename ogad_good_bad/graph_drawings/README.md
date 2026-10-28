# OGAD graph drawing examples
These figures keep the original matrix/graphon outputs untouched and add node-link visualizations for the same selected good/bad runs. Blue nodes are in the largest connected component; gray nodes are outside it. Node size is proportional to degree. Sampled OGAD graphs are regenerated from the stored graphons with seed 0.

Main figure: `ogad_good_bad_graph_drawings_contact_sheet.png`.

| Type | Dataset | Run | n_clusters | Mean rank | degree MMD | clustering MMD | spectrum MMD |
|---|---|---|---:|---:|---:|---:|---:|
| good | sbm_n100_4c | `20260731_055853_sbm_n100_4c_nc10` | 8 | 1 | 0.01636 | 0.2031 | 4.035e-07 |
| bad | sbm_n100_4c | `20260730_072137_sbm_n100_4c_nc6` | 1 | 5.67 | 0.278 | 54.01 | 3.232e-06 |
| good | expdeg_n100_lin | `20260801_171551_expdeg_n100_lin_nc17` | 8 | 1.67 | 0.02228 | 20.87 | 2.591e-05 |
| bad | expdeg_n100_lin | `20260731_135854_expdeg_n100_lin_nc12` | 1 | 5.67 | 0.5041 | 1494 | 4.683e-05 |
| good | er_n100_p005 | `20260730_045033_er_n100_p005_nc5` | 8 | 2 | 0.1257 | 171.4 | 5.394e-06 |
| bad | er_n100_p005 | `20260729_164206_er_n100_p005_nc0` | 1 | 6 | 0.7651 | 6272 | 5.963e-05 |
