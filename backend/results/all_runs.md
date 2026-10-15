| Run | Encoder | Loss | Packets | Params | Acc % | Macro-F1 % | Unknown AUROC | 10-shot new % | Train s |
|---|---|---|---|---|---|---|---|---|---|
| cnn_ce_p30_s0 | cnn | ce | 30 | 141,312 | 92.09 | 87.75 | 0.904 | nan | 485 |
| cnn_supcon_p30_s0 | cnn | supcon | 30 | 141,312 | 91.29 | 86.07 | 0.907 | nan | 362 |
| gbdt | gbdt | - | 30 | 0 | 91.24 | 87.17 | nan | nan | 111 |
| gbdt_ppi_derived_p30_s0 | gbdt | - | 30 | 0 | 86.96 | 80.53 | nan | nan | 63 |
| gbdt_ppi_derived_p30_s1 | gbdt | - | 30 | 0 | 89.26 | 83.37 | nan | nan | 52 |
| gbdt_ppi_derived_p30_s2 | gbdt | - | 30 | 0 | 89.66 | 83.83 | nan | nan | 58 |
| gbdt_ppi_p10_s0 | gbdt | - | 10 | 0 | 79.78 | 71.19 | nan | nan | 37 |
| gbdt_ppi_p15_s0 | gbdt | - | 15 | 0 | 86.85 | 80.77 | nan | nan | 39 |
| gbdt_ppi_p20_s0 | gbdt | - | 20 | 0 | 86.92 | 81.01 | nan | nan | 51 |
| gbdt_ppi_p30_s0 | gbdt | - | 30 | 0 | 87.08 | 81.22 | nan | nan | 60 |
| gbdt_ppi_p30_s1 | gbdt | - | 30 | 0 | 88.08 | 82.90 | nan | nan | 72 |
| gbdt_ppi_p30_s2 | gbdt | - | 30 | 0 | 85.83 | 79.38 | nan | nan | 70 |
| gbdt_ppi_p3_s0 | gbdt | - | 3 | 0 | 69.21 | 52.28 | nan | nan | 19 |
| gbdt_ppi_p5_s0 | gbdt | - | 5 | 0 | 69.00 | 56.92 | nan | nan | 35 |
| gru_ce_p30_s0 | gru | ce | 30 | 431,488 | 92.62 | 88.52 | 0.884 | nan | 555 |
| gru_supcon_p30_s0 | gru | supcon | 30 | 431,488 | 92.46 | 87.80 | 0.871 | nan | 544 |
| majority | majority | - | 30 | 0 | 12.74 | 1.13 | nan | nan | 0 |
| tcn_ce_p30_s0 | tcn | ce | 30 | 137,280 | 91.83 | 87.37 | nan | nan | 133 |
| tcn_ce_p30_s1 | tcn | ce | 30 | 137,280 | 91.73 | 87.24 | nan | nan | 135 |
| tcn_ce_p30_s2 | tcn | ce | 30 | 137,280 | 91.73 | 87.27 | nan | nan | 138 |
| tcn_supcon_p30 | tcn | supcon | 30 | 137,280 | 91.54 | 86.33 | 0.905 | 6.9 | 592 |
| tcn_supcon_p30_s0 | tcn | supcon | 30 | 137,280 | 91.54 | 86.28 | nan | nan | 251 |
| tcn_supcon_p30_s1 | tcn | supcon | 30 | 137,280 | 91.55 | 86.26 | nan | nan | 140 |
| tcn_supcon_p30_s2 | tcn | supcon | 30 | 137,280 | 91.60 | 86.50 | nan | nan | 136 |
| transformer_ce_p10_s0 | transformer | ce | 10 | 298,497 | 79.94 | 69.70 | nan | nan | 228 |
| transformer_ce_p15_s0 | transformer | ce | 15 | 298,497 | 87.63 | 81.57 | nan | nan | 228 |
| transformer_ce_p20_s0 | transformer | ce | 20 | 298,497 | 91.08 | 86.54 | nan | nan | 228 |
| transformer_ce_p30_s0 | transformer | ce | 30 | 298,497 | 92.82 | 89.05 | 0.951 | nan | 225 |
| transformer_ce_p30_s1 | transformer | ce | 30 | 298,497 | 92.85 | 88.85 | nan | nan | 421 |
| transformer_ce_p30_s2 | transformer | ce | 30 | 298,497 | 92.82 | 88.86 | nan | nan | 412 |
| transformer_ce_p3_s0 | transformer | ce | 3 | 298,497 | 64.72 | 49.42 | nan | nan | 325 |
| transformer_ce_p5_s0 | transformer | ce | 5 | 298,497 | 74.71 | 66.23 | nan | nan | 233 |
| transformer_supcon_p30_s0 | transformer | supcon | 30 | 298,497 | 91.97 | 87.02 | 0.910 | nan | 246 |

Notes: `gbdt` (no seed suffix) uses whole-flow statistics and is NOT a fair baseline; fair GBDT runs are `gbdt_ppi*`. `tcn_supcon_p30` is the original run, reproduced exactly by `tcn_supcon_p30_s0`. Unknown AUROC / few-shot columns are filled only for runs evaluated with evaluate.py.
