# E4: accuracy vs number of packets

Known test flows, seed 0. Truncation to N packets is applied to train, validation and test.

| Packets | Flows with ≥ N pkts | Transformer + CE (k-NN) acc / macro-F1 | GBDT, fair (packet sequence only) acc / macro-F1 |
|---|---|---|---|
| 3 | 98% | 64.72 / 49.42 | 69.21 / 52.28 |
| 5 | 98% | 74.71 / 66.23 | 69.00 / 56.92 |
| 10 | 96% | 79.94 / 69.70 | 79.78 / 71.19 |
| 15 | 86% | 87.63 / 81.57 | 86.85 / 80.77 |
| 20 | 68% | 91.08 / 86.54 | 86.92 / 81.01 |
| 30 | 44% | 92.82 / 89.05 | 87.08 / 81.22 |
