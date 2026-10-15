# Model selection (validation split only)

Criterion: mean **validation** macro-F1 of the k-NN decision layer, over available seeds. Validation = 30,000 week-44 flows held out from training. Test columns are for reference only.

| Rank | Config | Seeds | Params | Val acc % | Val macro-F1 % | Test acc % | Test macro-F1 % |
|---|---|---|---|---|---|---|---|
| 1 | transformer + ce | 3 | 298,497 | 92.81 ± 0.10 | 89.59 ± 0.15 | 92.83 ± 0.02 | 88.92 ± 0.11 |
| 2 | gru + ce | 1 | 431,488 | 92.27 | 88.68 | 92.62 | 88.52 |
| 3 | gru + supcon | 1 | 431,488 | 92.14 | 88.04 | 92.46 | 87.80 |
| 4 | cnn + ce | 1 | 141,312 | 91.76 | 88.03 | 92.09 | 87.75 |
| 5 | tcn + ce | 3 | 137,280 | 91.55 ± 0.08 | 87.75 ± 0.19 | 91.76 ± 0.06 | 87.29 ± 0.07 |
| 6 | transformer + supcon | 1 | 298,497 | 91.57 | 87.26 | 91.97 | 87.02 |
| 7 | tcn + supcon | 3 | 137,280 | 91.24 ± 0.06 | 86.68 ± 0.13 | 91.56 ± 0.03 | 86.34 ± 0.13 |
| 8 | cnn + supcon | 1 | 141,312 | 90.94 | 86.25 | 91.29 | 86.07 |

**Selected: transformer + ce**
