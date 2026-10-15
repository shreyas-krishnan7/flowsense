# Seed study: TCN SupCon vs cross-entropy, Transformer cross-entropy, and fair GBDT

Known test flows: 150,000 (CESNET-QUIC22 weeks 45-47; trained on week 44). Mean ± sample std over seeds; all numbers recomputed from saved predictions.

| Model | Seeds | Accuracy % | Macro-F1 % |
|---|---|---|---|
| TCN + SupCon (k-NN) | 3 | 91.56 ± 0.03 | 86.34 ± 0.13 |
| TCN + cross-entropy (k-NN) | 3 | 91.76 ± 0.06 | 87.29 ± 0.07 |
| TCN + cross-entropy (softmax) | 3 | 91.31 ± 0.01 | 86.48 ± 0.01 |
| Transformer + cross-entropy (k-NN) | 3 | 92.83 ± 0.02 | 88.92 ± 0.11 |
| Transformer + cross-entropy (softmax) | 3 | 92.56 ± 0.05 | 88.32 ± 0.30 |
| GBDT, ppi (90 values, fair) | 3 | 87.00 ± 1.12 | 81.17 ± 1.76 |
| GBDT, ppi_derived (fair) | 3 | 88.63 ± 1.45 | 82.58 ± 1.79 |

**SupCon − cross-entropy (k-NN, paired by seed):** accuracy: -0.20 ± 0.08 pts (per seed -0.29, -0.18, -0.13), macro_f1: -0.95 ± 0.16 pts (per seed -1.09, -0.98, -0.77)

## Per-class F1 (mean ± std)

| Service | TCN + SupCon (k-NN) | TCN + cross-entropy (k-NN) | TCN + cross-entropy (softmax) | Transformer + cross-entropy (k-NN) | Transformer + cross-entropy (softmax) | GBDT, ppi (90 values, fair) | GBDT, ppi_derived (fair) |
|---|---|---|---|---|---|---|---|
| apple-privaterelay | 99.9 ± 0.0 | 99.8 ± 0.0 | 99.8 ± 0.0 | 99.9 ± 0.1 | 99.9 ± 0.0 | 91.8 ± 3.0 | 93.3 ± 3.2 |
| cloudflare-cdnjs | 64.3 ± 1.3 | 68.0 ± 0.2 | 69.8 ± 1.8 | 70.6 ± 0.3 | 73.7 ± 0.5 | 78.3 ± 0.9 | 79.6 ± 0.5 |
| connectad | 94.4 ± 0.1 | 94.8 ± 0.3 | 93.8 ± 0.3 | 96.7 ± 0.1 | 96.3 ± 0.0 | 86.9 ± 2.1 | 91.1 ± 1.8 |
| discord | 94.8 ± 0.2 | 94.9 ± 0.2 | 94.5 ± 0.2 | 96.5 ± 0.2 | 96.4 ± 0.2 | 93.1 ± 0.4 | 94.3 ± 2.0 |
| dns-doh | 99.3 ± 0.1 | 99.4 ± 0.0 | 99.3 ± 0.1 | 98.8 ± 0.4 | 98.6 ± 0.3 | 94.2 ± 1.2 | 95.4 ± 1.0 |
| facebook-connect | 82.9 ± 0.2 | 82.7 ± 0.2 | 82.8 ± 0.1 | 85.8 ± 0.1 | 85.6 ± 0.0 | 82.6 ± 0.2 | 83.7 ± 0.4 |
| facebook-gamesgraph | 98.3 ± 0.0 | 98.7 ± 0.1 | 98.1 ± 0.2 | 98.7 ± 0.2 | 98.7 ± 0.3 | 77.1 ± 2.6 | 77.0 ± 2.9 |
| facebook-graph | 88.8 ± 0.1 | 88.6 ± 0.0 | 88.2 ± 0.1 | 90.4 ± 0.1 | 90.1 ± 0.2 | 82.5 ± 1.5 | 84.9 ± 1.3 |
| facebook-media | 90.5 ± 0.0 | 90.4 ± 0.1 | 89.9 ± 0.2 | 91.8 ± 0.2 | 91.5 ± 0.1 | 87.3 ± 0.6 | 88.6 ± 0.8 |
| facebook-messenger | 80.0 ± 0.3 | 81.2 ± 0.4 | 78.8 ± 0.9 | 82.3 ± 0.7 | 80.2 ± 1.1 | 68.3 ± 3.6 | 71.8 ± 3.0 |
| facebook-rupload | 84.3 ± 0.5 | 84.4 ± 0.4 | 86.2 ± 0.7 | 87.6 ± 0.3 | 88.9 ± 0.1 | 59.2 ± 8.3 | 58.1 ± 3.1 |
| facebook-web | 89.4 ± 0.2 | 89.3 ± 0.0 | 88.7 ± 0.2 | 90.5 ± 0.1 | 90.2 ± 0.1 | 83.0 ± 1.8 | 84.5 ± 3.3 |
| instagram | 93.3 ± 0.1 | 93.3 ± 0.1 | 92.9 ± 0.1 | 94.1 ± 0.2 | 93.7 ± 0.1 | 88.0 ± 0.5 | 89.1 ± 0.8 |
| jsdelivr | 58.3 ± 0.4 | 60.7 ± 0.4 | 60.9 ± 0.4 | 63.1 ± 0.1 | 61.5 ± 1.0 | 70.7 ± 2.1 | 73.2 ± 0.7 |
| microsoft-outlook | 93.7 ± 0.1 | 94.6 ± 0.2 | 94.2 ± 0.1 | 95.2 ± 0.3 | 93.8 ± 0.5 | 90.4 ± 1.0 | 90.5 ± 3.4 |
| microsoft-substrate | 74.8 ± 0.2 | 78.8 ± 0.8 | 75.7 ± 1.1 | 80.8 ± 1.4 | 75.8 ± 2.6 | 61.0 ± 7.7 | 64.7 ± 7.0 |
| openx | 89.5 ± 0.1 | 89.9 ± 0.1 | 87.9 ± 0.2 | 90.9 ± 0.2 | 90.1 ± 0.4 | 81.9 ± 2.6 | 84.2 ± 0.7 |
| snapchat | 97.7 ± 0.2 | 97.2 ± 0.0 | 96.8 ± 0.1 | 98.3 ± 0.1 | 98.3 ± 0.1 | 90.6 ± 1.7 | 93.5 ± 0.6 |
| spotify | 97.3 ± 0.0 | 97.2 ± 0.1 | 96.7 ± 0.1 | 97.3 ± 0.1 | 97.0 ± 0.2 | 92.8 ± 1.0 | 94.1 ± 1.6 |
| whatsapp | 55.4 ± 1.2 | 61.9 ± 1.0 | 54.8 ± 0.9 | 69.3 ± 1.2 | 66.0 ± 2.7 | 63.7 ± 2.8 | 59.8 ± 3.0 |
