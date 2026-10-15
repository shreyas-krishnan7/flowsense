# FlowSense

**Open-set identification of encrypted QUIC services from packet metadata.**
FlowSense looks at the first 30 packets of a flow — payload size, gap since the
previous packet, and direction — and decides which of 20 known services it belongs
to, routes ambiguous flows to review, and flags flows from services it has never seen.
It never reads payload bytes, IP addresses, ports or the server name.

> **Scope.** This is *service* identification (e.g. Instagram vs Spotify vs Outlook),
> not traffic-*category* classification. The dataset has no VPN, XR or gaming-category
> labels, and FlowSense makes no claims about them. Processing is validated **offline**
> (PCAP files and an HTTP API on extracted flows); live capture is not implemented.

## Headline results (CESNET-QUIC22, time-based split)

| | Value | Source |
|---|---|---|
| Accuracy / macro-F1, 20 services (3 seeds) | **92.83 ± 0.02% / 88.92 ± 0.11%** | `results/seed_study.md` |
| Fair GBDT on the same 30-packet input | 87.00 ± 1.12% / 81.17 ± 1.76% | `results/seed_study.md` |
| Accuracy on the 80% most-confident flows | 99.1% | `runs/transformer_ce_p30_s1/eval.json` |
| Calibration error (ECE, softmax / k-NN vote) | 0.004 / 0.013 | `results/eval_extra.md` |
| Unseen-service detection AUROC | 0.938 | `runs/transformer_ce_p30_s1/eval.json` |
| Open-set macro-F1 (20 services + unknown) | 83.6% | `results/eval_extra.md` |
| Accuracy across three later test weeks | 92.6 – 93.0% | `runs/transformer_ce_p30_s1/eval.json` |
| Accuracy on clients never seen in training | 92.3% (vs 93.0% seen) | `results/eval_extra.md` |
| HTTP latency p50 / throughput (16 clients) | 2.2 ms / 885 req/s | `results/serving.md` |

## Method

```
first 30 packets ─► [payload bytes, gap ms, direction] ─► log + standardise
        │
        ▼
  Transformer encoder (2 layers, 4 heads, attention pooling, 298k params)
  trained with cross-entropy (cosine head) ─► 128-d L2-normalised embedding
        │
        ├─► FAISS k-NN vote (k=10, HNSW) ─► label + vote gap ─► classified / needs_review
        └─► mean similarity to nearest training flows ─► novel_traffic
```

Thresholds are set on the validation split only: novelty at 95% known-flow retention,
review at the 10% lowest vote gaps.

| Component | File |
|---|---|
| Packet features, PCAP extraction (CESNET conventions) | `backend/features.py` |
| Encoders: CNN, TCN, TCN + attention, GRU, Transformer; SupCon + cosine CE | `backend/model.py` |
| Decision layer: k-NN vote, novelty scores, thresholds | `backend/core.py` |
| Training (any encoder × loss × seed × packet count) | `backend/train.py` |
| Export (ONNX + index + validation thresholds + manifest) | `backend/export.py` |
| Inference, HTTP API, offline PCAP classification | `backend/inference.py`, `serve.py`, `classify_pcap.py` |

## Data and protocol

[CESNET-QUIC22](https://zenodo.org/records/10728760) (Czech national research ISP), via `cesnet-datazoo`.

- **Train / validation:** week 44 of 2022 (300,000 / 30,000 flows).
- **Test:** weeks 45–47 — 150,000 flows of the 20 known services + 50,000 flows of 85 unknown services.
- **Leakage checks:** no time overlap; 2.75% of test flows are byte-identical to a training flow and removing them leaves accuracy unchanged (checked on the TCN: 91.51% vs 91.54%); 21.7% of known test flows come from unseen source IPs (accuracy 92.3% vs 93.0%).
- **Model selection:** mean validation macro-F1 over seeds — never test numbers (`results/model_selection.md`).

## Results

### Encoders and training objective (`results/seed_study.md`, `results/model_selection.md`)

| Model | Seeds | Accuracy % | Macro-F1 % |
|---|---|---|---|
| **Transformer + cross-entropy (selected)** | 3 | **92.83 ± 0.02** | **88.92 ± 0.11** |
| GRU + cross-entropy | 1 | 92.62 | 88.52 |
| CNN + cross-entropy | 1 | 92.09 | 87.75 |
| TCN + cross-entropy | 3 | 91.76 ± 0.06 | 87.29 ± 0.07 |
| TCN + SupCon | 3 | 91.56 ± 0.03 | 86.34 ± 0.13 |
| GBDT, 30 packets + aggregates of them | 3 | 88.63 ± 1.45 | 82.58 ± 1.79 |
| GBDT, same 90 values as the encoders | 3 | 87.00 ± 1.12 | 81.17 ± 1.76 |
| Majority class | – | 12.74 | 1.13 |

- Cross-entropy beat supervised contrastive learning for every encoder; for the TCN the paired
  difference is −0.95 ± 0.16 macro-F1 points (all three seeds negative).
- An earlier GBDT baseline (91.24%) used whole-flow statistics (total bytes, packets, duration)
  that are not available after 30 packets; with only the same 30 packets it drops 4.2 points.
- Weakest services: jsdelivr / cloudflare-cdnjs (two JavaScript CDNs confused with each other)
  and whatsapp (confused with facebook-media — shared Meta infrastructure).

### How early can a service be identified? (`results/e4_packets.md`)

| Packets seen | 3 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| Transformer accuracy % | 64.7 | 74.7 | 79.9 | 87.6 | 91.1 | 92.8 |
| Fair GBDT accuracy % | 69.2 | 69.0 | 79.8 | 86.9 | 86.9 | 87.1 |

GBDT plateaus at 15 packets; the Transformer's lead comes entirely from packets 15–30.

### Open set, calibration, breakdowns (`results/eval_extra.md`)

- At the validation threshold, 81.2% of unknown-service flows are flagged and 93.4% of accepted known flows are correct.
- The threshold was tuned for 5% false alarms on week 44 but gives 8.4% on weeks 45–47: it needs periodic re-calibration.
- 80% of unknown test flows are Google services (86% flagged); independent services are flagged 69% of the time.

### Serving (`results/serving.md`)

ONNX Runtime + FAISS HNSW behind FastAPI (1 worker, same laptop, real flows):
2.2 ms p50 at 1 client, 885 req/s at 16 clients. HNSW changes accuracy by −0.007 points vs exact search.
The first benchmark (22.8 req/s, failure at 16 clients) exposed per-request thread-pool explosion, fixed in `inference.py`.

## What did not work

- **Few-shot registration of new services** (`results/e7_fewshot.md`): with 10 labeled flows per new service,
  prototype classification reached 34.6% (pre-set bar: 70%). The claim was removed; adding a service requires retraining.
- **Supervised contrastive learning** as the main objective: consistently worse than cross-entropy here.

## Limitations

- One dataset and one protocol (QUIC); no cross-network evaluation yet.
- Labels derive from SNI; the practical value is where SNI is hidden (Encrypted ClientHello).
- Novelty threshold drifts across weeks; unknown-service results are dominated by Google services.
- Offline only: no live capture or streaming flow assembly. A decision needs up to 30 packets.

## Reproduce

```bash
cd backend
pip install -r requirements-lock.txt          # CUDA torch: see the file header
python -m pytest tests -q                     # 35 tests
bash reproduce.sh                             # every experiment, in order
```

Every run directory (`backend/runs/<run>/`) holds `summary.json` (config + metrics),
`predictions.npz`, `model.pt` and, where evaluated, `eval.json` and `plots/`.
All tables in `backend/results/` are regenerated from those files. `results/all_runs.md` lists all 33 runs.

Offline PCAP classification with the exported model:

```bash
python classify_pcap.py capture.pcap --udp-443-only --out flows.csv
```
