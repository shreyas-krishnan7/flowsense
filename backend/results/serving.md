# Serving benchmark (HTTP, real CESNET test flows)

Model: transformer_ce_p30_s1 (transformer + ce), index: hnsw (efSearch=128), uvicorn 1 worker, client and server on the same laptop (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel / 16 logical CPUs).
Latency is end-to-end client-side (JSON request -> response) for one flow per request. 2,000 requests per level after 100 warm-up requests.

| Concurrency | Throughput (req/s) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| 1 | 368.3 | 2.23 | 5.31 | 6.91 |
| 4 | 735.5 | 5.21 | 8.0 | 9.83 |
| 16 | 884.8 | 15.96 | 33.01 | 47.73 |

## Before fixes (same model, exact index, default threading)
| Concurrency | Throughput (req/s) | p50 ms | p99 ms |
|---|---|---|---|
| 1 | 22.8 | 42.58 | 60.70 |
| 4 | 41.7 | 94.98 | 128.27 |
| 16 | timed out (>30 s), OpenBLAS memory exhaustion | – | – |

Fixes: (1) one thread per library call (OpenBLAS/OpenMP/FAISS/ONNX Runtime) instead of a full pool per concurrent request; (2) HNSW approximate k-NN index (M=32, efSearch=128) instead of exact search over 300k vectors (10.8 ms -> 0.16 ms per query).

## Accuracy cost of HNSW (known test flows, 150k)
| Index | Accuracy % | Macro-F1 % |
|---|---|---|
| exact | 92.852 | 88.854 |
| hnsw_ef64 | 92.831 | 88.804 |
| hnsw_ef128 | 92.845 | 88.831 |

Scope: this measures model serving on pre-extracted flows. It does not include packet capture or flow assembly; a decision also requires observing up to 30 packets of the flow (see E4).
