#!/usr/bin/env bash
# Reproduce every reported FlowSense result, in order. Approximate times on an
# RTX 3050 laptop in brackets. Existing run directories are never overwritten.
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python.exe}

$PY download_data.py --data-root C:/datasets/CESNET-QUIC22 --size M     # once, ~13 GB
$PY prepare_data.py                                                    # data/cesnet_top20.npz  [~5-15 min]
$PY extract_meta.py                                                    # data/cesnet_top20_meta.npz (source IPs, eval only) [~4 min]
$PY -m pytest tests -q                                                 # 35 tests

bash run_comparison.sh        # E3: CNN/GRU/Transformer x SupCon/CE, seed 0      [~2 h]
bash run_seeds.sh             # TCN SupCon vs CE x3 seeds + fair GBDT x3 seeds    [~35 min]
for s in 1 2; do $PY train.py --encoder transformer --loss ce --seed $s; done   # [~25 min]
$PY compare_seeds.py          # results/seed_study.md
$PY select_model.py           # results/model_selection.md  (validation only)
bash run_e4.sh                # E4: packet-count sweep                            [~45 min]
$PY e4_packets.py             # results/e4_packets.md
$PY evaluate.py runs/tcn_supcon_p30 --tsne                       # E1/E5/E6/E8 + legacy E7 + t-SNE
$PY fewshot.py runs/transformer_ce_p30_s1                        # E7 (prototype) -> negative result
$PY eval_extra.py runs/transformer_ce_p30_s1                     # calibration, open-set F1, breakdowns, source-IP
$PY export.py runs/transformer_ce_p30_s1 --index hnsw            # artifacts/ + manifest.json
# E9: in one terminal `uvicorn serve:app --port 8000`, then:
# $PY loadtest.py --requests 2000 --concurrency 1 4 16           # results/serving_benchmark.json
$PY evaluate.py --table                                          # results/all_runs.md
