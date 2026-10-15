#!/usr/bin/env bash
# Seed study: TCN SupCon vs TCN cross-entropy (GPU) and fair GBDT (CPU), seeds 0-2,
# identical cached data split. Existing runs are never overwritten.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
(
  for f in ppi ppi_derived; do for s in 0 1 2; do
    OMP_NUM_THREADS=6 $PY -u baseline.py --features $f --seed $s 2>&1 | grep --line-buffered -E "GBDT|refusing|Error"
  done; done
  echo "=== GBDT done $(date +%H:%M)"
) &
for s in 0 1 2; do for loss in supcon ce; do
  echo "=== tcn_${loss}_p30_s${s} $(date +%H:%M)"
  $PY -u train.py --encoder tcn --loss $loss --epochs 20 --seed $s 2>&1 \
    | grep --line-buffered -E "epoch 20/|k-NN|softmax head|Error|error|refusing"
done; done
wait
echo "=== all done $(date +%H:%M)"
