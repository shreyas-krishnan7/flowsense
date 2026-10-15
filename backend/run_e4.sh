#!/usr/bin/env bash
# E4: accuracy vs number of packets (selected model = Transformer + CE, seed 0; fair GBDT on CPU).
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
(
  for p in 3 5 10 15 20; do
    OMP_NUM_THREADS=6 $PY -u baseline.py --features ppi --packets $p --seed 0 2>&1 | grep --line-buffered -E "GBDT|refusing|Error"
  done
  echo "=== GBDT E4 done $(date +%H:%M)"
) &
for p in 3 5 10 15 20; do
  echo "=== transformer_ce_p${p}_s0 $(date +%H:%M)"
  $PY -u train.py --encoder transformer --loss ce --epochs 20 --seed 0 --packets $p 2>&1 \
    | grep --line-buffered -E "epoch 20/|k-NN|softmax head|Error|error|refusing"
done
wait
echo "=== all done $(date +%H:%M)"
