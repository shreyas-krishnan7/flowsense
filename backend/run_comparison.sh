#!/usr/bin/env bash
# Encoder x loss comparison on real CESNET-QUIC22 data (E3), seed 0.
# Runs sequentially on one GPU; existing runs are never overwritten.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
for cfg in "cnn supcon" "cnn ce" "gru supcon" "gru ce" "transformer supcon" "transformer ce"; do
  set -- $cfg
  run="${1}_${2}_p30_s0"
  echo "=== $run $(date +%H:%M)"
  $PY -u train.py --encoder "$1" --loss "$2" --epochs 20 --seed 0 2>&1 \
    | grep --line-buffered -E "params|epoch +(1|5|10|15|20)/|k-NN|softmax head|Error|error|refusing"
  [ -f "runs/$run/embeddings.npz" ] && $PY -u evaluate.py "runs/$run" --skip-fewshot > "runs/$run/eval.log" 2>&1 \
    && echo "evaluated $run $(date +%H:%M)"
done
$PY evaluate.py --table
echo "=== done $(date +%H:%M)"
