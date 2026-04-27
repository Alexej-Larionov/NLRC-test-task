#!/usr/bin/env bash
set -e

mkdir -p run_logs

ts=$(date +"%Y%m%d_%H%M%S")

echo "=== train adam ==="
python -u train.py --mode adam 2>&1 | tee "run_logs/train_adam_${ts}.log"

echo "=== train muon ==="
python -u train.py --mode muon 2>&1 | tee "run_logs/train_muon_${ts}.log"

echo "=== train mixed ==="
python -u train.py --mode mixed 2>&1 | tee "run_logs/train_mixed_${ts}.log"

echo "=== eval all checkpoints ==="
python -u my_eval_lm.py 2>&1 | tee "run_logs/eval_all_${ts}.log"

echo "done"