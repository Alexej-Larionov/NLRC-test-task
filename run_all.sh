#!/usr/bin/env bash
set -e

mkdir -p run_logs

ts=$(date +"%Y%m%d_%H%M%S")

echo "=== train ==="
python -u train.py --mode adam --config Experiment1.json
python -u train.py --mode muon --config Experiment1.json
python -u train.py --mode mixed --config Experiment1.json
python -u train.py --mode adam --config Experiment2.json
python -u train.py --mode muon --config Experiment2.json
python -u train.py --mode mixed --config Experiment2.json
python -u train.py --mode adam --config Experiment3.json
python -u train.py --mode muon --config Experiment3.json
python -u train.py --mode mixed --config Experiment3.json

echo "=== eval  ==="
python -u my_eval_lm.py 2>&1 | tee "run_logs/eval_all_${ts}.log"

echo "done"