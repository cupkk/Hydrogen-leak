#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/SDIFT_model56
mkdir -p logs
export PYTHONUNBUFFERED=1

RATE_TEST_AGG="results/advisor_study/clean48_holdout_rate0100_formal_20260413_fix/test_eval/aggregate_metrics.json"
POS_TEST_AGG="results/advisor_study/clean48_holdout400_formal_20260413_fix/test_eval/aggregate_metrics.json"
SCALE_ROOT="results/advisor_study/train_scale_lowflow_focus_clean48_20260413"
STATUS="logs/clean48_nightly_status_20260413.txt"

echo "[$(date)] nightly scheduler started" | tee -a "${STATUS}"

while [[ ! -f "${RATE_TEST_AGG}" || ! -f "${POS_TEST_AGG}" ]]; do
  echo "[$(date)] waiting formal eval: rate=$([[ -f "${RATE_TEST_AGG}" ]] && echo done || echo running), position=$([[ -f "${POS_TEST_AGG}" ]] && echo done || echo running)" | tee -a "${STATUS}"
  sleep 120
done

echo "[$(date)] formal eval done; launching train-size helpers" | tee -a "${STATUS}"

CUDA_VISIBLE_DEVICES=0 bash scripts/remote_clean48_train_scale_helper_20260413.sh 24 0 1 2 \
  > logs/clean48_scale_helper_n024_gpu0_20260413.log 2>&1 &
PID24=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/remote_clean48_train_scale_helper_20260413.sh 31 0 1 2 \
  > logs/clean48_scale_helper_n031_gpu1_20260413.log 2>&1 &
PID31=$!

echo "[$(date)] helpers started: n024=${PID24}, n031=${PID31}" | tee -a "${STATUS}"
wait "${PID24}" "${PID31}"
echo "[$(date)] helpers finished" | tee -a "${STATUS}"

while pgrep -f "run_training_scale_repeated_study.py --repo_root . --train_h5 data/splits_clean/holdout_400_0_0_val_300_0_0/train.h5" >/dev/null; do
  count="$(find "${SCALE_ROOT}" -path '*lowflow_focus_v1_n*_r*/aggregate_metrics.json' 2>/dev/null | wc -l)"
  echo "[$(date)] waiting parent scale process; completed aggregates=${count}/12" | tee -a "${STATUS}"
  sleep 300
done

count="$(find "${SCALE_ROOT}" -path '*lowflow_focus_v1_n*_r*/aggregate_metrics.json' 2>/dev/null | wc -l)"
echo "[$(date)] parent exited; completed aggregates=${count}/12" | tee -a "${STATUS}"

echo "[$(date)] final error scan" | tee -a "${STATUS}"
for f in \
  logs/clean48_rate_formal_eval_fix_20260413.log \
  logs/clean48_position_formal_eval_fix_20260413.log \
  logs/clean48_scale_lowflow_focus_20260413.log \
  logs/clean48_scale_helper_n024_gpu0_20260413.log \
  logs/clean48_scale_helper_n031_gpu1_20260413.log; do
  echo "--- ${f}" | tee -a "${STATUS}"
  grep -nE 'Traceback|Error|FileNotFoundError|RuntimeError|CUDA|Killed|No such file|returned non-zero' "${f}" | tail -20 | tee -a "${STATUS}" || true
done

echo "[$(date)] nightly scheduler done" | tee -a "${STATUS}"
