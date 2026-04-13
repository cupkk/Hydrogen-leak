#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/SDIFT_model56
mkdir -p logs

RATE_TEST_AGG="results/advisor_study/clean48_holdout_rate0100_formal_20260413_fix/test_eval/aggregate_metrics.json"
POS_TEST_AGG="results/advisor_study/clean48_holdout400_formal_20260413_fix/test_eval/aggregate_metrics.json"
STATUS="logs/clean48_extra_gpu2_status_20260413.txt"

echo "[$(date)] extra GPU2 scheduler started" | tee -a "${STATUS}"
while [[ ! -f "${RATE_TEST_AGG}" || ! -f "${POS_TEST_AGG}" ]]; do
  echo "[$(date)] waiting formal eval before GPU2 helper" | tee -a "${STATUS}"
  sleep 120
done

echo "[$(date)] formal eval done; launching n012_r02 on GPU2" | tee -a "${STATUS}"
CUDA_VISIBLE_DEVICES=2 bash scripts/remote_clean48_train_scale_helper_20260413.sh 12 2 \
  > logs/clean48_scale_helper_n012r02_gpu2_20260413.log 2>&1
echo "[$(date)] extra GPU2 scheduler done" | tee -a "${STATUS}"
