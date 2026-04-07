#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 <expr_name> <dataset_name> <core_path> <basis_path> <test_h5> <test_manifest> <metadata_path> <sensor_csv> [gpsd_steps] [posterior_steps] [train_batch_size] [save_model_iters]"
  exit 1
fi

EXPR_NAME="$1"
DATASET_NAME="$2"
CORE_PATH="$3"
BASIS_PATH="$4"
TEST_H5="$5"
TEST_MANIFEST="$6"
METADATA_PATH="$7"
SENSOR_CSV="$8"
GPSD_STEPS="${9:-300}"
POSTERIOR_STEPS="${10:-10}"
TRAIN_BATCH_SIZE="${11:-16}"
SAVE_MODEL_ITERS="${12:-100}"

echo "[info] expr_name=${EXPR_NAME}"
echo "[info] dataset_name=${DATASET_NAME}"
echo "[info] core_path=${CORE_PATH}"
echo "[info] basis_path=${BASIS_PATH}"

python3 train_GPSD.py \
  --expr "${EXPR_NAME}" \
  --dataset "${DATASET_NAME}" \
  --core_path "${CORE_PATH}" \
  --spatial_dims 3 \
  --img_size_3d 4 8 8 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --num_steps "${GPSD_STEPS}" \
  --save_model_iters "${SAVE_MODEL_ITERS}" \
  --log_step "${SAVE_MODEL_ITERS}" \
  --save_signals_step 0

EXP_DIR="$(ls -td exps/${EXPR_NAME}_${DATASET_NAME}_* | head -n 1)"
MODEL_PATH="${EXP_DIR}/checkpoints/ema_$((GPSD_STEPS - 1)).pth"
CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"
HOLDOUT_EVAL_DIR="${EXP_DIR}/holdout_eval"

echo "[info] exp_dir=${EXP_DIR}"
echo "[info] model_path=${MODEL_PATH}"

python3 run_holdout_reconstruction_eval.py \
  --repo_root . \
  --test_h5 "${TEST_H5}" \
  --test_manifest "${TEST_MANIFEST}" \
  --metadata_path "${METADATA_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --basis_path "${BASIS_PATH}" \
  --model_path "${MODEL_PATH}" \
  --core_mean_std_path "${CORE_MEAN_STD_PATH}" \
  --output_dir "${HOLDOUT_EVAL_DIR}" \
  --dataset_prefix "${EXPR_NAME}_eval" \
  --num_posterior_samples 1 \
  --total_steps "${POSTERIOR_STEPS}"

echo "[done] aggregate_json=${HOLDOUT_EVAL_DIR}/aggregate_metrics.json"
echo "[done] aggregate_csv=${HOLDOUT_EVAL_DIR}/aggregate_metrics.csv"
