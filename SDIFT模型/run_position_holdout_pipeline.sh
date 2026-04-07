#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <train_data_name> <expr_name> <split_dir> <metadata_path> <sensor_csv> [train_batch_size] [gpsd_steps] [posterior_steps]"
  exit 1
fi

TRAIN_DATA_NAME="$1"
EXPR_NAME="$2"
SPLIT_DIR="$3"
METADATA_PATH="$4"
SENSOR_CSV="$5"
TRAIN_BATCH_SIZE="${6:-16}"
GPSD_STEPS="${7:-8000}"
POSTERIOR_STEPS="${8:-20}"

echo "[info] train_data_name=${TRAIN_DATA_NAME}"
echo "[info] expr_name=${EXPR_NAME}"
echo "[info] split_dir=${SPLIT_DIR}"

python3 train_FTM.py \
  --data_name "${TRAIN_DATA_NAME}" \
  --data_path "${SPLIT_DIR}/train.h5" \
  --metadata_path "${METADATA_PATH}" \
  --R 4 8 8 \
  --max_iter 2000 \
  --save_last

CORE_PATH="$(ls -t data/core_${TRAIN_DATA_NAME}_4x8x8_*_last.mat | head -n 1)"
BASIS_PATH="$(ls -t ckp/basis_${TRAIN_DATA_NAME}_4x8x8_*_last.pth | head -n 1)"

echo "[info] core_path=${CORE_PATH}"
echo "[info] basis_path=${BASIS_PATH}"

python3 train_GPSD.py \
  --expr "${EXPR_NAME}" \
  --dataset "${TRAIN_DATA_NAME}" \
  --core_path "${CORE_PATH}" \
  --spatial_dims 3 \
  --img_size_3d 4 8 8 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --num_steps "${GPSD_STEPS}" \
  --save_model_iters 1000 \
  --save_signals_step 0

EXP_DIR="$(ls -td exps/${EXPR_NAME}_${TRAIN_DATA_NAME}_* | head -n 1)"
MODEL_PATH="${EXP_DIR}/checkpoints/ema_$((GPSD_STEPS - 1)).pth"
CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"
HOLDOUT_EVAL_DIR="${EXP_DIR}/holdout_eval"

echo "[info] exp_dir=${EXP_DIR}"
echo "[info] model_path=${MODEL_PATH}"

python3 run_holdout_reconstruction_eval.py \
  --repo_root . \
  --test_h5 "${SPLIT_DIR}/test.h5" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --metadata_path "${METADATA_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --basis_path "${BASIS_PATH}" \
  --model_path "${MODEL_PATH}" \
  --core_mean_std_path "${CORE_MEAN_STD_PATH}" \
  --output_dir "${HOLDOUT_EVAL_DIR}" \
  --dataset_prefix "${TRAIN_DATA_NAME}_holdout_eval" \
  --num_posterior_samples 1 \
  --total_steps "${POSTERIOR_STEPS}"

echo "[done] aggregate_json=${HOLDOUT_EVAL_DIR}/aggregate_metrics.json"
echo "[done] aggregate_csv=${HOLDOUT_EVAL_DIR}/aggregate_metrics.csv"
