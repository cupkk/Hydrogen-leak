#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
DATASET_NAME="${2:-cfd38_holdout400}"
EXPR_NAME="${3:-gp-edm_holdout400}"
POSTERIOR_STEPS="${4:-20}"
GPSD_STEPS="${5:-8000}"
FTM_ITERS="${6:-2000}"
TEST_CORE_ITERS="${7:-600}"
SAMPLE_WEIGHT_MODE="${8:-none}"

cd "${REPO_ROOT}"

FULL_H5="data/cfd38_all_T120_interp48.h5"
FULL_MANIFEST="data/cfd38_all_T120_interp48_manifest.csv"
META_PATH="data/cfd38_all_T120_interp48_meta.npy"
SENSOR_CSV="data/sensors_real_12.csv"
SPLIT_DIR="data/splits/holdout_400_0_0"

mkdir -p "${SPLIT_DIR}"

if [[ ! -f "${SPLIT_DIR}/train.h5" || ! -f "${SPLIT_DIR}/test.h5" ]]; then
  python3 subset_h5_by_manifest.py \
    --input_h5 "${FULL_H5}" \
    --manifest_csv "${FULL_MANIFEST}" \
    --holdout_position 400,0,0 \
    --out_train_h5 "${SPLIT_DIR}/train.h5" \
    --out_train_manifest "${SPLIT_DIR}/train_manifest.csv" \
    --out_test_h5 "${SPLIT_DIR}/test.h5" \
    --out_test_manifest "${SPLIT_DIR}/test_manifest.csv" \
    --out_split_json "${SPLIT_DIR}/split.json"
fi

python3 train_FTM.py \
  --data_name "${DATASET_NAME}_train" \
  --data_path "${SPLIT_DIR}/train.h5" \
  --manifest_path "${SPLIT_DIR}/train_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sample_weight_mode "${SAMPLE_WEIGHT_MODE}" \
  --R 4 8 8 \
  --max_iter "${FTM_ITERS}" \
  --batch_size 8 \
  --save_last

TRAIN_CORE_PATH="$(ls -t data/core_${DATASET_NAME}_train_4x8x8_*_last.mat | head -n 1)"
BASIS_PATH="$(ls -t ckp/basis_${DATASET_NAME}_train_4x8x8_*_last.pth | head -n 1)"

python3 infer_FTM_core.py \
  --basis_path "${BASIS_PATH}" \
  --data_name "${DATASET_NAME}_test" \
  --data_path "${SPLIT_DIR}/test.h5" \
  --metadata_path "${META_PATH}" \
  --R 4 8 8 \
  --batch_size 4 \
  --max_iter "${TEST_CORE_ITERS}" \
  --out_core_path "data/core_${DATASET_NAME}_test_4x8x8_encoded.mat" \
  --out_json "results/${DATASET_NAME}_test_core_summary.json"

TEST_CORE_PATH="data/core_${DATASET_NAME}_test_4x8x8_encoded.mat"

python3 run_source_param_model_comparison.py \
  --train_manifest "${SPLIT_DIR}/train_manifest.csv" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --train_field_h5 "${SPLIT_DIR}/train.h5" \
  --test_field_h5 "${SPLIT_DIR}/test.h5" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --sample_weight_mode "${SAMPLE_WEIGHT_MODE}" \
  --train_core_path "${TRAIN_CORE_PATH}" \
  --test_core_path "${TEST_CORE_PATH}" \
  --out_dir "results/${DATASET_NAME}_param_compare"

python3 train_GPSD.py \
  --expr "${EXPR_NAME}" \
  --dataset "${DATASET_NAME}_train" \
  --core_path "${TRAIN_CORE_PATH}" \
  --manifest_path "${SPLIT_DIR}/train_manifest.csv" \
  --spatial_dims 3 \
  --img_size_3d 4 8 8 \
  --sample_weight_mode "${SAMPLE_WEIGHT_MODE}" \
  --train_batch_size 8 \
  --num_steps "${GPSD_STEPS}" \
  --save_model_iters 1000 \
  --save_signals_step 0

EXP_DIR="$(ls -td exps/${EXPR_NAME}_${DATASET_NAME}_train_* | head -n 1)"
MODEL_PATH="${EXP_DIR}/checkpoints/ema_$((GPSD_STEPS - 1)).pth"
CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"

python3 run_holdout_reconstruction_eval.py \
  --repo_root . \
  --test_h5 "${SPLIT_DIR}/test.h5" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --basis_path "${BASIS_PATH}" \
  --model_path "${MODEL_PATH}" \
  --core_mean_std_path "${CORE_MEAN_STD_PATH}" \
  --output_dir "${EXP_DIR}/holdout_eval" \
  --dataset_prefix "${DATASET_NAME}_holdout_eval" \
  --num_posterior_samples 1 \
  --total_steps "${POSTERIOR_STEPS}"

echo "[done] basis_path=${BASIS_PATH}"
echo "[done] train_core_path=${TRAIN_CORE_PATH}"
echo "[done] test_core_path=${TEST_CORE_PATH}"
echo "[done] param_compare=results/${DATASET_NAME}_param_compare/ranking.json"
echo "[done] holdout_eval=${EXP_DIR}/holdout_eval/aggregate_metrics.json"
