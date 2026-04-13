#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
DATASET_NAME="${2:-cfd56_holdout400_val0200}"
EXPR_NAME="${3:-gp-edm_holdout400_cfd56}"
SPLIT_DIR="${4:-data/splits/holdout_400_0_0_val_0200}"
META_PATH="${5:-data/cfd56_all_T120_interp48_meta.npy}"
SENSOR_CSV="${6:-data/sensors_real_12.csv}"
FTM_ITERS="${7:-2000}"
GPSD_STEPS="${8:-8000}"
POSTERIOR_STEPS="${9:-50}"

MPDPS_WEIGHT="${MPDPS_WEIGHT:-16}"
OBS_RHO="${OBS_RHO:-0.01}"
ZETA="${ZETA:-0.03}"
OBS_INJECTION_MODE="${OBS_INJECTION_MODE:-direct_inner}"
OBS_SCALE_SCHEDULE="${OBS_SCALE_SCHEDULE:-constant}"
OBS_SCALE_BLEND="${OBS_SCALE_BLEND:-replace}"
OBS_TARGET_RATIO="${OBS_TARGET_RATIO:-0.25}"
OBS_MIN_SCALE="${OBS_MIN_SCALE:-0.0}"
OBS_MAX_SCALE="${OBS_MAX_SCALE:-500.0}"
OBS_INNER_STEPS="${OBS_INNER_STEPS:-3}"
OBS_INNER_DECAY="${OBS_INNER_DECAY:-0.7}"
SAMPLE_WEIGHT_MODE="${SAMPLE_WEIGHT_MODE:-lowflow_focus_v1}"

cd "${REPO_ROOT}"
mkdir -p data ckp exps logs results

latest_match() {
  local pattern="$1"
  local match
  match="$(ls -t ${pattern} 2>/dev/null | head -n 1 || true)"
  echo "${match}"
}

TRAIN_CORE_PATH="$(latest_match "data/core_${DATASET_NAME}_train_4x8x8_*_last.mat")"
BASIS_PATH="$(latest_match "ckp/basis_${DATASET_NAME}_train_4x8x8_*_last.pth")"
if [[ -z "${TRAIN_CORE_PATH}" || -z "${BASIS_PATH}" ]]; then
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
  TRAIN_CORE_PATH="$(latest_match "data/core_${DATASET_NAME}_train_4x8x8_*_last.mat")"
  BASIS_PATH="$(latest_match "ckp/basis_${DATASET_NAME}_train_4x8x8_*_last.pth")"
else
  echo "[skip] FTM already available: ${TRAIN_CORE_PATH}"
fi

EXP_DIR="$(latest_match "exps/${EXPR_NAME}_${DATASET_NAME}_train_*")"
MODEL_PATH=""
CORE_MEAN_STD_PATH=""
if [[ -n "${EXP_DIR}" ]]; then
  MODEL_PATH="$(latest_match "${EXP_DIR}/checkpoints/ema_*.pth")"
  CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"
  if [[ ! -f "${CORE_MEAN_STD_PATH}" ]]; then
    CORE_MEAN_STD_PATH=""
  fi
fi

if [[ -z "${MODEL_PATH}" || -z "${CORE_MEAN_STD_PATH}" ]]; then
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
  EXP_DIR="$(latest_match "exps/${EXPR_NAME}_${DATASET_NAME}_train_*")"
  MODEL_PATH="$(latest_match "${EXP_DIR}/checkpoints/ema_*.pth")"
  CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"
else
  echo "[skip] GPSD already available: ${EXP_DIR}"
fi

SENSOR_BASELINE_JSON="results/${DATASET_NAME}_sensor_param_baseline/sensor_param_baseline.json"

if [[ ! -f "${SENSOR_BASELINE_JSON}" ]]; then
  python3 run_sensor_param_baseline.py \
    --repo_root . \
    --train_manifest "${SPLIT_DIR}/train_manifest.csv" \
    --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
    --train_field_h5 "${SPLIT_DIR}/train.h5" \
    --test_field_h5 "${SPLIT_DIR}/test.h5" \
    --metadata_path "${META_PATH}" \
    --sensor_csv "${SENSOR_CSV}" \
    --output_dir "results/${DATASET_NAME}_sensor_param_baseline" \
    --sample_weight_mode "${SAMPLE_WEIGHT_MODE}"
else
  echo "[skip] sensor baseline already available: ${SENSOR_BASELINE_JSON}"
fi

for SPLIT_NAME in val test; do
  AGG_JSON="${EXP_DIR}/${SPLIT_NAME}_eval/aggregate_metrics.json"
  if [[ ! -f "${AGG_JSON}" ]]; then
    python3 run_holdout_reconstruction_eval.py \
      --repo_root . \
      --test_h5 "${SPLIT_DIR}/${SPLIT_NAME}.h5" \
      --test_manifest "${SPLIT_DIR}/${SPLIT_NAME}_manifest.csv" \
      --metadata_path "${META_PATH}" \
      --sensor_csv "${SENSOR_CSV}" \
      --basis_path "${BASIS_PATH}" \
      --model_path "${MODEL_PATH}" \
      --core_mean_std_path "${CORE_MEAN_STD_PATH}" \
      --output_dir "${EXP_DIR}/${SPLIT_NAME}_eval" \
      --dataset_prefix "${DATASET_NAME}_${SPLIT_NAME}_eval" \
      --num_posterior_samples 2 \
      --total_steps "${POSTERIOR_STEPS}" \
      --mpdps_weight "${MPDPS_WEIGHT}" \
      --obs_rho "${OBS_RHO}" \
      --zeta "${ZETA}" \
      --obs_injection_mode "${OBS_INJECTION_MODE}" \
      --obs_scale_schedule "${OBS_SCALE_SCHEDULE}" \
      --obs_scale_blend "${OBS_SCALE_BLEND}" \
      --obs_target_ratio "${OBS_TARGET_RATIO}" \
      --obs_min_scale "${OBS_MIN_SCALE}" \
      --obs_max_scale "${OBS_MAX_SCALE}" \
      --obs_inner_steps "${OBS_INNER_STEPS}" \
      --obs_inner_decay "${OBS_INNER_DECAY}" \
      --cleanup_recon_mat \
      --cleanup_sensor_cache
  else
    echo "[skip] ${SPLIT_NAME} eval already available: ${AGG_JSON}"
  fi
done

echo "[done] basis_path=${BASIS_PATH}"
echo "[done] model_path=${MODEL_PATH}"
echo "[done] val_eval=${EXP_DIR}/val_eval/aggregate_metrics.json"
echo "[done] test_eval=${EXP_DIR}/test_eval/aggregate_metrics.json"
echo "[done] sensor_param=results/${DATASET_NAME}_sensor_param_baseline/sensor_param_baseline.json"
