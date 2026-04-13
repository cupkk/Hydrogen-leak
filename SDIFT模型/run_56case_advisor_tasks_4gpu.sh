#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "${REPO_ROOT}"

mkdir -p data ckp exps logs results/advisor_study

DATASET_NAME="cfd56_holdout400_val0200"
EXPR_NAME="gp-edm_holdout400_cfd56"
SPLIT_DIR="data/splits/holdout_400_0_0_val_0200"
META_PATH="data/cfd56_all_T120_interp48_meta.npy"
SENSOR_CSV="data/sensors_real_12.csv"

export MPDPS_WEIGHT="${MPDPS_WEIGHT:-16}"
export OBS_RHO="${OBS_RHO:-0.01}"
export ZETA="${ZETA:-0.03}"
export OBS_INJECTION_MODE="${OBS_INJECTION_MODE:-direct_inner}"
export OBS_SCALE_SCHEDULE="${OBS_SCALE_SCHEDULE:-constant}"
export OBS_SCALE_BLEND="${OBS_SCALE_BLEND:-replace}"
export OBS_TARGET_RATIO="${OBS_TARGET_RATIO:-0.25}"
export OBS_MIN_SCALE="${OBS_MIN_SCALE:-0.0}"
export OBS_MAX_SCALE="${OBS_MAX_SCALE:-500.0}"
export OBS_INNER_STEPS="${OBS_INNER_STEPS:-3}"
export OBS_INNER_DECAY="${OBS_INNER_DECAY:-0.7}"
export SAMPLE_WEIGHT_MODE="${SAMPLE_WEIGHT_MODE:-lowflow_focus_v1}"

BASELINE_LOG="logs/cfd56_formal_pipeline_gpu0.log"
SENSOR_LOG="logs/cfd56_sensor_conditions_gpu0.log"
SCALE_NONE_LOG="logs/cfd56_scale_none_gpu1.log"
SCALE_FOCUS_LOG="logs/cfd56_scale_focus_gpu2.log"
SCALE_BALANCED_LOG="logs/cfd56_scale_balanced_gpu3.log"

CUDA_VISIBLE_DEVICES=0 bash ./run_56case_formal_pipeline.sh \
  . \
  "${DATASET_NAME}" \
  "${EXPR_NAME}" \
  "${SPLIT_DIR}" \
  "${META_PATH}" \
  "${SENSOR_CSV}" \
  2000 \
  8000 \
  50 \
  > "${BASELINE_LOG}" 2>&1 &
BASELINE_PID=$!

CUDA_VISIBLE_DEVICES=1 python3 run_training_scale_repeated_study.py \
  --repo_root . \
  --train_h5 "${SPLIT_DIR}/train.h5" \
  --train_manifest "${SPLIT_DIR}/train_manifest.csv" \
  --test_h5 "${SPLIT_DIR}/test.h5" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --train_sizes 6 12 24 42 \
  --repeats 3 \
  --output_root results/advisor_study/train_scale_none_cfd56 \
  --study_tag cfd56_scale_none \
  --sample_weight_modes none \
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
  > "${SCALE_NONE_LOG}" 2>&1 &
SCALE_NONE_PID=$!

CUDA_VISIBLE_DEVICES=2 python3 run_training_scale_repeated_study.py \
  --repo_root . \
  --train_h5 "${SPLIT_DIR}/train.h5" \
  --train_manifest "${SPLIT_DIR}/train_manifest.csv" \
  --test_h5 "${SPLIT_DIR}/test.h5" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --train_sizes 6 12 24 42 \
  --repeats 3 \
  --output_root results/advisor_study/train_scale_lowflow_focus_cfd56 \
  --study_tag cfd56_scale_focus \
  --sample_weight_modes lowflow_focus_v1 \
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
  > "${SCALE_FOCUS_LOG}" 2>&1 &
SCALE_FOCUS_PID=$!

CUDA_VISIBLE_DEVICES=3 python3 run_training_scale_repeated_study.py \
  --repo_root . \
  --train_h5 "${SPLIT_DIR}/train.h5" \
  --train_manifest "${SPLIT_DIR}/train_manifest.csv" \
  --test_h5 "${SPLIT_DIR}/test.h5" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --train_sizes 6 12 24 42 \
  --repeats 3 \
  --output_root results/advisor_study/train_scale_lowflow_balanced_cfd56 \
  --study_tag cfd56_scale_balanced \
  --sample_weight_modes lowflow_balanced_v1 \
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
  > "${SCALE_BALANCED_LOG}" 2>&1 &
SCALE_BALANCED_PID=$!

wait "${BASELINE_PID}"

BASIS_PATH="$(ls -t ckp/basis_${DATASET_NAME}_train_4x8x8_*_last.pth | head -n 1)"
EXP_DIR="$(ls -td exps/${EXPR_NAME}_${DATASET_NAME}_train_* | head -n 1)"
MODEL_PATH="${EXP_DIR}/checkpoints/ema_7999.pth"
CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"

CUDA_VISIBLE_DEVICES=0 python3 run_sensor_condition_study.py \
  --repo_root . \
  --test_h5 "${SPLIT_DIR}/test.h5" \
  --test_manifest "${SPLIT_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --sensor_counts 6 12 30 \
  --observed_time_steps 20 60 120 \
  --basis_path "${BASIS_PATH}" \
  --model_path "${MODEL_PATH}" \
  --core_mean_std_path "${CORE_MEAN_STD_PATH}" \
  --output_root results/advisor_study/sensor_conditions_cfd56_holdout400 \
  --dataset_prefix cfd56_sensor_cond \
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
  --num_posterior_samples 2 \
  --total_steps 50 \
  > "${SENSOR_LOG}" 2>&1

wait "${SCALE_NONE_PID}"
wait "${SCALE_FOCUS_PID}"
wait "${SCALE_BALANCED_PID}"

echo "[done] baseline=${BASELINE_LOG}"
echo "[done] sensor=${SENSOR_LOG}"
echo "[done] scale_none=${SCALE_NONE_LOG}"
echo "[done] scale_focus=${SCALE_FOCUS_LOG}"
echo "[done] scale_balanced=${SCALE_BALANCED_LOG}"
