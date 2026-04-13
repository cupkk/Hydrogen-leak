#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "${REPO_ROOT}"

mkdir -p logs results/advisor_study

SPLIT_POS_DIR="${SPLIT_POS_DIR:-data/splits/holdout_400_0_0_val_0200}"
SPLIT_RATE_DIR="${SPLIT_RATE_DIR:-data/splits/holdout_rate_0100_val_0200}"
META_PATH="${META_PATH:-data/cfd56_all_T120_interp48_meta.npy}"
SENSOR_CSV="${SENSOR_CSV:-data/sensors_real_12.csv}"

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
RATE_SAMPLE_WEIGHT_MODE="${RATE_SAMPLE_WEIGHT_MODE:-lowflow_balanced_v1}"

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python3 run_training_scale_repeated_study.py \
  --repo_root . \
  --train_h5 "${SPLIT_POS_DIR}/train.h5" \
  --train_manifest "${SPLIT_POS_DIR}/train_manifest.csv" \
  --test_h5 "${SPLIT_POS_DIR}/test.h5" \
  --test_manifest "${SPLIT_POS_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --train_sizes 6 12 24 42 \
  --repeats 3 \
  --output_root results/advisor_study/train_scale_none_cfd56_relabel20260411 \
  --study_tag cfd56_scale_none_relabel20260411 \
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
  > logs/cfd56_scale_none_relabel20260411.log 2>&1 &
PID_NONE=$!

CUDA_VISIBLE_DEVICES=1 python3 run_training_scale_repeated_study.py \
  --repo_root . \
  --train_h5 "${SPLIT_POS_DIR}/train.h5" \
  --train_manifest "${SPLIT_POS_DIR}/train_manifest.csv" \
  --test_h5 "${SPLIT_POS_DIR}/test.h5" \
  --test_manifest "${SPLIT_POS_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --train_sizes 6 12 24 42 \
  --repeats 3 \
  --output_root results/advisor_study/train_scale_lowflow_focus_cfd56_relabel20260411 \
  --study_tag cfd56_scale_focus_relabel20260411 \
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
  > logs/cfd56_scale_lowflow_focus_relabel20260411.log 2>&1 &
PID_FOCUS=$!

CUDA_VISIBLE_DEVICES=2 python3 run_training_scale_repeated_study.py \
  --repo_root . \
  --train_h5 "${SPLIT_POS_DIR}/train.h5" \
  --train_manifest "${SPLIT_POS_DIR}/train_manifest.csv" \
  --test_h5 "${SPLIT_POS_DIR}/test.h5" \
  --test_manifest "${SPLIT_POS_DIR}/test_manifest.csv" \
  --metadata_path "${META_PATH}" \
  --sensor_csv "${SENSOR_CSV}" \
  --train_sizes 6 12 24 42 \
  --repeats 3 \
  --output_root results/advisor_study/train_scale_lowflow_balanced_cfd56_relabel20260411 \
  --study_tag cfd56_scale_balanced_relabel20260411 \
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
  > logs/cfd56_scale_lowflow_balanced_relabel20260411.log 2>&1 &
PID_BALANCED=$!

CUDA_VISIBLE_DEVICES=3 SAMPLE_WEIGHT_MODE="${RATE_SAMPLE_WEIGHT_MODE}" bash ./run_56case_formal_pipeline.sh \
  . \
  "cfd56_holdoutrate0100_val0200" \
  "gp-edm_holdoutrate0100_cfd56" \
  "${SPLIT_RATE_DIR}" \
  "${META_PATH}" \
  "${SENSOR_CSV}" \
  2000 \
  8000 \
  50 \
  > logs/cfd56_holdoutrate0100_val0200_formal.log 2>&1 &
PID_RATE=$!

echo "[pids] none=${PID_NONE} focus=${PID_FOCUS} balanced=${PID_BALANCED} holdout_rate=${PID_RATE}"

wait "${PID_NONE}"
wait "${PID_FOCUS}"
wait "${PID_BALANCED}"
wait "${PID_RATE}"

echo "[done] scale_none=logs/cfd56_scale_none_relabel20260411.log"
echo "[done] scale_focus=logs/cfd56_scale_lowflow_focus_relabel20260411.log"
echo "[done] scale_balanced=logs/cfd56_scale_lowflow_balanced_relabel20260411.log"
echo "[done] holdout_rate=logs/cfd56_holdoutrate0100_val0200_formal.log"
