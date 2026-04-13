#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
EXP_DIR_NAME="${2:-gp-edm_holdout400_cfd38_holdout400_train_20260407-1601}"
WAIT_PATTERN="${3:-run_holdout_reconstruction_eval.py --repo_root . --test_h5 data/splits/holdout_400_0_0/test.h5}"

cd "${REPO_ROOT}"

EXP_DIR="exps/${EXP_DIR_NAME}"
BASIS_PATH="$(ls -t ckp/basis_cfd38_holdout400_train_4x8x8_*_last.pth | head -n 1)"
MODEL_PATH="${EXP_DIR}/checkpoints/ema_7999.pth"
CORE_MEAN_STD_PATH="${EXP_DIR}/core_mean_std.mat"

mkdir -p results/advisor_study

while pgrep -f "${WAIT_PATTERN}" >/dev/null 2>&1; do
  echo "[wait] current holdout reconstruction still running..."
  sleep 30
done

echo "[run] sensor condition study"
python3 run_sensor_condition_study.py \
  --repo_root . \
  --test_h5 data/splits/holdout_400_0_0/test.h5 \
  --test_manifest data/splits/holdout_400_0_0/test_manifest.csv \
  --metadata_path data/cfd38_all_T120_interp48_meta.npy \
  --sensor_csv data/sensors_real.csv \
  --sensor_counts 6 12 30 \
  --observed_time_steps 20 60 120 \
  --basis_path "${BASIS_PATH}" \
  --model_path "${MODEL_PATH}" \
  --core_mean_std_path "${CORE_MEAN_STD_PATH}" \
  --output_root results/advisor_study/sensor_conditions_holdout400 \
  --dataset_prefix advisor_holdout400 \
  --total_steps 20 \
  --num_posterior_samples 1 \
  --low_rates 50 100

echo "[run] training scale study with low-flow weighting"
python3 run_training_scale_study.py \
  --repo_root . \
  --train_h5 data/splits/holdout_400_0_0/train.h5 \
  --train_manifest data/splits/holdout_400_0_0/train_manifest.csv \
  --test_h5 data/splits/holdout_400_0_0/test.h5 \
  --test_manifest data/splits/holdout_400_0_0/test_manifest.csv \
  --metadata_path data/cfd38_all_T120_interp48_meta.npy \
  --sensor_csv data/sensors_real_12.csv \
  --train_sizes 6 12 24 31 \
  --output_root results/advisor_study/train_scale_lowflow_focus \
  --study_tag advisor_scale_lowflow \
  --ftm_batch_size 8 \
  --ftm_max_iter 2000 \
  --gpsd_train_batch_size 8 \
  --gpsd_num_steps 8000 \
  --gpsd_learning_rate 2e-4 \
  --gpsd_accumulation_steps 2 \
  --gpsd_save_model_iters 1000 \
  --gpsd_warmup 1000 \
  --gpsd_save_signals_step 0 \
  --eval_total_steps 20 \
  --num_posterior_samples 1 \
  --sample_weight_mode lowflow_focus_v1 \
  --low_rates 50 100

echo "[done] sensor study: results/advisor_study/sensor_conditions_holdout400/sensor_condition_study.csv"
echo "[done] scale study: results/advisor_study/train_scale_lowflow_focus/training_scale_study.csv"
