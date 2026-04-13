#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/SDIFT_model56
export PYTHONUNBUFFERED=1

SIZE="$1"
shift
REPEATS=("$@")

META="data/cfd48_clean_T120_interp48_meta.npy"
SENSOR="data/sensors_real_12.csv"
TEST_H5="data/splits_clean/holdout_400_0_0_val_300_0_0/test.h5"
TEST_MANIFEST="data/splits_clean/holdout_400_0_0_val_300_0_0/test_manifest.csv"
OUTPUT_ROOT="results/advisor_study/train_scale_lowflow_focus_clean48_20260413"
SUBSET_ROOT="${OUTPUT_ROOT}/subsets"
STUDY_TAG="clean48_scale_focus_20260413"
WEIGHT_MODE="lowflow_focus_v1"
EXPR_NAME="${STUDY_TAG}_${WEIGHT_MODE}_gpsd"

newest_match() {
  local pattern="$1"
  python3 - "$pattern" <<'PY'
import glob, os, sys
matches = glob.glob(sys.argv[1])
if not matches:
    raise SystemExit(1)
matches.sort(key=os.path.getmtime)
print(matches[-1])
PY
}

run_subset() {
  local size="$1"
  local repeat="$2"
  local subset_dir="${SUBSET_ROOT}/size_$(printf '%03d' "${size}")/repeat_$(printf '%02d' "${repeat}")"
  local subset_h5="${subset_dir}/train_$(printf '%03d' "${size}")_r$(printf '%02d' "${repeat}").h5"
  local subset_manifest="${subset_dir}/train_$(printf '%03d' "${size}")_r$(printf '%02d' "${repeat}")_manifest.csv"
  local dataset_name="${STUDY_TAG}_${WEIGHT_MODE}_n$(printf '%03d' "${size}")_r$(printf '%02d' "${repeat}")"
  local eval_out="${OUTPUT_ROOT}/${WEIGHT_MODE}_n$(printf '%03d' "${size}")_r$(printf '%02d' "${repeat}")"
  local agg="${eval_out}/aggregate_metrics.json"
  local lock_dir="${eval_out}.helper_lock"

  if [[ -f "${agg}" ]]; then
    echo "skip completed ${dataset_name}"
    return 0
  fi
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "skip locked ${dataset_name}"
    return 0
  fi
  trap 'rm -rf "${lock_dir}"' RETURN

  mkdir -p "${eval_out}"
  echo "[$(date)] start ${dataset_name}"

  local core_pattern="data/core_${dataset_name}_4x8x8_*_last.mat"
  local basis_pattern="ckp/basis_${dataset_name}_4x8x8_*_last.pth"
  local core_path=""
  local basis_path=""
  core_path="$(newest_match "${core_pattern}" 2>/dev/null || true)"
  basis_path="$(newest_match "${basis_pattern}" 2>/dev/null || true)"

  if [[ -z "${core_path}" || -z "${basis_path}" ]]; then
    python3 train_FTM.py \
      --data_name "${dataset_name}" \
      --data_path "${subset_h5}" \
      --metadata_path "${META}" \
      --R 4 8 8 \
      --batch_size 4 \
      --learning_rate 0.0002 \
      --max_iter 1200 \
      --seed "$((231 + repeat))" \
      --manifest_path "${subset_manifest}" \
      --sample_weight_mode "${WEIGHT_MODE}" \
      --save_last
    core_path="$(newest_match "${core_pattern}")"
    basis_path="$(newest_match "${basis_pattern}")"
  fi

  local exp_pattern="exps/${EXPR_NAME}_${dataset_name}_*"
  local run_dir=""
  local ema_path=""
  local core_std=""
  run_dir="$(newest_match "${exp_pattern}" 2>/dev/null || true)"
  if [[ -n "${run_dir}" ]]; then
    ema_path="$(newest_match "${run_dir}/checkpoints/ema_*.pth" 2>/dev/null || true)"
    [[ -f "${run_dir}/core_mean_std.mat" ]] && core_std="${run_dir}/core_mean_std.mat"
  fi

  if [[ -z "${ema_path}" || -z "${core_std}" ]]; then
    python3 train_GPSD.py \
      --expr "${EXPR_NAME}" \
      --dataset "${dataset_name}" \
      --core_path "${core_path}" \
      --spatial_dims 3 \
      --img_size_3d 4 8 8 \
      --seed "$((231 + repeat))" \
      --train_batch_size 16 \
      --num_steps 4000 \
      --learning_rate 0.0002 \
      --accumulation_steps 2 \
      --save_model_iters 1000 \
      --warmup 1000 \
      --save_signals_step 0 \
      --total_steps 20 \
      --manifest_path "${subset_manifest}" \
      --sample_weight_mode "${WEIGHT_MODE}"
    run_dir="$(newest_match "${exp_pattern}")"
    ema_path="$(newest_match "${run_dir}/checkpoints/ema_*.pth")"
    core_std="${run_dir}/core_mean_std.mat"
  fi

  if [[ ! -f "${agg}" ]]; then
    python3 run_holdout_reconstruction_eval.py \
      --repo_root . \
      --test_h5 "${TEST_H5}" \
      --test_manifest "${TEST_MANIFEST}" \
      --metadata_path "${META}" \
      --sensor_csv "${SENSOR}" \
      --basis_path "${basis_path}" \
      --model_path "${ema_path}" \
      --core_mean_std_path "${core_std}" \
      --output_dir "${eval_out}" \
      --dataset_prefix "${dataset_name}" \
      --spatial_dims 3 \
      --img_size_3d 4 8 8 \
      --mpdps_weight 16 \
      --obs_rho 0.01 \
      --zeta 0.03 \
      --obs_injection_mode direct_inner \
      --obs_scale_schedule constant \
      --obs_scale_blend replace \
      --obs_target_ratio 0.25 \
      --obs_min_scale 0.0 \
      --obs_max_scale 500.0 \
      --obs_inner_steps 3 \
      --obs_inner_decay 0.7 \
      --missing_type 1 \
      --num_posterior_samples 1 \
      --total_steps 20 \
      --truth_threshold 1e-5 \
      --cleanup_recon_mat \
      --cleanup_sensor_cache
  fi

  echo "[$(date)] done ${dataset_name}"
  rm -rf "${lock_dir}"
  trap - RETURN
}

for repeat in "${REPEATS[@]}"; do
  run_subset "${SIZE}" "${repeat}"
done
