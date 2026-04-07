#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
cd /root/Hydrogen-leak/SDIFT_model
mkdir -p logs ckp exps results

echo "[$(date '+%F %T')] Starting FTM training"
python3 train_FTM.py   --data_name cfd16_t120_run1   --data_path ./data/cfd16_all_T120_interp48.h5   --metadata_path ./data/cfd16_all_T120_interp48_meta.npy   --batch_size 1   --R 4 8 8   --learning_rate 2e-4   --max_iter 2000   --save_last   >> ./logs/ftm_train.log 2>&1

CORE_PATH=$(ls -t ./data/core_cfd16_t120_run1_4x8x8_*_last.mat | head -n1)
if [ -z "${CORE_PATH:-}" ]; then
  echo "[$(date '+%F %T')] ERROR: FTM core file not found" | tee -a ./logs/pipeline.log
  exit 1
fi

echo "[$(date '+%F %T')] Using core: $CORE_PATH" | tee -a ./logs/pipeline.log

echo "[$(date '+%F %T')] Starting GPSD training" | tee -a ./logs/pipeline.log
python3 train_GPSD.py   --expr exps   --dataset cfd16_t120_run1   --core_path "$CORE_PATH"   --spatial_dims 3   --img_size_3d 4 8 8   --train_batch_size 8   --num_steps 8000   --learning_rate 2e-4   --accumulation_steps 1   --save_model_iters 1000   --log_step 50   --warmup 1000   --save_signals_step 0   --total_steps 30   >> ./logs/gpsd_train.log 2>&1

echo "[$(date '+%F %T')] Training pipeline finished" | tee -a ./logs/pipeline.log
