#!/usr/bin/env bash
set -euo pipefail
cd /root/Hydrogen-leak/SDIFT_model
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export PIP_TRUSTED_HOST=mirrors.aliyun.com
export PIP_DEFAULT_TIMEOUT=600
python3 -m pip install --upgrade pip
python3 -m pip install -v --no-cache-dir --find-links https://mirrors.aliyun.com/pytorch-wheels/cu121/ "torch==2.5.1+cu121"
python3 -m pip install -v --no-cache-dir h5py==3.13.0 matplotlib==3.10.3 numpy==2.3.0 pandas==2.3.0 scipy==1.15.3 tqdm==4.67.1
python3 - <<"PY"
import torch, h5py, numpy, scipy, matplotlib, pandas, tqdm
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
with h5py.File('data/cfd16_all_T120_interp48.h5', 'r') as f:
    key='data' if 'data' in f else list(f.keys())[0]
    print('shape', f[key].shape)
PY
