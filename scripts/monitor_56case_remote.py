import os
import re
import sys
import time
from datetime import datetime

import paramiko


HOST = os.environ["REMOTE_HOST"]
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ["REMOTE_USER"]
PASSWORD = os.environ["REMOTE_PASSWORD"]
REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/hy-tmp/SDIFT_model56")
POLL_SECONDS = int(os.environ.get("MONITOR_POLL_SECONDS", "120"))
LOG_PATH = os.environ.get("MONITOR_LOG_PATH", os.path.join(os.getcwd(), "tmp", "monitor_56case_remote.log"))


STUDIES = [
    {
        "name": "cfd56_scale_none",
        "pattern": "--study_tag cfd56_scale_none",
        "output_root": "results/advisor_study/train_scale_none_cfd56",
        "sample_weight_mode": "none",
        "preferred_gpu": 1,
    },
    {
        "name": "cfd56_scale_focus",
        "pattern": "--study_tag cfd56_scale_focus",
        "output_root": "results/advisor_study/train_scale_lowflow_focus_cfd56",
        "sample_weight_mode": "lowflow_focus_v1",
        "preferred_gpu": 2,
    },
    {
        "name": "cfd56_scale_balanced",
        "pattern": "--study_tag cfd56_scale_balanced",
        "output_root": "results/advisor_study/train_scale_lowflow_balanced_cfd56",
        "sample_weight_mode": "lowflow_balanced_v1",
        "preferred_gpu": 3,
    },
]


def log(msg: str):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    return client


def run(client, cmd: str):
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    return out, err


def current_ps(client):
    out, err = run(
        client,
        r"""bash -lc 'ps -ef | grep -E "run_training_scale_repeated_study|train_GPSD|train_FTM|run_holdout_reconstruction_eval" | grep -v grep || true'""",
    )
    return out


def current_gpus(client):
    out, err = run(
        client,
        r"""bash -lc 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true'""",
    )
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "memory_used": int(parts[1]),
                    "utilization": int(parts[2]),
                }
            )
    return gpus


def choose_gpu(gpus, preferred):
    for gpu in gpus:
        if gpu["index"] == preferred and gpu["memory_used"] < 1000:
            return gpu["index"]
    idle = [g["index"] for g in gpus if g["memory_used"] < 1000]
    if idle:
        return idle[0]
    return None


def launch_study(client, study, gpu_index):
    cmd = (
        "bash -lc 'cd {root} && "
        "nohup env CUDA_VISIBLE_DEVICES={gpu} python3 run_training_scale_repeated_study.py "
        "--repo_root . "
        "--train_h5 data/splits/holdout_400_0_0_val_0200/train.h5 "
        "--train_manifest data/splits/holdout_400_0_0_val_0200/train_manifest.csv "
        "--test_h5 data/splits/holdout_400_0_0_val_0200/test.h5 "
        "--test_manifest data/splits/holdout_400_0_0_val_0200/test_manifest.csv "
        "--metadata_path data/cfd56_all_T120_interp48_meta.npy "
        "--sensor_csv data/sensors_real_12.csv "
        "--train_sizes 6 12 24 42 "
        "--repeats 3 "
        "--output_root {output_root} "
        "--study_tag {study_tag} "
        "--sample_weight_modes {weight_mode} "
        "--mpdps_weight 16 --obs_rho 0.01 --zeta 0.03 "
        "--obs_injection_mode direct_inner "
        "--obs_scale_schedule constant "
        "--obs_scale_blend replace "
        "--obs_target_ratio 0.25 "
        "--obs_min_scale 0.0 "
        "--obs_max_scale 500.0 "
        "--obs_inner_steps 3 "
        "--obs_inner_decay 0.7 "
        "> logs/{study_tag}_monitor_restart.log 2>&1 < /dev/null & echo restarted'"
    ).format(
        root=REMOTE_ROOT,
        gpu=gpu_index,
        output_root=study["output_root"],
        study_tag=study["name"],
        weight_mode=study["sample_weight_mode"],
    )
    out, err = run(client, cmd)
    return out.strip(), err.strip()


def main():
    log("monitor started")
    while True:
        try:
            client = connect()
            ps = current_ps(client)
            gpus = current_gpus(client)
            for study in STUDIES:
                if study["pattern"] in ps:
                    continue
                gpu_index = choose_gpu(gpus, study["preferred_gpu"])
                if gpu_index is None:
                    log(f"{study['name']}: missing, but no idle GPU available; will retry")
                    continue
                out, err = launch_study(client, study, gpu_index)
                log(f"{study['name']}: relaunched on GPU {gpu_index}; out={out} err={err}")
            client.close()
        except Exception as e:
            log(f"monitor exception: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
