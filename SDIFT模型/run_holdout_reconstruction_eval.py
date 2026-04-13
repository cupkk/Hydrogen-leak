import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

from make_sensor_observations import load_meta, nearest_index, parse_sensor_csv


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    for row in rows:
        row["data_index"] = int(row["data_index"])
        for key in ["source_x_mm", "source_y_mm", "source_z_mm", "leak_rate_ml_min"]:
            if key in row and row[key] != "":
                row[key] = int(float(row[key]))
    return rows


def run_cmd(cmd, cwd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_wrong_sensor_geometry(sensor_csv, metadata_path):
    _, _, _, _, u_real, v_real, w_real, _ = load_meta(metadata_path)
    sensor_xyz_real = parse_sensor_csv(sensor_csv)
    if sensor_xyz_real.shape[1] > 3:
        sensor_xyz_real = sensor_xyz_real[:, :3]

    wrong = sensor_xyz_real.copy()
    wrong[:, 0] = (float(u_real.min()) + float(u_real.max())) - wrong[:, 0]
    wrong[:, 1] = (float(v_real.min()) + float(v_real.max())) - wrong[:, 1]
    wrong[:, 2] = (float(w_real.min()) + float(w_real.max())) - wrong[:, 2]
    wrong[:, 0] = np.clip(wrong[:, 0], float(u_real.min()), float(u_real.max()))
    wrong[:, 1] = np.clip(wrong[:, 1], float(v_real.min()), float(v_real.max()))
    wrong[:, 2] = np.clip(wrong[:, 2], float(w_real.min()), float(w_real.max()))

    u_uni, v_uni, w_uni, _, _, _, _, _ = load_meta(metadata_path)
    sensor_xyz_uni = np.stack(
        [
            np.interp(wrong[:, 0], u_real, u_uni),
            np.interp(wrong[:, 1], v_real, v_uni),
            np.interp(wrong[:, 2], w_real, w_uni),
        ],
        axis=1,
    ).astype(np.float32)
    sensor_idx = np.empty((wrong.shape[0], 3), dtype=np.int64)
    for i, (x, y, z) in enumerate(wrong):
        sensor_idx[i, 0] = nearest_index(u_real, x)
        sensor_idx[i, 1] = nearest_index(v_real, y)
        sensor_idx[i, 2] = nearest_index(w_real, z)
    return wrong.astype(np.float32), sensor_xyz_uni, sensor_idx


def corrupt_sensor_observation(sensor_npy_path, mode, metadata_path, sensor_csv, seed, wrong_sensor_csv=""):
    if mode == "correct":
        return sensor_npy_path

    payload = np.load(sensor_npy_path, allow_pickle=True).item()
    rng = np.random.default_rng(seed)
    y = np.asarray(payload["y"], dtype=np.float32)

    if mode == "shuffled":
        shuffled = y.reshape(-1).copy()
        rng.shuffle(shuffled)
        payload["y"] = shuffled.reshape(y.shape)
    elif mode == "zeros":
        payload["y"] = np.zeros_like(y, dtype=np.float32)
    elif mode == "wrong_positions":
        csv_path = wrong_sensor_csv or sensor_csv
        wrong_real, wrong_uni, wrong_idx = build_wrong_sensor_geometry(csv_path, metadata_path)
        payload["sensor_xyz_real"] = wrong_real
        payload["sensor_xyz"] = wrong_uni
        payload["sensor_idx"] = wrong_idx
    else:
        raise ValueError(f"unsupported sensor_input_mode: {mode}")

    base, ext = os.path.splitext(sensor_npy_path)
    out_path = base + f"_{mode}" + ext
    np.save(out_path, payload)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run sensor extraction, reconstruction, and evaluation for a held-out test split.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--basis_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--core_mean_std_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_prefix", required=True)
    parser.add_argument("--spatial_dims", type=int, default=3)
    parser.add_argument("--img_size_3d", type=int, nargs=3, default=[4, 8, 8])
    parser.add_argument("--mpdps_weight", type=float, default=0.4)
    parser.add_argument("--obs_rho", type=float, default=0.01)
    parser.add_argument("--zeta", type=float, default=0.009)
    parser.add_argument("--obs_injection_mode", choices=["legacy", "adaptive_ratio", "direct_inner"], default="legacy")
    parser.add_argument("--obs_scale_schedule", choices=["legacy_decay", "constant"], default="legacy_decay")
    parser.add_argument("--obs_scale_blend", choices=["max", "replace"], default="max")
    parser.add_argument("--obs_target_ratio", type=float, default=0.25)
    parser.add_argument("--obs_min_scale", type=float, default=0.0)
    parser.add_argument("--obs_max_scale", type=float, default=500.0)
    parser.add_argument("--obs_inner_steps", type=int, default=3)
    parser.add_argument("--obs_inner_decay", type=float, default=0.7)
    parser.add_argument("--missing_type", type=int, default=1)
    parser.add_argument("--num_posterior_samples", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=20)
    parser.add_argument("--observed_time_steps", type=int, default=0)
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    parser.add_argument(
        "--sensor_input_mode",
        choices=["correct", "shuffled", "zeros", "wrong_positions"],
        default="correct",
    )
    parser.add_argument("--wrong_sensor_csv", default="")
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--cleanup_recon_mat", action="store_true", default=False)
    parser.add_argument("--cleanup_sensor_cache", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sensor_dir = os.path.join(output_dir, "sensors")
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(sensor_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    rows = load_manifest(args.test_manifest)
    eval_jsons = []
    case_results = []

    for row in rows:
        idx = row["data_index"]
        case_id = row.get("case_id", f"case_{idx:04d}")
        raw_case_name = row.get("raw_case_name", case_id)
        dataset_name = f"{args.dataset_prefix}_{case_id}"
        sensor_out = os.path.join(sensor_dir, f"{case_id}_sensors.npy")
        timings = {}

        t0 = time.time()
        run_cmd(
            [
                sys.executable,
                "make_sensor_observations.py",
                "--field_h5",
                args.test_h5,
                "--metadata_path",
                args.metadata_path,
                "--out",
                sensor_out,
                "--sensor_csv",
                args.sensor_csv,
                "--sample_index",
                str(idx),
            ],
            cwd=repo_root,
        )
        sensor_for_recon = corrupt_sensor_observation(
            sensor_out,
            mode=args.sensor_input_mode,
            metadata_path=args.metadata_path,
            sensor_csv=args.sensor_csv,
            seed=args.seed + idx,
            wrong_sensor_csv=args.wrong_sensor_csv,
        )
        timings["sensor_seconds"] = time.time() - t0

        t0 = time.time()
        run_cmd(
            [
                sys.executable,
                "message_passing_DPS.py",
                "--dataset",
                dataset_name,
                "--spatial_dims",
                str(args.spatial_dims),
                "--img_size_3d",
                *(str(x) for x in args.img_size_3d),
                "--metadata_path",
                args.metadata_path,
                "--sensor_path",
                sensor_for_recon,
                "--core_mean_std_path",
                args.core_mean_std_path,
                "--basis_path",
                args.basis_path,
                "--model_path",
                args.model_path,
                "--mpdps_weight",
                str(args.mpdps_weight),
                "--obs_rho",
                str(args.obs_rho),
                "--zeta",
                str(args.zeta),
                "--obs_injection_mode",
                args.obs_injection_mode,
                "--obs_scale_schedule",
                args.obs_scale_schedule,
                "--obs_scale_blend",
                args.obs_scale_blend,
                "--obs_target_ratio",
                str(args.obs_target_ratio),
                "--obs_min_scale",
                str(args.obs_min_scale),
                "--obs_max_scale",
                str(args.obs_max_scale),
                "--obs_inner_steps",
                str(args.obs_inner_steps),
                "--obs_inner_decay",
                str(args.obs_inner_decay),
                "--missing_type",
                str(args.missing_type),
                "--num_posterior_samples",
                str(args.num_posterior_samples),
                "--total_steps",
                str(args.total_steps),
                "--observed_time_steps",
                str(args.observed_time_steps),
            ],
            cwd=repo_root,
        )
        timings["reconstruction_seconds"] = time.time() - t0

        result_name = f"{dataset_name}_mpdps_{args.mpdps_weight}_recon_rho{args.obs_rho}_mode_{args.missing_type}"
        recon_mat = os.path.join(repo_root, "results", result_name + ".mat")
        eval_json = os.path.join(eval_dir, f"{case_id}_eval.json")
        eval_csv = os.path.join(eval_dir, f"{case_id}_per_time.csv")

        t0 = time.time()
        run_cmd(
            [
                sys.executable,
                "evaluate_reconstruction.py",
                "--recon_mat",
                recon_mat,
                "--truth_path",
                args.test_h5,
                "--meta",
                args.metadata_path,
                "--recon_sample_index",
                "0",
                "--truth_sample_index",
                str(idx),
                "--out_json",
                eval_json,
                "--out_csv",
                eval_csv,
                "--truth_threshold",
                str(args.truth_threshold),
            ],
            cwd=repo_root,
        )
        timings["evaluation_seconds"] = time.time() - t0

        eval_jsons.append(eval_json)
        case_results.append(
            {
                "case_id": case_id,
                "raw_case_name": raw_case_name,
                "data_index": idx,
                "sensor_path": sensor_out,
                "sensor_input_mode": args.sensor_input_mode,
                "sensor_used_path": sensor_for_recon,
                "zeta": float(args.zeta),
                "obs_injection_mode": args.obs_injection_mode,
                "obs_scale_schedule": args.obs_scale_schedule,
                "obs_scale_blend": args.obs_scale_blend,
                "obs_target_ratio": float(args.obs_target_ratio),
                "obs_min_scale": float(args.obs_min_scale),
                "obs_max_scale": float(args.obs_max_scale),
                "obs_inner_steps": int(args.obs_inner_steps),
                "obs_inner_decay": float(args.obs_inner_decay),
                "recon_mat": recon_mat,
                "eval_json": eval_json,
                "eval_csv": eval_csv,
                "timings": timings,
            }
        )

        if args.cleanup_recon_mat:
            cleanup_paths = [
                recon_mat,
                os.path.join(repo_root, "results", result_name + "_summary.json"),
                os.path.join(repo_root, "results", result_name + "_source_est.npy"),
            ]
            for path in cleanup_paths:
                if os.path.exists(path):
                    os.remove(path)

        if args.cleanup_sensor_cache:
            cleanup_paths = [sensor_out]
            if sensor_for_recon != sensor_out:
                cleanup_paths.append(sensor_for_recon)
            for path in cleanup_paths:
                if os.path.exists(path):
                    os.remove(path)

    aggregate_json = os.path.join(output_dir, "aggregate_metrics.json")
    aggregate_csv = os.path.join(output_dir, "aggregate_metrics.csv")
    cmd = [
        sys.executable,
        "aggregate_reconstruction_metrics.py",
        "--manifest_csv",
        args.test_manifest,
        "--out_json",
        aggregate_json,
        "--out_csv",
        aggregate_csv,
    ]
    for path in eval_jsons:
        cmd.extend(["--eval_json", path])
    run_cmd(cmd, cwd=repo_root)

    timing_summary = {}
    if case_results:
        for key in ["sensor_seconds", "reconstruction_seconds", "evaluation_seconds"]:
            vals = [float(item["timings"].get(key, 0.0)) for item in case_results]
            timing_summary[key] = {
                "mean": float(sum(vals) / len(vals)),
                "max": float(max(vals)),
                "min": float(min(vals)),
            }

    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_prefix": args.dataset_prefix,
                "num_cases": len(case_results),
                "observed_time_steps": int(args.observed_time_steps),
                "cases": case_results,
                "timing_summary": timing_summary,
                "aggregate_json": aggregate_json,
                "aggregate_csv": aggregate_csv,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved: {aggregate_json}")
    print(f"saved: {aggregate_csv}")


if __name__ == "__main__":
    main()
