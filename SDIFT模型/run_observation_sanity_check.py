import argparse
import csv
import json
import os
import subprocess
import sys
import time


METRIC_KEYS = [
    "global_rmse",
    "global_mae",
    "global_rel_l1_mean",
    "global_rel_l1_active_mean",
    "global_rel_l2",
    "mass_mean_rel_error",
]


def run_cmd(cmd, cwd, dry_run=False):
    print("RUN:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check whether MPDPS reconstruction meaningfully uses sensor observations.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--basis_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--core_mean_std_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_prefix", default="obs_sanity")
    parser.add_argument("--modes", default="correct,shuffled,zeros,wrong_positions")
    parser.add_argument("--wrong_sensor_csv", default="")
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
    parser.add_argument("--observed_time_steps", type=int, default=120)
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--cleanup_recon_mat", action="store_true", default=False)
    parser.add_argument("--cleanup_sensor_cache", action="store_true", default=False)
    parser.add_argument("--skip_existing_modes", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    rows = []
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    for mode in modes:
        condition_dir = os.path.join(output_root, mode)
        os.makedirs(condition_dir, exist_ok=True)
        aggregate_json = os.path.join(condition_dir, "aggregate_metrics.json")
        run_summary_json = os.path.join(condition_dir, "run_summary.json")
        if args.skip_existing_modes and os.path.exists(aggregate_json) and os.path.exists(run_summary_json):
            payload = load_json(aggregate_json)
            run_summary = load_json(run_summary_json)
            timing_summary = run_summary.get("timing_summary", {})
            row = {
                "mode": mode,
                "output_dir": condition_dir,
                "wall_seconds": 0.0,
                "num_test_cases": int(payload["count"]),
                "sensor_seconds_mean": timing_summary.get("sensor_seconds", {}).get("mean"),
                "reconstruction_seconds_mean": timing_summary.get("reconstruction_seconds", {}).get("mean"),
                "evaluation_seconds_mean": timing_summary.get("evaluation_seconds", {}).get("mean"),
            }
            for key in METRIC_KEYS:
                metric = payload["metrics"].get(key, {})
                row[f"{key}_mean"] = metric.get("mean")
                row[f"{key}_std"] = metric.get("std")
            rows.append(row)
            continue
        cmd = [
            sys.executable,
            "run_holdout_reconstruction_eval.py",
            "--repo_root",
            repo_root,
            "--test_h5",
            args.test_h5,
            "--test_manifest",
            args.test_manifest,
            "--metadata_path",
            args.metadata_path,
            "--sensor_csv",
            args.sensor_csv,
            "--basis_path",
            args.basis_path,
            "--model_path",
            args.model_path,
            "--core_mean_std_path",
            args.core_mean_std_path,
            "--output_dir",
            condition_dir,
            "--dataset_prefix",
            f"{args.dataset_prefix}_{mode}",
            "--spatial_dims",
            str(args.spatial_dims),
            "--img_size_3d",
            *(str(x) for x in args.img_size_3d),
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
            "--truth_threshold",
            str(args.truth_threshold),
            "--sensor_input_mode",
            mode,
            "--seed",
            str(args.seed),
        ]
        if args.cleanup_recon_mat:
            cmd.append("--cleanup_recon_mat")
        if args.cleanup_sensor_cache:
            cmd.append("--cleanup_sensor_cache")
        if mode == "wrong_positions" and args.wrong_sensor_csv:
            cmd.extend(["--wrong_sensor_csv", args.wrong_sensor_csv])

        t0 = time.time()
        run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)
        wall_seconds = time.time() - t0

        row = {
            "mode": mode,
            "output_dir": condition_dir,
            "wall_seconds": wall_seconds,
        }
        if not args.dry_run:
            payload = load_json(aggregate_json)
            row["num_test_cases"] = int(payload["count"])
            run_summary = load_json(os.path.join(condition_dir, "run_summary.json"))
            timing_summary = run_summary.get("timing_summary", {})
            row["sensor_seconds_mean"] = timing_summary.get("sensor_seconds", {}).get("mean")
            row["reconstruction_seconds_mean"] = timing_summary.get("reconstruction_seconds", {}).get("mean")
            row["evaluation_seconds_mean"] = timing_summary.get("evaluation_seconds", {}).get("mean")
            for key in METRIC_KEYS:
                metric = payload["metrics"].get(key, {})
                row[f"{key}_mean"] = metric.get("mean")
                row[f"{key}_std"] = metric.get("std")
        rows.append(row)

    rows.sort(key=lambda r: modes.index(r["mode"]))
    out_json = os.path.join(output_root, "observation_sanity_check.json")
    out_csv = os.path.join(output_root, "observation_sanity_check.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    fieldnames = [
        "mode",
        "num_test_cases",
        "wall_seconds",
        "sensor_seconds_mean",
        "reconstruction_seconds_mean",
        "evaluation_seconds_mean",
        "output_dir",
    ]
    for key in METRIC_KEYS:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
