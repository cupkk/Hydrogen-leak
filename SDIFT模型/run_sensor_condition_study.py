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
LOW_GROUP_PREFIX = "low_rates_"


def run_cmd(cmd, cwd, dry_run=False):
    print("RUN:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a sensor-count and observation-horizon study on a fixed test split.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True, help="Sensor pool CSV. Nested subsets are derived from this file.")
    parser.add_argument("--sensor_counts", type=int, nargs="+", required=True)
    parser.add_argument("--observed_time_steps", type=int, nargs="+", required=True, help="Use 0 for the full time horizon.")
    parser.add_argument("--basis_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--core_mean_std_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_prefix", default="sensor_study")
    parser.add_argument("--sensor_prefix", default="sensor_subset")
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
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    parser.add_argument("--low_rates", type=float, nargs="+", default=[50.0, 100.0])
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    sensor_dir = os.path.join(output_root, "sensor_subsets")
    run_cmd(
        [
            sys.executable,
            "make_nested_sensor_subsets.py",
            "--sensor_csv",
            args.sensor_csv,
            "--counts",
            *(str(x) for x in args.sensor_counts),
            "--out_dir",
            sensor_dir,
            "--prefix",
            args.sensor_prefix,
            "--deduplicate",
        ],
        cwd=repo_root,
        dry_run=args.dry_run,
    )

    rows = []
    low_group_name = LOW_GROUP_PREFIX + "_".join(
        str(int(x)) if float(x).is_integer() else str(x) for x in sorted(set(float(x) for x in args.low_rates))
    )
    sensor_counts = sorted(set(int(x) for x in args.sensor_counts))
    obs_steps_list = sorted(set(int(x) for x in args.observed_time_steps))
    for sensor_count in sensor_counts:
        sensor_csv = os.path.join(sensor_dir, f"{args.sensor_prefix}_{sensor_count}.csv")
        for obs_steps in obs_steps_list:
            obs_label = "full" if obs_steps <= 0 else f"{obs_steps:03d}"
            condition_name = f"s{sensor_count:02d}_t{obs_label}"
            out_dir = os.path.join(output_root, condition_name)
            os.makedirs(out_dir, exist_ok=True)

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
                sensor_csv,
                "--basis_path",
                args.basis_path,
                "--model_path",
                args.model_path,
                "--core_mean_std_path",
                args.core_mean_std_path,
                "--output_dir",
                out_dir,
                "--dataset_prefix",
                f"{args.dataset_prefix}_{condition_name}",
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
                "--truth_threshold",
                str(args.truth_threshold),
                "--observed_time_steps",
                str(obs_steps),
            ]

            t0 = time.time()
            run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)
            wall_seconds = time.time() - t0

            row = {
                "condition": condition_name,
                "sensor_count": int(sensor_count),
                "observed_time_steps": int(obs_steps),
                "sensor_csv": sensor_csv,
                "output_dir": out_dir,
                "wall_seconds": wall_seconds,
            }
            if not args.dry_run:
                aggregate_json = os.path.join(out_dir, "aggregate_metrics.json")
                payload = load_json(aggregate_json)
                row["num_test_cases"] = int(payload["count"])
                run_summary = load_json(os.path.join(out_dir, "run_summary.json"))
                timing_summary = run_summary.get("timing_summary", {})
                row["sensor_seconds_mean"] = timing_summary.get("sensor_seconds", {}).get("mean")
                row["reconstruction_seconds_mean"] = timing_summary.get("reconstruction_seconds", {}).get("mean")
                row["evaluation_seconds_mean"] = timing_summary.get("evaluation_seconds", {}).get("mean")
                for key in METRIC_KEYS:
                    metric = payload["metrics"].get(key, {})
                    row[f"{key}_mean"] = metric.get("mean")
                    row[f"{key}_std"] = metric.get("std")
                low_block = payload.get("subgroups", {}).get(low_group_name, {})
                row["low_rates_label"] = ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in sorted(set(float(x) for x in args.low_rates)))
                row["low_rates_count"] = low_block.get("count")
                for key in METRIC_KEYS:
                    metric = low_block.get("metrics", {}).get(key, {})
                    row[f"low_{key}_mean"] = metric.get("mean")
                    row[f"low_{key}_std"] = metric.get("std")
            rows.append(row)

    summary_json = os.path.join(output_root, "sensor_condition_study.json")
    summary_csv = os.path.join(output_root, "sensor_condition_study.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sensor_counts": sensor_counts,
                "observed_time_steps": obs_steps_list,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    fieldnames = [
        "condition",
        "sensor_count",
        "observed_time_steps",
        "num_test_cases",
        "low_rates_label",
        "low_rates_count",
        "wall_seconds",
        "sensor_seconds_mean",
        "reconstruction_seconds_mean",
        "evaluation_seconds_mean",
        "sensor_csv",
        "output_dir",
    ]
    for key in METRIC_KEYS:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    for key in METRIC_KEYS:
        fieldnames.extend([f"low_{key}_mean", f"low_{key}_std"])
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {summary_json}")
    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()
