import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict


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


def newest_match(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no files matched: {pattern}")
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def maybe_newest_match(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return ""
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_group(rows, metric_keys):
    summary = {
        "count": len(rows),
    }
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            summary[key] = {"mean": mean, "std": var ** 0.5}
        else:
            summary[key] = {"mean": None, "std": None}
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run repeated train-size scaling with stratified subsets and weight-mode comparison.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--train_h5", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--train_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--study_tag", default="scale_study_repeat")
    parser.add_argument("--spatial_dims", type=int, default=3)
    parser.add_argument("--img_size_3d", type=int, nargs=3, default=[4, 8, 8])
    parser.add_argument("--ftm_batch_size", type=int, default=4)
    parser.add_argument("--ftm_max_iter", type=int, default=1200)
    parser.add_argument("--ftm_learning_rate", type=float, default=2e-4)
    parser.add_argument("--gpsd_train_batch_size", type=int, default=16)
    parser.add_argument("--gpsd_num_steps", type=int, default=4000)
    parser.add_argument("--gpsd_learning_rate", type=float, default=2e-4)
    parser.add_argument("--gpsd_accumulation_steps", type=int, default=2)
    parser.add_argument("--gpsd_save_model_iters", type=int, default=1000)
    parser.add_argument("--gpsd_warmup", type=int, default=1000)
    parser.add_argument("--gpsd_save_signals_step", type=int, default=0)
    parser.add_argument("--eval_total_steps", type=int, default=20)
    parser.add_argument("--num_posterior_samples", type=int, default=1)
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
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    parser.add_argument("--low_rates", type=float, nargs="+", default=[50.0, 100.0])
    parser.add_argument(
        "--sample_weight_modes",
        nargs="+",
        default=["none", "lowflow_focus_v1", "lowflow_balanced_v1"],
        choices=["none", "balanced_by_rate", "lowflow_focus_v1", "lowflow_balanced_v1"],
    )
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    subset_root = os.path.join(output_root, "subsets")
    run_cmd(
        [
            sys.executable,
            "build_repeated_train_size_subsets.py",
            "--input_h5",
            args.train_h5,
            "--manifest_csv",
            args.train_manifest,
            "--out_dir",
            subset_root,
            "--sizes",
            *(str(x) for x in sorted(set(args.train_sizes))),
            "--repeats",
            str(args.repeats),
            "--low_rates",
            *(str(x) for x in args.low_rates),
            "--seed",
            str(args.seed),
            "--prefix",
            "train",
        ],
        cwd=repo_root,
        dry_run=args.dry_run,
    )

    low_group_name = LOW_GROUP_PREFIX + "_".join(
        str(int(x)) if float(x).is_integer() else str(x) for x in sorted(set(float(x) for x in args.low_rates))
    )

    detailed_rows = []
    for sample_weight_mode in args.sample_weight_modes:
        for size in sorted(set(int(x) for x in args.train_sizes)):
            for repeat in range(int(args.repeats)):
                subset_dir = os.path.join(subset_root, f"size_{size:03d}", f"repeat_{repeat:02d}")
                subset_h5 = os.path.join(subset_dir, f"train_{size:03d}_r{repeat:02d}.h5")
                subset_manifest = os.path.join(subset_dir, f"train_{size:03d}_r{repeat:02d}_manifest.csv")
                dataset_name = f"{args.study_tag}_{sample_weight_mode}_n{size:03d}_r{repeat:02d}"
                expr_name = f"{args.study_tag}_{sample_weight_mode}_gpsd"

                row = {
                    "sample_weight_mode": sample_weight_mode,
                    "train_size": int(size),
                    "repeat": int(repeat),
                    "subset_h5": subset_h5,
                    "subset_manifest": subset_manifest,
                }

                out_dir_pattern = os.path.join(repo_root, "exps", f"{expr_name}_{dataset_name}_*")
                eval_out_dir = os.path.join(output_root, f"{sample_weight_mode}_n{size:03d}_r{repeat:02d}")
                os.makedirs(eval_out_dir, exist_ok=True)

                ftm_t0 = time.time()
                core_pattern = os.path.join(repo_root, "data", f"core_{dataset_name}_4x8x8_*_last.mat")
                basis_pattern = os.path.join(repo_root, "ckp", f"basis_{dataset_name}_4x8x8_*_last.pth")
                core_path = maybe_newest_match(core_pattern)
                basis_path = maybe_newest_match(basis_pattern)
                if not (core_path and basis_path):
                    run_cmd(
                        [
                            sys.executable,
                            "train_FTM.py",
                            "--data_name",
                            dataset_name,
                            "--data_path",
                            subset_h5,
                            "--metadata_path",
                            args.metadata_path,
                            "--R",
                            *(str(x) for x in args.img_size_3d),
                            "--batch_size",
                            str(args.ftm_batch_size),
                            "--learning_rate",
                            str(args.ftm_learning_rate),
                            "--max_iter",
                            str(args.ftm_max_iter),
                            "--seed",
                            str(args.seed + repeat),
                            "--manifest_path",
                            subset_manifest,
                            "--sample_weight_mode",
                            sample_weight_mode,
                            "--save_last",
                        ],
                        cwd=repo_root,
                        dry_run=args.dry_run,
                    )
                    core_path = newest_match(core_pattern)
                    basis_path = newest_match(basis_pattern)
                row["ftm_wall_seconds"] = time.time() - ftm_t0
                row["core_path"] = core_path
                row["basis_path"] = basis_path

                gpsd_t0 = time.time()
                exps_before = set(glob.glob(out_dir_pattern))
                run_dir = maybe_newest_match(out_dir_pattern)
                ema_path = ""
                core_mean_std_path = ""
                if run_dir:
                    ema_path = maybe_newest_match(os.path.join(run_dir, "checkpoints", "ema_*.pth"))
                    core_mean_std_path = os.path.join(run_dir, "core_mean_std.mat")
                    if not os.path.isfile(core_mean_std_path):
                        core_mean_std_path = ""
                if not (ema_path and core_mean_std_path):
                    run_cmd(
                        [
                            sys.executable,
                            "train_GPSD.py",
                            "--expr",
                            expr_name,
                            "--dataset",
                            dataset_name,
                            "--core_path",
                            core_path,
                            "--spatial_dims",
                            str(args.spatial_dims),
                            "--img_size_3d",
                            *(str(x) for x in args.img_size_3d),
                            "--seed",
                            str(args.seed + repeat),
                            "--train_batch_size",
                            str(args.gpsd_train_batch_size),
                            "--num_steps",
                            str(args.gpsd_num_steps),
                            "--learning_rate",
                            str(args.gpsd_learning_rate),
                            "--accumulation_steps",
                            str(args.gpsd_accumulation_steps),
                            "--save_model_iters",
                            str(args.gpsd_save_model_iters),
                            "--warmup",
                            str(args.gpsd_warmup),
                            "--save_signals_step",
                            str(args.gpsd_save_signals_step),
                            "--total_steps",
                            str(args.eval_total_steps),
                            "--manifest_path",
                            subset_manifest,
                            "--sample_weight_mode",
                            sample_weight_mode,
                        ],
                        cwd=repo_root,
                        dry_run=args.dry_run,
                    )
                    exps_after = set(glob.glob(out_dir_pattern))
                    run_dirs = sorted(exps_after if exps_after else exps_before)
                    if not run_dirs:
                        raise FileNotFoundError(f"no experiment dir found for pattern: {out_dir_pattern}")
                    run_dir = run_dirs[-1]
                    ema_path = newest_match(os.path.join(run_dir, "checkpoints", "ema_*.pth"))
                    core_mean_std_path = os.path.join(run_dir, "core_mean_std.mat")
                row["gpsd_wall_seconds"] = time.time() - gpsd_t0
                row["gpsd_run_dir"] = run_dir
                row["model_path"] = ema_path
                row["core_mean_std_path"] = core_mean_std_path

                eval_t0 = time.time()
                aggregate_json = os.path.join(eval_out_dir, "aggregate_metrics.json")
                if not os.path.isfile(aggregate_json):
                    run_cmd(
                        [
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
                            basis_path,
                            "--model_path",
                            ema_path,
                            "--core_mean_std_path",
                            core_mean_std_path,
                            "--output_dir",
                            eval_out_dir,
                            "--dataset_prefix",
                            dataset_name,
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
                            str(args.eval_total_steps),
                            "--truth_threshold",
                            str(args.truth_threshold),
                        ],
                        cwd=repo_root,
                        dry_run=args.dry_run,
                    )
                row["eval_wall_seconds"] = time.time() - eval_t0
                row["output_dir"] = eval_out_dir

                if not args.dry_run:
                    payload = load_json(aggregate_json)
                    row["num_test_cases"] = int(payload["count"])
                    for key in METRIC_KEYS:
                        metric = payload["metrics"].get(key, {})
                        row[key] = metric.get("mean")
                    low_block = payload.get("subgroups", {}).get(low_group_name, {})
                    row["low_rates_label"] = ",".join(
                        str(int(x)) if float(x).is_integer() else str(x) for x in sorted(set(float(x) for x in args.low_rates))
                    )
                    row["low_rates_count"] = low_block.get("count")
                    for key in METRIC_KEYS:
                        metric = low_block.get("metrics", {}).get(key, {})
                        row[f"low_{key}"] = metric.get("mean")

                detailed_rows.append(row)

    detail_json = os.path.join(output_root, "training_scale_repeated_detail.json")
    detail_csv = os.path.join(output_root, "training_scale_repeated_detail.csv")
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump({"rows": detailed_rows}, f, ensure_ascii=False, indent=2)
    detail_fields = [
        "sample_weight_mode",
        "train_size",
        "repeat",
        "num_test_cases",
        "low_rates_label",
        "low_rates_count",
        "ftm_wall_seconds",
        "gpsd_wall_seconds",
        "eval_wall_seconds",
        "subset_h5",
        "subset_manifest",
        "core_path",
        "basis_path",
        "gpsd_run_dir",
        "model_path",
        "core_mean_std_path",
        "output_dir",
    ]
    for key in METRIC_KEYS:
        detail_fields.append(key)
    for key in METRIC_KEYS:
        detail_fields.append(f"low_{key}")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detailed_rows)

    grouped = defaultdict(list)
    for row in detailed_rows:
        grouped[(row["sample_weight_mode"], int(row["train_size"]))].append(row)

    summary_rows = []
    metrics_for_summary = METRIC_KEYS + [f"low_{key}" for key in METRIC_KEYS]
    for (sample_weight_mode, train_size), rows in sorted(grouped.items()):
        summary = {
            "sample_weight_mode": sample_weight_mode,
            "train_size": train_size,
            "num_repeats": len(rows),
            "low_rates_label": rows[0].get("low_rates_label"),
            "low_rates_count_mean": sum(float(r.get("low_rates_count") or 0) for r in rows) / len(rows),
        }
        for key in ["ftm_wall_seconds", "gpsd_wall_seconds", "eval_wall_seconds"] + metrics_for_summary:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = var ** 0.5
            else:
                summary[f"{key}_mean"] = None
                summary[f"{key}_std"] = None
        summary_rows.append(summary)

    summary_json = os.path.join(output_root, "training_scale_repeated_summary.json")
    summary_csv = os.path.join(output_root, "training_scale_repeated_summary.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"rows": summary_rows}, f, ensure_ascii=False, indent=2)
    summary_fields = ["sample_weight_mode", "train_size", "num_repeats", "low_rates_label", "low_rates_count_mean"]
    for key in ["ftm_wall_seconds", "gpsd_wall_seconds", "eval_wall_seconds"] + metrics_for_summary:
        summary_fields.extend([f"{key}_mean", f"{key}_std"])
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"saved: {detail_json}")
    print(f"saved: {detail_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()
