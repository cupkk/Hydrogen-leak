import argparse
import csv
import glob
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


def newest_match(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"no files matched: {pattern}")
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_newest_match(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return ""
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def main():
    parser = argparse.ArgumentParser(description="Run a train-size scaling study on a fixed holdout split.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--train_h5", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--train_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--study_tag", default="scale_study")
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
    parser.add_argument("--missing_type", type=int, default=1)
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    parser.add_argument("--low_rates", type=float, nargs="+", default=[50.0, 100.0])
    parser.add_argument("--sample_weight_mode", type=str, default="none", choices=["none", "balanced_by_rate", "lowflow_focus_v1", "lowflow_balanced_v1"])
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
            "build_train_size_subsets.py",
            "--input_h5",
            args.train_h5,
            "--manifest_csv",
            args.train_manifest,
            "--out_dir",
            subset_root,
            "--sizes",
            *(str(x) for x in sorted(set(args.train_sizes))),
            "--prefix",
            "train",
        ],
        cwd=repo_root,
        dry_run=args.dry_run,
    )

    rows = []
    low_group_name = LOW_GROUP_PREFIX + "_".join(
        str(int(x)) if float(x).is_integer() else str(x) for x in sorted(set(float(x) for x in args.low_rates))
    )
    for size in sorted(set(int(x) for x in args.train_sizes)):
        size_dir = os.path.join(subset_root, f"size_{size:03d}")
        subset_h5 = os.path.join(size_dir, f"train_{size:03d}.h5")
        subset_manifest = os.path.join(size_dir, f"train_{size:03d}_manifest.csv")
        dataset_name = f"{args.study_tag}_n{size:03d}"
        expr_name = f"{args.study_tag}_gpsd"

        row = {
            "train_size": int(size),
            "subset_h5": subset_h5,
            "subset_manifest": subset_manifest,
        }

        ftm_cmd = [
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
            str(args.seed),
            "--manifest_path",
            subset_manifest,
            "--sample_weight_mode",
            str(args.sample_weight_mode),
            "--save_last",
        ]

        existing_core = maybe_newest_match(
            os.path.join(
                repo_root,
                "data",
                f"core_{dataset_name}_{args.img_size_3d[0]}x{args.img_size_3d[1]}x{args.img_size_3d[2]}_*_last.mat",
            )
        )
        existing_basis = maybe_newest_match(
            os.path.join(
                repo_root,
                "ckp",
                f"basis_{dataset_name}_{args.img_size_3d[0]}x{args.img_size_3d[1]}x{args.img_size_3d[2]}_*_last.pth",
            )
        )
        if existing_core and existing_basis and not args.dry_run:
            print(f"SKIP FTM: reuse existing core/basis for {dataset_name}")
            row["ftm_seconds"] = 0.0
        else:
            t0 = time.time()
            run_cmd(ftm_cmd, cwd=repo_root, dry_run=args.dry_run)
            row["ftm_seconds"] = time.time() - t0

        if args.dry_run:
            rows.append(row)
            continue

        core_path = newest_match(os.path.join(repo_root, "data", f"core_{dataset_name}_{args.img_size_3d[0]}x{args.img_size_3d[1]}x{args.img_size_3d[2]}_*_last.mat"))
        basis_path = newest_match(os.path.join(repo_root, "ckp", f"basis_{dataset_name}_{args.img_size_3d[0]}x{args.img_size_3d[1]}x{args.img_size_3d[2]}_*_last.pth"))
        row["core_path"] = core_path
        row["basis_path"] = basis_path

        gpsd_cmd = [
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
            str(args.seed),
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
            str(args.sample_weight_mode),
        ]
        existing_exp_dir = maybe_newest_match(os.path.join(repo_root, "exps", f"{expr_name}_{dataset_name}_*"))
        existing_model = ""
        if existing_exp_dir:
            existing_model = maybe_newest_match(os.path.join(existing_exp_dir, "checkpoints", "ema_*.pth"))
        if existing_exp_dir and existing_model:
            print(f"SKIP GPSD: reuse existing checkpoint for {dataset_name}")
            row["gpsd_seconds"] = 0.0
        else:
            t0 = time.time()
            run_cmd(gpsd_cmd, cwd=repo_root, dry_run=False)
            row["gpsd_seconds"] = time.time() - t0
        row["total_train_seconds"] = row["ftm_seconds"] + row["gpsd_seconds"]

        exp_dir = newest_match(os.path.join(repo_root, "exps", f"{expr_name}_{dataset_name}_*"))
        core_mean_std_path = os.path.join(exp_dir, "core_mean_std.mat")
        model_path = newest_match(os.path.join(exp_dir, "checkpoints", "ema_*.pth"))
        row["gpsd_dir"] = exp_dir
        row["core_mean_std_path"] = core_mean_std_path
        row["model_path"] = model_path

        eval_out_dir = os.path.join(output_root, "eval", f"size_{size:03d}")
        eval_cmd = [
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
            model_path,
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
            "--missing_type",
            str(args.missing_type),
            "--num_posterior_samples",
            str(args.num_posterior_samples),
            "--total_steps",
            str(args.eval_total_steps),
            "--truth_threshold",
            str(args.truth_threshold),
        ]
        aggregate_json = os.path.join(eval_out_dir, "aggregate_metrics.json")
        if os.path.exists(aggregate_json):
            print(f"SKIP EVAL: reuse existing metrics for {dataset_name}")
            row["evaluation_seconds"] = 0.0
        else:
            t0 = time.time()
            run_cmd(eval_cmd, cwd=repo_root, dry_run=False)
            row["evaluation_seconds"] = time.time() - t0
        row["eval_dir"] = eval_out_dir

        payload = load_json(aggregate_json)
        row["num_test_cases"] = int(payload["count"])
        row["sample_weight_mode"] = args.sample_weight_mode
        low_block = payload.get("subgroups", {}).get(low_group_name, {})
        row["low_rates_label"] = ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in sorted(set(float(x) for x in args.low_rates)))
        row["low_rates_count"] = low_block.get("count")
        for key in METRIC_KEYS:
            metric = payload["metrics"].get(key, {})
            row[f"{key}_mean"] = metric.get("mean")
            row[f"{key}_std"] = metric.get("std")
            low_metric = low_block.get("metrics", {}).get(key, {})
            row[f"low_{key}_mean"] = low_metric.get("mean")
            row[f"low_{key}_std"] = low_metric.get("std")

        rows.append(row)

    summary_json = os.path.join(output_root, "training_scale_study.json")
    summary_csv = os.path.join(output_root, "training_scale_study.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_tag": args.study_tag,
                "train_sizes": sorted(set(int(x) for x in args.train_sizes)),
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "train_size",
        "num_test_cases",
        "sample_weight_mode",
        "low_rates_label",
        "low_rates_count",
        "ftm_seconds",
        "gpsd_seconds",
        "total_train_seconds",
        "evaluation_seconds",
        "subset_h5",
        "subset_manifest",
        "core_path",
        "basis_path",
        "gpsd_dir",
        "core_mean_std_path",
        "model_path",
        "eval_dir",
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
