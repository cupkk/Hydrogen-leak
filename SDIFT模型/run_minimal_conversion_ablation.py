import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List

from build_cfd_multicase_dataset import discover_case_dirs


DEFAULT_SCAN_ROOTS = [r"E:\氢泄漏", r"F:\氢泄露"]
DEFAULT_SENSOR_CSV = os.path.join("data", "sensors_real_12.csv")
METRIC_PRIORITY = [
    "global_rmse",
    "global_mae",
    "global_rel_l1_active_mean",
    "global_rel_l2",
    "mass_mean_rel_error",
]

PRESETS: Dict[str, Dict] = {
    "mini3": {
        "description": "3-case fast screen. Two train cases at x=300, one unseen-position test at x=400.",
        "holdout_position": (400, 0, 0),
        "cases": [
            "Q100-X300-Y0-Fraction",
            "Q400-X300-Y0-Fraction",
            "Q400-X400-Y0-Fraction",
        ],
        "rationale": [
            "only uses clean cases from the new 22-case batch",
            "keeps an unseen-position test",
            "fastest option for a first parameter screen",
        ],
    },
    "mini5": {
        "description": "5-case default screen. Three train cases at x=300, two unseen-position tests at x=400.",
        "holdout_position": (400, 0, 0),
        "cases": [
            "Q100-X300-Y0-Fraction",
            "Q400-X300-Y0-Fraction",
            "Q1000-X300-Y0-Fraction",
            "Q100-X400-Y0-Fraction",
            "Q400-X400-Y0-Fraction",
        ],
        "rationale": [
            "only uses clean cases from the new 22-case batch",
            "covers low, medium, and high leak rates in training",
            "holds out a whole unseen source position",
            "more stable than mini3 while still cheap",
        ],
    },
}

CONFIGS: Dict[str, Dict[str, float]] = {
    "k1_p2": {
        "bins": 48,
        "interp_k": 1,
        "interp_power": 2.0,
        "description": "nearest-like lower bound at fixed 48^3 grid",
    },
    "k8_p2": {
        "bins": 48,
        "interp_k": 8,
        "interp_power": 2.0,
        "description": "current baseline used by the main pipeline",
    },
    "k16_p2": {
        "bins": 48,
        "interp_k": 16,
        "interp_power": 2.0,
        "description": "smoother interpolation with more neighbors",
    },
}


def run_cmd(cmd: List[str], cwd: str):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_matching_cases(scan_roots: List[str]):
    cases = discover_case_dirs(scan_roots)
    return {case_name: data_dir for case_name, data_dir in cases}


def find_new_or_latest(pattern: str, before: set):
    matches = set(glob.glob(pattern))
    new_matches = sorted(matches - before, key=os.path.getmtime)
    if new_matches:
        return new_matches[-1]
    all_matches = sorted(matches, key=os.path.getmtime)
    if not all_matches:
        raise FileNotFoundError(f"No match for pattern: {pattern}")
    return all_matches[-1]


def load_aggregate(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ranking_key(row: Dict):
    values = []
    metrics = row.get("metrics", {})
    for key in METRIC_PRIORITY:
        metric = metrics.get(key, {})
        value = metric.get("mean")
        values.append(float("inf") if value is None else float(value))
    return tuple(values)


def summarize_config_result(config_name: str, config_spec: Dict, aggregate: Dict, config_dir: str):
    metrics = aggregate.get("metrics", {})
    return {
        "config_name": config_name,
        "description": config_spec["description"],
        "bins": config_spec["bins"],
        "interp_k": config_spec["interp_k"],
        "interp_power": config_spec["interp_power"],
        "count": aggregate.get("count"),
        "global_rmse_mean": metrics.get("global_rmse", {}).get("mean"),
        "global_mae_mean": metrics.get("global_mae", {}).get("mean"),
        "global_rel_l1_active_mean": metrics.get("global_rel_l1_active_mean", {}).get("mean"),
        "global_rel_l2_mean": metrics.get("global_rel_l2", {}).get("mean"),
        "mass_mean_rel_error": metrics.get("mass_mean_rel_error", {}).get("mean"),
        "aggregate_json": os.path.join(config_dir, "eval", "aggregate_metrics.json"),
        "aggregate_csv": os.path.join(config_dir, "eval", "aggregate_metrics.csv"),
    }


def build_experiment_plan(args, preset_name: str, preset: Dict, selected_configs: Dict[str, Dict], case_map: Dict[str, str]):
    train_cases = []
    test_cases = []
    holdout_position = tuple(preset["holdout_position"])
    for case_name in preset["cases"]:
        if case_name not in case_map:
            continue
        if case_name.endswith("X400-Y0-Fraction") or case_name.startswith("6,400"):
            test_cases.append(case_name)
        else:
            train_cases.append(case_name)
    return {
        "preset": preset_name,
        "description": preset["description"],
        "holdout_position_mm": list(holdout_position),
        "selected_cases": preset["cases"],
        "train_cases_expected": train_cases,
        "test_cases_expected": test_cases,
        "rationale": preset["rationale"],
        "configs": selected_configs,
        "metric_priority": METRIC_PRIORITY,
        "decision_rule": "Sort by global_rmse, then global_mae, then global_rel_l1_active_mean.",
        "training": {
            "ftm_max_iter": args.ftm_max_iter,
            "ftm_batch_size": args.ftm_batch_size,
            "gpsd_steps": args.gpsd_steps,
            "gpsd_batch_size": args.gpsd_batch_size,
            "posterior_steps": args.posterior_steps,
            "seed": args.seed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run a minimal conversion-parameter ablation on 3-5 representative CFD cases.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--scan_root", action="append", default=[])
    parser.add_argument("--work_root", default=os.path.join("exps", "minimal_conversion_ablation"))
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="mini5")
    parser.add_argument("--config", action="append", default=[], help="Restrict to a subset of conversion configs.")
    parser.add_argument("--sensor_csv", default=DEFAULT_SENSOR_CSV)
    parser.add_argument("--selected_steps", type=int, default=120)
    parser.add_argument("--compression_level", type=int, default=1)
    parser.add_argument("--ftm_max_iter", type=int, default=400)
    parser.add_argument("--ftm_batch_size", type=int, default=16)
    parser.add_argument("--gpsd_steps", type=int, default=1000)
    parser.add_argument("--gpsd_batch_size", type=int, default=8)
    parser.add_argument("--posterior_steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--convert_only", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    work_root = args.work_root if os.path.isabs(args.work_root) else os.path.join(repo_root, args.work_root)
    work_root = os.path.abspath(work_root)
    sensor_csv = args.sensor_csv if os.path.isabs(args.sensor_csv) else os.path.join(repo_root, args.sensor_csv)
    ensure_dir(work_root)

    scan_roots = args.scan_root or DEFAULT_SCAN_ROOTS
    preset = PRESETS[args.preset]
    selected_configs = {name: CONFIGS[name] for name in (args.config or CONFIGS.keys())}
    case_map = find_matching_cases(scan_roots)
    missing_cases = [case_name for case_name in preset["cases"] if case_name not in case_map]
    if missing_cases:
        raise SystemExit(f"Missing selected case(s): {missing_cases}")

    selected_case_rows = [
        {
            "case_name": case_name,
            "data_dir": case_map[case_name],
            "role": "test" if case_name.startswith("Q") and "-X400-" in case_name else "train",
        }
        for case_name in preset["cases"]
    ]
    write_csv(os.path.join(work_root, f"{args.preset}_selected_cases.csv"), selected_case_rows, ["case_name", "data_dir", "role"])

    plan = build_experiment_plan(args, args.preset, preset, selected_configs, case_map)
    write_json(os.path.join(work_root, f"{args.preset}_plan.json"), plan)

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"saved: {os.path.join(work_root, f'{args.preset}_selected_cases.csv')}")
        print(f"saved: {os.path.join(work_root, f'{args.preset}_plan.json')}")
        return

    result_rows = []
    for config_name, config_spec in selected_configs.items():
        config_dir = os.path.join(work_root, args.preset, config_name)
        data_dir = os.path.join(config_dir, "data")
        split_dir = os.path.join(config_dir, "split")
        eval_dir = os.path.join(config_dir, "eval")
        ensure_dir(data_dir)
        ensure_dir(split_dir)
        ensure_dir(eval_dir)

        out_h5 = os.path.join(data_dir, f"{args.preset}_{config_name}.h5")
        out_meta = os.path.join(data_dir, f"{args.preset}_{config_name}_meta.npy")
        out_manifest = os.path.join(data_dir, f"{args.preset}_{config_name}_manifest.csv")
        out_report = os.path.join(data_dir, f"{args.preset}_{config_name}_report.json")

        convert_cmd = [
            sys.executable,
            "build_cfd_multicase_dataset.py",
            "--out_h5",
            out_h5,
            "--out_meta",
            out_meta,
            "--out_manifest",
            out_manifest,
            "--out_report",
            out_report,
            "--selected_steps",
            str(args.selected_steps),
            "--bins",
            str(config_spec["bins"]),
            "--interp_k",
            str(config_spec["interp_k"]),
            "--interp_power",
            str(config_spec["interp_power"]),
            "--compression_level",
            str(args.compression_level),
            "--overwrite",
        ]
        for scan_root in scan_roots:
            convert_cmd.extend(["--scan_root", scan_root])
        for case_name in preset["cases"]:
            convert_cmd.extend(["--include_case", case_name])
        run_cmd(convert_cmd, cwd=repo_root)

        split_cmd = [
            sys.executable,
            "subset_h5_by_manifest.py",
            "--input_h5",
            out_h5,
            "--manifest_csv",
            out_manifest,
            "--holdout_position",
            ",".join(str(x) for x in preset["holdout_position"]),
            "--out_train_h5",
            os.path.join(split_dir, "train.h5"),
            "--out_train_manifest",
            os.path.join(split_dir, "train_manifest.csv"),
            "--out_test_h5",
            os.path.join(split_dir, "test.h5"),
            "--out_test_manifest",
            os.path.join(split_dir, "test_manifest.csv"),
            "--out_split_json",
            os.path.join(split_dir, "split.json"),
        ]
        run_cmd(split_cmd, cwd=repo_root)

        if args.convert_only:
            continue

        train_data_name = f"abl_{args.preset}_{config_name}"
        expr_name = f"abl_{args.preset}_{config_name}"

        core_pattern = os.path.join(repo_root, "data", f"core_{train_data_name}_4x8x8_*_last.mat")
        basis_pattern = os.path.join(repo_root, "ckp", f"basis_{train_data_name}_4x8x8_*_last.pth")
        exp_pattern = os.path.join(repo_root, "exps", f"{expr_name}_{train_data_name}_*")
        before_cores = set(glob.glob(core_pattern))
        before_basises = set(glob.glob(basis_pattern))
        before_exps = set(glob.glob(exp_pattern))

        ftm_cmd = [
            sys.executable,
            "train_FTM.py",
            "--data_name",
            train_data_name,
            "--data_path",
            os.path.join(split_dir, "train.h5"),
            "--metadata_path",
            out_meta,
            "--R",
            "4",
            "8",
            "8",
            "--batch_size",
            str(args.ftm_batch_size),
            "--max_iter",
            str(args.ftm_max_iter),
            "--save_last",
            "--seed",
            str(args.seed),
        ]
        run_cmd(ftm_cmd, cwd=repo_root)
        core_path = find_new_or_latest(core_pattern, before_cores)
        basis_path = find_new_or_latest(basis_pattern, before_basises)

        gpsd_cmd = [
            sys.executable,
            "train_GPSD.py",
            "--expr",
            expr_name,
            "--dataset",
            train_data_name,
            "--core_path",
            core_path,
            "--spatial_dims",
            "3",
            "--img_size_3d",
            "4",
            "8",
            "8",
            "--train_batch_size",
            str(args.gpsd_batch_size),
            "--num_steps",
            str(args.gpsd_steps),
            "--save_model_iters",
            str(min(args.gpsd_steps, 200)),
            "--save_signals_step",
            "0",
            "--seed",
            str(args.seed),
        ]
        run_cmd(gpsd_cmd, cwd=repo_root)
        exp_dir = find_new_or_latest(exp_pattern, before_exps)
        model_path = os.path.join(exp_dir, "checkpoints", f"ema_{args.gpsd_steps - 1}.pth")
        core_mean_std_path = os.path.join(exp_dir, "core_mean_std.mat")
        if not os.path.exists(model_path):
            matches = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "ema_*.pth")), key=os.path.getmtime)
            if not matches:
                raise FileNotFoundError(f"Missing EMA checkpoints under {exp_dir}")
            model_path = matches[-1]

        eval_cmd = [
            sys.executable,
            "run_holdout_reconstruction_eval.py",
            "--repo_root",
            repo_root,
            "--test_h5",
            os.path.join(split_dir, "test.h5"),
            "--test_manifest",
            os.path.join(split_dir, "test_manifest.csv"),
            "--metadata_path",
            out_meta,
            "--sensor_csv",
            sensor_csv,
            "--basis_path",
            basis_path,
            "--model_path",
            model_path,
            "--core_mean_std_path",
            core_mean_std_path,
            "--output_dir",
            eval_dir,
            "--dataset_prefix",
            train_data_name,
            "--num_posterior_samples",
            "1",
            "--total_steps",
            str(args.posterior_steps),
            "--obs_rho",
            "0.01",
            "--mpdps_weight",
            "0.4",
        ]
        run_cmd(eval_cmd, cwd=repo_root)

        aggregate = load_aggregate(os.path.join(eval_dir, "aggregate_metrics.json"))
        result_rows.append(
            {
                "config_name": config_name,
                "count": aggregate.get("count"),
                "metrics": aggregate.get("metrics", {}),
                "config_dir": config_dir,
            }
        )

    if args.convert_only:
        print("Conversion and split preparation finished.")
        return

    ranked = sorted(result_rows, key=ranking_key)
    ranking_rows = []
    for rank, row in enumerate(ranked, start=1):
        config_name = row["config_name"]
        ranking_rows.append(
            {
                "rank": rank,
                **summarize_config_result(config_name, selected_configs[config_name], {"count": row.get("count"), "metrics": row["metrics"]}, row["config_dir"]),
            }
        )

    write_csv(
        os.path.join(work_root, args.preset, "ranking.csv"),
        ranking_rows,
        [
            "rank",
            "config_name",
            "description",
            "bins",
            "interp_k",
            "interp_power",
            "count",
            "global_rmse_mean",
            "global_mae_mean",
            "global_rel_l1_active_mean",
            "global_rel_l2_mean",
            "mass_mean_rel_error",
            "aggregate_json",
            "aggregate_csv",
        ],
    )

    write_json(
        os.path.join(work_root, args.preset, "ranking.json"),
        {
            "preset": args.preset,
            "metric_priority": METRIC_PRIORITY,
            "recommended_config": ranked[0]["config_name"] if ranked else None,
            "rows": ranking_rows,
        },
    )
    print(f"saved: {os.path.join(work_root, args.preset, 'ranking.csv')}")
    print(f"saved: {os.path.join(work_root, args.preset, 'ranking.json')}")


if __name__ == "__main__":
    main()
