import argparse
import csv
import itertools
import json
import os
import subprocess
import sys


def run_cmd(cmd, cwd, dry_run=False):
    print("RUN:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_list(text, cast):
    vals = [cast(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("grid cannot be empty")
    return vals


def metric_from_rows(rows, mode, key):
    for row in rows:
        if row["mode"] == mode:
            return row.get(key)
    return None


def main():
    parser = argparse.ArgumentParser(description="Tune MPDPS observation hyperparameters using correct-vs-corrupted sanity checks.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--basis_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--core_mean_std_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_prefix", default="mpdps_tune")
    parser.add_argument("--modes", default="correct,shuffled,zeros,wrong_positions")
    parser.add_argument("--mpdps_weights", default="0.4,1.0,2.0")
    parser.add_argument("--obs_rhos", default="0.01,0.003")
    parser.add_argument("--total_steps_grid", default="20,30")
    parser.add_argument("--posterior_samples_grid", default="1,2")
    parser.add_argument("--zeta_grid", default="0.009")
    parser.add_argument("--obs_injection_mode", choices=["legacy", "adaptive_ratio", "direct_inner"], default="legacy")
    parser.add_argument("--obs_scale_schedule", choices=["legacy_decay", "constant"], default="legacy_decay")
    parser.add_argument("--obs_scale_blend", choices=["max", "replace"], default="max")
    parser.add_argument("--obs_target_ratio", type=float, default=0.25)
    parser.add_argument("--obs_min_scale", type=float, default=0.0)
    parser.add_argument("--obs_max_scale", type=float, default=500.0)
    parser.add_argument("--obs_inner_steps", type=int, default=3)
    parser.add_argument("--obs_inner_decay", type=float, default=0.7)
    parser.add_argument("--spatial_dims", type=int, default=3)
    parser.add_argument("--img_size_3d", type=int, nargs=3, default=[4, 8, 8])
    parser.add_argument("--missing_type", type=int, default=1)
    parser.add_argument("--observed_time_steps", type=int, default=120)
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--cleanup_recon_mat", action="store_true", default=False)
    parser.add_argument("--cleanup_sensor_cache", action="store_true", default=False)
    parser.add_argument("--skip_existing_tags", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    rows = []
    mpdps_weights = parse_list(args.mpdps_weights, float)
    obs_rhos = parse_list(args.obs_rhos, float)
    total_steps_grid = parse_list(args.total_steps_grid, int)
    posterior_samples_grid = parse_list(args.posterior_samples_grid, int)
    zeta_grid = parse_list(args.zeta_grid, float)
    if len(obs_rhos) > 1:
        print(
            "WARNING: obs_rho is ignored by message_passing_DPS.py when sensor_path is used; "
            "sweeping obs_rho in this script is redundant."
        )
    for mpdps_weight, obs_rho, total_steps, num_posterior_samples in itertools.product(
        mpdps_weights, obs_rhos, total_steps_grid, posterior_samples_grid
    ):
        for zeta in zeta_grid:
            tag = (
                f"w{mpdps_weight:g}_rho{obs_rho:g}_steps{int(total_steps):02d}_"
                f"post{int(num_posterior_samples):02d}_z{zeta:g}"
            )
            out_dir = os.path.join(output_root, tag)
            summary_json = os.path.join(out_dir, "observation_sanity_check.json")
            if args.skip_existing_tags and os.path.exists(summary_json):
                payload = load_json(summary_json)
                sanity_rows = payload["rows"]
                row = {
                    "tag": tag,
                    "mpdps_weight": mpdps_weight,
                    "obs_rho": obs_rho,
                    "total_steps": total_steps,
                    "num_posterior_samples": num_posterior_samples,
                    "zeta": zeta,
                    "obs_injection_mode": args.obs_injection_mode,
                    "obs_scale_schedule": args.obs_scale_schedule,
                    "obs_scale_blend": args.obs_scale_blend,
                    "obs_target_ratio": float(args.obs_target_ratio),
                    "obs_inner_steps": int(args.obs_inner_steps),
                    "output_dir": out_dir,
                }
                for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
                    for metric_key in ["global_rmse_mean", "global_mae_mean", "global_rel_l2_mean", "global_rel_l1_active_mean_mean"]:
                        value = metric_from_rows(sanity_rows, mode, metric_key)
                        row[f"{mode}_{metric_key}"] = value
                c = row.get("correct_global_rel_l2_mean")
                s = row.get("shuffled_global_rel_l2_mean")
                if c is not None and s is not None:
                    row["shuffled_minus_correct_rel_l2"] = float(s - c)
                    row["shuffled_over_correct_rel_l2"] = float(s / c) if c != 0 else None
                c = row.get("correct_global_rmse_mean")
                s = row.get("shuffled_global_rmse_mean")
                if c is not None and s is not None:
                    row["shuffled_minus_correct_rmse"] = float(s - c)
                    row["shuffled_over_correct_rmse"] = float(s / c) if c != 0 else None
                rows.append(row)
                continue
            cmd = [
                sys.executable,
                "run_observation_sanity_check.py",
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
                "--output_root",
                out_dir,
                "--dataset_prefix",
                f"{args.dataset_prefix}_{tag}",
                "--modes",
                args.modes,
                "--spatial_dims",
                str(args.spatial_dims),
                "--img_size_3d",
                *(str(x) for x in args.img_size_3d),
                "--mpdps_weight",
                str(mpdps_weight),
                "--obs_rho",
                str(obs_rho),
                "--zeta",
                str(zeta),
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
                str(num_posterior_samples),
                "--total_steps",
                str(total_steps),
                "--observed_time_steps",
                str(args.observed_time_steps),
                "--truth_threshold",
                str(args.truth_threshold),
                "--seed",
                str(args.seed),
            ]
            if args.cleanup_recon_mat:
                cmd.append("--cleanup_recon_mat")
            if args.cleanup_sensor_cache:
                cmd.append("--cleanup_sensor_cache")
            if args.skip_existing_tags:
                cmd.append("--skip_existing_modes")
            run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)
            row = {
                "tag": tag,
                "mpdps_weight": mpdps_weight,
                "obs_rho": obs_rho,
                "total_steps": total_steps,
                "num_posterior_samples": num_posterior_samples,
                "zeta": zeta,
                "obs_injection_mode": args.obs_injection_mode,
                "obs_scale_schedule": args.obs_scale_schedule,
                "obs_scale_blend": args.obs_scale_blend,
                "obs_target_ratio": float(args.obs_target_ratio),
                "obs_inner_steps": int(args.obs_inner_steps),
                "output_dir": out_dir,
            }
            if not args.dry_run:
                payload = load_json(os.path.join(out_dir, "observation_sanity_check.json"))
                sanity_rows = payload["rows"]
                for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
                    for metric_key in ["global_rmse_mean", "global_mae_mean", "global_rel_l2_mean", "global_rel_l1_active_mean_mean"]:
                        value = metric_from_rows(sanity_rows, mode, metric_key)
                        row[f"{mode}_{metric_key}"] = value
                c = row.get("correct_global_rel_l2_mean")
                s = row.get("shuffled_global_rel_l2_mean")
                if c is not None and s is not None:
                    row["shuffled_minus_correct_rel_l2"] = float(s - c)
                    row["shuffled_over_correct_rel_l2"] = float(s / c) if c != 0 else None
                c = row.get("correct_global_rmse_mean")
                s = row.get("shuffled_global_rmse_mean")
                if c is not None and s is not None:
                    row["shuffled_minus_correct_rmse"] = float(s - c)
                    row["shuffled_over_correct_rmse"] = float(s / c) if c != 0 else None
            rows.append(row)

    summary_json = os.path.join(output_root, "mpdps_observation_tuning.json")
    summary_csv = os.path.join(output_root, "mpdps_observation_tuning.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    fieldnames = [
        "tag",
        "mpdps_weight",
        "obs_rho",
        "total_steps",
        "num_posterior_samples",
        "zeta",
        "obs_injection_mode",
        "obs_scale_schedule",
        "obs_scale_blend",
        "obs_target_ratio",
        "obs_inner_steps",
        "output_dir",
    ]
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        for metric_key in ["global_rmse_mean", "global_mae_mean", "global_rel_l2_mean", "global_rel_l1_active_mean_mean"]:
            fieldnames.append(f"{mode}_{metric_key}")
    fieldnames.extend(
        [
            "shuffled_minus_correct_rel_l2",
            "shuffled_over_correct_rel_l2",
            "shuffled_minus_correct_rmse",
            "shuffled_over_correct_rmse",
        ]
    )
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {summary_json}")
    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()
