import argparse
import json
import os
import subprocess
import sys

from run_holdout_reconstruction_eval import corrupt_sensor_observation


def run_cmd(cmd, cwd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run one-case MPDPS reconstruction with diagnostics enabled.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--basis_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--core_mean_std_path", required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_prefix", required=True)
    parser.add_argument("--case_id", default="")
    parser.add_argument("--spatial_dims", type=int, default=3)
    parser.add_argument("--img_size_3d", type=int, nargs=3, default=[4, 8, 8])
    parser.add_argument("--mpdps_weight", type=float, default=16.0)
    parser.add_argument("--obs_rho", type=float, default=0.01)
    parser.add_argument("--zeta", type=float, default=0.03)
    parser.add_argument("--obs_injection_mode", choices=["legacy", "adaptive_ratio", "direct_inner"], default="legacy")
    parser.add_argument("--obs_scale_schedule", choices=["legacy_decay", "constant"], default="legacy_decay")
    parser.add_argument("--obs_scale_blend", choices=["max", "replace"], default="max")
    parser.add_argument("--obs_target_ratio", type=float, default=0.25)
    parser.add_argument("--obs_min_scale", type=float, default=0.0)
    parser.add_argument("--obs_max_scale", type=float, default=500.0)
    parser.add_argument("--obs_inner_steps", type=int, default=3)
    parser.add_argument("--obs_inner_decay", type=float, default=0.7)
    parser.add_argument("--missing_type", type=int, default=1)
    parser.add_argument("--num_posterior_samples", type=int, default=2)
    parser.add_argument("--total_steps", type=int, default=50)
    parser.add_argument("--observed_time_steps", type=int, default=120)
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
    os.makedirs(sensor_dir, exist_ok=True)

    case_id = args.case_id or f"case_{args.sample_index:04d}"
    sensor_out = os.path.join(sensor_dir, f"{case_id}_sensors.npy")
    diagnostics_json = os.path.join(output_dir, f"{case_id}_{args.sensor_input_mode}_diagnostics.json")
    eval_json = os.path.join(output_dir, f"{case_id}_{args.sensor_input_mode}_eval.json")
    eval_csv = os.path.join(output_dir, f"{case_id}_{args.sensor_input_mode}_per_time.csv")

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
            str(args.sample_index),
        ],
        cwd=repo_root,
    )
    sensor_for_recon = corrupt_sensor_observation(
        sensor_out,
        mode=args.sensor_input_mode,
        metadata_path=args.metadata_path,
        sensor_csv=args.sensor_csv,
        seed=args.seed + args.sample_index,
        wrong_sensor_csv=args.wrong_sensor_csv,
    )

    dataset_name = f"{args.dataset_prefix}_{case_id}_{args.sensor_input_mode}"
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
            "--save_diagnostics",
            "--diagnostics_out",
            diagnostics_json,
        ],
        cwd=repo_root,
    )

    result_name = f"{dataset_name}_mpdps_{args.mpdps_weight}_recon_rho{args.obs_rho}_mode_{args.missing_type}"
    recon_mat = os.path.join(repo_root, "results", result_name + ".mat")
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
            str(args.sample_index),
            "--out_json",
            eval_json,
            "--out_csv",
            eval_csv,
            "--truth_threshold",
            str(args.truth_threshold),
        ],
        cwd=repo_root,
    )

    summary = {
        "case_id": case_id,
        "sample_index": int(args.sample_index),
        "sensor_input_mode": args.sensor_input_mode,
        "sensor_path": sensor_out,
        "sensor_used_path": sensor_for_recon,
        "obs_injection_mode": args.obs_injection_mode,
        "obs_scale_schedule": args.obs_scale_schedule,
        "obs_scale_blend": args.obs_scale_blend,
        "obs_target_ratio": float(args.obs_target_ratio),
        "obs_inner_steps": int(args.obs_inner_steps),
        "diagnostics_json": diagnostics_json,
        "eval_json": eval_json,
        "eval_csv": eval_csv,
    }
    with open(os.path.join(output_dir, f"{case_id}_{args.sensor_input_mode}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.cleanup_recon_mat:
        for path in [
            recon_mat,
            os.path.join(repo_root, "results", result_name + "_summary.json"),
            os.path.join(repo_root, "results", result_name + "_source_est.npy"),
        ]:
            if os.path.exists(path):
                os.remove(path)

    if args.cleanup_sensor_cache:
        for path in {sensor_out, sensor_for_recon}:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()
