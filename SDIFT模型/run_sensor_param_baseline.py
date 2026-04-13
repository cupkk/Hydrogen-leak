import argparse
import os
import subprocess
import sys


def run_cmd(cmd, cwd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train the formal sensor-only source/leak parameter baseline.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--train_field_h5", required=True)
    parser.add_argument("--test_field_h5", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--early_steps", type=int, default=20)
    parser.add_argument("--observed_time_steps", type=int, default=120)
    parser.add_argument(
        "--sample_weight_mode",
        type=str,
        default="none",
        choices=["none", "balanced_by_rate", "lowflow_focus_v1", "lowflow_balanced_v1"],
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    out_json = os.path.join(output_dir, "sensor_param_baseline.json")
    out_csv = os.path.join(output_dir, "sensor_param_baseline_train_predictions.csv")
    run_cmd(
        [
            sys.executable,
            "train_source_param_regressor.py",
            "--input_mode",
            "sensor",
            "--train_manifest",
            args.train_manifest,
            "--test_manifest",
            args.test_manifest,
            "--field_h5",
            args.train_field_h5,
            "--test_field_h5",
            args.test_field_h5,
            "--metadata_path",
            args.metadata_path,
            "--sensor_csv",
            args.sensor_csv,
            "--out_json",
            out_json,
            "--out_csv",
            out_csv,
            "--alpha",
            str(args.alpha),
            "--early_steps",
            str(args.early_steps),
            "--observed_time_steps",
            str(args.observed_time_steps),
            "--sample_weight_mode",
            args.sample_weight_mode,
        ],
        cwd=repo_root,
    )
    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
