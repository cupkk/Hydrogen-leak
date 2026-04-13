import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=None):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def plot_sensor_study(csv_path, out_dir):
    rows = load_csv_rows(csv_path)
    metrics = [
        "global_rmse_mean",
        "global_mae_mean",
        "global_rel_l1_active_mean_mean",
        "low_global_rel_l1_active_mean_mean",
    ]
    os.makedirs(out_dir, exist_ok=True)
    sensor_counts = sorted({int(float(r["sensor_count"])) for r in rows})
    obs_steps = sorted({int(float(r["observed_time_steps"])) for r in rows})

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for obs in obs_steps:
            xs, ys = [], []
            for r in rows:
                if int(float(r["observed_time_steps"])) != obs:
                    continue
                y = as_float(r, metric)
                if y is None:
                    continue
                xs.append(int(float(r["sensor_count"])))
                ys.append(y)
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                label = "full" if obs <= 0 else str(obs)
                plt.plot(xs, ys, marker="o", label=f"time={label}")
        plt.xlabel("Sensor Count")
        plt.ylabel(metric)
        plt.xticks(sensor_counts)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sensor_{metric}.png"), dpi=160)
        plt.close()

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for sensor_count in sensor_counts:
            xs, ys = [], []
            for r in rows:
                if int(float(r["sensor_count"])) != sensor_count:
                    continue
                y = as_float(r, metric)
                if y is None:
                    continue
                xs.append(int(float(r["observed_time_steps"])))
                ys.append(y)
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                plt.plot(xs, ys, marker="o", label=f"sensors={sensor_count}")
        plt.xlabel("Observed Time Steps")
        plt.ylabel(metric)
        plt.xticks(obs_steps)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"time_{metric}.png"), dpi=160)
        plt.close()


def plot_scale_study(csv_path, out_dir):
    rows = load_csv_rows(csv_path)
    metrics = [
        "global_rmse_mean",
        "global_mae_mean",
        "global_rel_l1_active_mean_mean",
        "low_global_rel_l1_active_mean_mean",
    ]
    os.makedirs(out_dir, exist_ok=True)
    xs = [int(float(r["train_size"])) for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    rows = [rows[i] for i in order]

    for metric in metrics:
        ys = [as_float(r, metric) for r in rows]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Train Size")
        plt.ylabel(metric)
        plt.xticks(xs)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scale_{metric}.png"), dpi=160)
        plt.close()

    ys = [as_float(r, "total_train_seconds") for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Train Size")
    plt.ylabel("total_train_seconds")
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scale_total_train_seconds.png"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot advisor study curves from CSV summaries.")
    parser.add_argument("--sensor_csv", default="")
    parser.add_argument("--scale_csv", default="")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if not args.sensor_csv and not args.scale_csv:
        raise ValueError("at least one of --sensor_csv or --scale_csv is required")

    if args.sensor_csv:
        plot_sensor_study(args.sensor_csv, os.path.join(args.out_dir, "sensor"))
    if args.scale_csv:
        plot_scale_study(args.scale_csv, os.path.join(args.out_dir, "scale"))


if __name__ == "__main__":
    main()
