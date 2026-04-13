import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "advisor_results_20260411"
OUT = ROOT / "docs" / "report_figures_20260412"
OUT.mkdir(parents=True, exist_ok=True)


def savefig(name):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_sensor_time_lines():
    data = load_json(SRC / "sensor_condition_study.json")
    rows = data["rows"]
    sensors = sorted({int(r["sensor_count"]) for r in rows})
    times = sorted({int(r["observed_time_steps"]) for r in rows})
    matrix = np.full((len(sensors), len(times)), np.nan)
    for r in rows:
        i = sensors.index(int(r["sensor_count"]))
        j = times.index(int(r["observed_time_steps"]))
        matrix[i, j] = float(r["global_rmse_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    sensor_colors = ["#2F6B5F", "#C9792B", "#2563EB"]
    time_colors = ["#4F8E72", "#D97706", "#1D4ED8"]

    for i, sensor in enumerate(sensors):
        ys = matrix[i, :]
        axes[0].plot(times, ys, marker="o", linewidth=2.2, color=sensor_colors[i], label=f"{sensor} sensors")
        for x, y in zip(times, ys):
            axes[0].text(x, y, f"{y:.4f}", fontsize=8, ha="center", va="bottom")
    axes[0].set_title("Observation length effect")
    axes[0].set_xlabel("Observed time steps")
    axes[0].set_ylabel("Global RMSE")
    axes[0].set_xticks(times)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for j, time_steps in enumerate(times):
        ys = matrix[:, j]
        axes[1].plot(sensors, ys, marker="o", linewidth=2.2, color=time_colors[j], label=f"{time_steps} steps")
        for x, y in zip(sensors, ys):
            axes[1].text(x, y, f"{y:.4f}", fontsize=8, ha="center", va="bottom")
    axes[1].set_title("Sensor count effect")
    axes[1].set_xlabel("Sensor count")
    axes[1].set_xticks(sensors)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    savefig("sensor_time_rmse_lines.png")


def plot_source_param():
    data = load_json(SRC / "sensor_param_baseline.json")
    tm = data["test_metrics"]
    labels = ["Source error\n(mm)", "Leak-rate MAE\n(mL/min)", "Leak-rate rel.\nerror (%)"]
    vals = [
        tm["source_l2_error_mm_mean"],
        tm["leak_rate_mae_ml_min"],
        tm["leak_rate_rel_error_mean"] * 100.0,
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, vals, color=["#2F6B5F", "#4F8E72", "#C9792B"])
    ax.set_title("Sensor regressor source-parameter baseline")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    savefig("source_param_sensor_baseline.png")


def plot_quality_cases():
    q56 = load_json(ROOT / "docs" / "cfd56_quality_report.json")
    q48 = load_json(ROOT / "docs" / "cfd48_clean_quality_report.json")
    labels = ["Original 56-case", "Clean 48-case"]
    valid = [q56["case_count"] - q56["flagged_count"], q48["case_count"] - q48["flagged_count"]]
    invalid = [q56["flagged_count"], q48["flagged_count"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(labels, valid, label="Valid", color="#2F6B5F")
    ax.bar(labels, invalid, bottom=valid, label="Invalid", color="#C2410C")
    ax.set_title("Dataset quality check")
    ax.set_ylabel("Case count")
    ax.legend(frameon=False)
    for i, (v, inv) in enumerate(zip(valid, invalid)):
        ax.text(i, v / 2, str(v), ha="center", va="center", color="white", fontweight="bold")
        if inv:
            ax.text(i, v + inv / 2, str(inv), ha="center", va="center", color="white", fontweight="bold")
    savefig("dataset_quality_valid_invalid.png")


def main():
    plot_sensor_time_lines()
    plot_source_param()
    plot_quality_cases()


if __name__ == "__main__":
    main()
