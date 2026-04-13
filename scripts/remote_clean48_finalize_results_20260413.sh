#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/hy-tmp/SDIFT_model56}"
STUDY_ROOT="$REPO_ROOT/results/advisor_study/train_scale_lowflow_focus_clean48_20260413"
OUT_DIR="$REPO_ROOT/results/advisor_study/clean48_final_reports_20260413"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/clean48_finalize_results_20260413.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

{
  echo "[$(date)] waiting for train-scale aggregates"
  while true; do
    count=$(find "$STUDY_ROOT" -name aggregate_metrics.json 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date)] aggregate_count=$count"
    if [ "$count" = "12" ]; then
      break
    fi
    sleep 300
  done

  python3 - <<'PY'
import csv
import json
import math
from pathlib import Path

repo = Path("/hy-tmp/SDIFT_model56")
study = repo / "results/advisor_study/train_scale_lowflow_focus_clean48_20260413"
out = repo / "results/advisor_study/clean48_final_reports_20260413"
out.mkdir(parents=True, exist_ok=True)

rows = []
for path in sorted(study.glob("lowflow_focus_v1_n*_r*/aggregate_metrics.json")):
    name = path.parent.name
    n = int(name.split("_n", 1)[1].split("_r", 1)[0])
    r = int(name.rsplit("_r", 1)[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    low = data.get("subgroups", {}).get("low_rates_50_100", {}).get("metrics", {})
    rows.append({
        "train_size": n,
        "repeat": r,
        "rmse": metrics["global_rmse"]["mean"],
        "mae": metrics["global_mae"]["mean"],
        "active_rel_l1": metrics["global_rel_l1_active_mean"]["mean"],
        "mass_rel_error": metrics["mass_mean_rel_error"]["mean"],
        "low_rmse": low.get("global_rmse", {}).get("mean", ""),
        "low_mae": low.get("global_mae", {}).get("mean", ""),
        "path": str(path),
    })

def mean(values):
    return sum(values) / len(values)

def std(values):
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

summary = []
for n in sorted({row["train_size"] for row in rows}):
    group = [row for row in rows if row["train_size"] == n]
    rmse = [float(row["rmse"]) for row in group]
    mae = [float(row["mae"]) for row in group]
    active = [float(row["active_rel_l1"]) for row in group]
    mass = [float(row["mass_rel_error"]) for row in group]
    low_rmse = [float(row["low_rmse"]) for row in group if row["low_rmse"] != ""]
    summary.append({
        "train_size": n,
        "completed_repeats": len(group),
        "rmse_mean": mean(rmse),
        "rmse_std": std(rmse),
        "mae_mean": mean(mae),
        "mae_std": std(mae),
        "active_rel_l1_mean": mean(active),
        "active_rel_l1_std": std(active),
        "mass_rel_error_mean": mean(mass),
        "mass_rel_error_std": std(mass),
        "low_rmse_mean": mean(low_rmse) if low_rmse else "",
        "low_rmse_std": std(low_rmse) if low_rmse else "",
    })

with (out / "clean48_train_scale_all_runs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r["train_size"], r["repeat"])))

with (out / "clean48_train_scale_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

(out / "clean48_train_scale_summary.json").write_text(
    json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

lines = [
    "# Clean48 Train-Size Final Summary",
    "",
    "| train_size | repeats | RMSE mean | RMSE std | MAE mean | low-flow RMSE mean |",
    "| ---: | ---: | ---: | ---: | ---: | ---: |",
]
for item in summary:
    lines.append(
        f"| {item['train_size']} | {item['completed_repeats']} | "
        f"{item['rmse_mean']:.6g} | {item['rmse_std']:.6g} | "
        f"{item['mae_mean']:.6g} | "
        f"{item['low_rmse_mean']:.6g} |"
    )
(out / "clean48_train_scale_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", out)
PY

  echo "[$(date)] final summary written to $OUT_DIR"
  echo "[$(date)] current clean48 error scan"
  grep -RInE 'Traceback|FileNotFoundError|RuntimeError|CUDA out|Killed|No such file|Error|Exception' \
    "$LOG_DIR"/clean48_* "$STUDY_ROOT" 2>/dev/null | tail -n 100 || true
} >> "$LOG_FILE" 2>&1
