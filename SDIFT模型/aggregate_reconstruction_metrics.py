import argparse
import csv
import glob
import json
import os


DEFAULT_COLUMNS = [
    "case_index",
    "case_id",
    "raw_case_name",
    "source_x_mm",
    "source_y_mm",
    "source_z_mm",
    "leak_rate_ml_min",
    "global_rmse",
    "global_mae",
    "global_rel_l1_mean",
    "global_rel_l1_active_mean",
    "global_rel_l2",
    "mass_mean_rel_error",
]


def load_manifest(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_index = {}
    for row in rows:
        by_index[int(row["data_index"])] = row
    return by_index


def expand_eval_paths(items):
    out = []
    for item in items:
        matches = sorted(glob.glob(item))
        if matches:
            out.extend(matches)
        else:
            out.append(item)
    return out


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_row(payload, manifest_row):
    metrics = payload["metrics"]
    case_index = payload.get("truth_sample_index", payload.get("sample_index", 0))
    row = {
        "case_index": int(case_index),
        "case_id": manifest_row.get("case_id", ""),
        "raw_case_name": manifest_row.get("raw_case_name", ""),
        "source_x_mm": manifest_row.get("source_x_mm", ""),
        "source_y_mm": manifest_row.get("source_y_mm", ""),
        "source_z_mm": manifest_row.get("source_z_mm", ""),
        "leak_rate_ml_min": manifest_row.get("leak_rate_ml_min", ""),
        "global_rmse": metrics.get("global_rmse"),
        "global_mae": metrics.get("global_mae"),
        "global_rel_l1_mean": metrics.get("global_rel_l1_mean"),
        "global_rel_l1_active_mean": metrics.get("global_rel_l1_active_mean"),
        "global_rel_l2": metrics.get("global_rel_l2"),
        "mass_mean_rel_error": safe_get(metrics, "mass", "mean_rel_error"),
        "eval_json": payload.get("eval_json", ""),
    }
    return row


def mean_std(values):
    vals = [float(v) for v in values if v is not None and v != ""]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, var ** 0.5


def summarize_metric_block(rows, keys):
    block = {}
    for key in keys:
        mean, std = mean_std([row[key] for row in rows])
        block[key] = {"mean": mean, "std": std}
    return block


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-case reconstruction evaluation JSON files.")
    parser.add_argument("--eval_json", action="append", required=True, help="Evaluation JSON path or glob. Can be provided multiple times.")
    parser.add_argument("--manifest_csv", default="")
    parser.add_argument("--low_rates", type=float, nargs="*", default=[50.0, 100.0])
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    eval_paths = expand_eval_paths(args.eval_json)
    manifest = load_manifest(args.manifest_csv)
    rows = []
    for path in eval_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["eval_json"] = path
        case_index = int(payload.get("truth_sample_index", payload.get("sample_index", 0)))
        manifest_row = manifest.get(case_index, {})
        rows.append(build_row(payload, manifest_row))

    rows.sort(key=lambda x: x["case_index"])
    summary = {
        "count": len(rows),
        "metrics": {},
        "rows": rows,
    }
    metric_keys = [
        "global_rmse",
        "global_mae",
        "global_rel_l1_mean",
        "global_rel_l1_active_mean",
        "global_rel_l2",
        "mass_mean_rel_error",
    ]
    summary["metrics"] = summarize_metric_block(rows, metric_keys)

    low_rates = {float(x) for x in args.low_rates}
    subgroups = {}
    low_rows = [row for row in rows if row.get("leak_rate_ml_min", "") != "" and float(row["leak_rate_ml_min"]) in low_rates]
    if low_rows:
        subgroup_name = "low_rates_" + "_".join(str(int(x)) if float(x).is_integer() else str(x) for x in sorted(low_rates))
        subgroups[subgroup_name] = {
            "count": len(low_rows),
            "rates": sorted(low_rates),
            "metrics": summarize_metric_block(low_rows, metric_keys),
        }
    rate_groups = {}
    for row in rows:
        value = row.get("leak_rate_ml_min", "")
        if value == "":
            continue
        rate_groups.setdefault(float(value), []).append(row)
    for rate, rate_rows in sorted(rate_groups.items()):
        subgroups[f"rate_{int(rate) if float(rate).is_integer() else rate}"] = {
            "count": len(rate_rows),
            "rates": [rate],
            "metrics": summarize_metric_block(rate_rows, metric_keys),
        }
    summary["subgroups"] = subgroups

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS + ["eval_json"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
