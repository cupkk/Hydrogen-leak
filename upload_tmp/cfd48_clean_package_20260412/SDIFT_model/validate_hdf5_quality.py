import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def load_manifest(path):
    if path is None:
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                index = int(row.get("data_index", ""))
            except ValueError:
                continue
            rows[index] = row
    return rows


def get_dataset(handle, key):
    if key:
        return handle[key]
    if "data" in handle:
        return handle["data"]
    keys = list(handle.keys())
    if len(keys) != 1:
        raise KeyError(f"Cannot infer dataset key from keys: {keys}")
    return handle[keys[0]]


def case_stats(array):
    finite = np.isfinite(array)
    if not np.all(finite):
        valid = array[finite]
    else:
        valid = array
    if valid.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "sum": None,
            "finite_ratio": 0.0,
        }
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "sum": float(np.sum(valid, dtype=np.float64)),
        "finite_ratio": float(valid.size / array.size),
    }


def flags_for_stats(stats, args):
    flags = []
    if stats["finite_ratio"] < 1.0:
        flags.append("non_finite_values")
    if stats["std"] is not None and stats["std"] <= args.constant_std_threshold:
        flags.append("near_constant_field")
    if stats["mean"] is not None and stats["mean"] >= args.high_mean_threshold:
        flags.append("mean_too_high")
    if stats["max"] is not None and stats["max"] >= args.high_max_threshold:
        flags.append("max_too_high")
    if stats["max"] is not None and stats["max"] <= args.near_zero_max_threshold:
        flags.append("near_zero_field")
    return flags


def main():
    parser = argparse.ArgumentParser(
        description="Validate converted CFD HDF5 tensors before training."
    )
    parser.add_argument("--h5", required=True, help="Path to converted HDF5 file.")
    parser.add_argument("--manifest", help="Optional manifest CSV with case metadata.")
    parser.add_argument("--dataset-key", default="data", help="HDF5 dataset key.")
    parser.add_argument("--out-csv", help="Optional per-case CSV report path.")
    parser.add_argument("--out-json", help="Optional summary JSON report path.")
    parser.add_argument("--constant-std-threshold", type=float, default=1e-7)
    parser.add_argument("--high-mean-threshold", type=float, default=5e-2)
    parser.add_argument("--high-max-threshold", type=float, default=1.5)
    parser.add_argument("--near-zero-max-threshold", type=float, default=1e-12)
    parser.add_argument("--fail-on-flag", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = []
    with h5py.File(args.h5, "r") as f:
        data = get_dataset(f, args.dataset_key)
        if data.ndim < 2:
            raise ValueError(f"Expected case-first tensor, got shape {data.shape}")
        for index in range(data.shape[0]):
            stats = case_stats(np.asarray(data[index]))
            meta = manifest.get(index, {})
            flags = flags_for_stats(stats, args)
            row = {
                "data_index": index,
                "case_id": meta.get("case_id", f"case_{index:04d}"),
                "raw_case_name": meta.get("raw_case_name", ""),
                "source_x_mm": meta.get("source_x_mm", ""),
                "source_y_mm": meta.get("source_y_mm", ""),
                "source_z_mm": meta.get("source_z_mm", ""),
                "leak_rate_ml_min": meta.get("leak_rate_ml_min", ""),
                **stats,
                "flags": ";".join(flags),
            }
            rows.append(row)

    flagged = [row for row in rows if row["flags"]]
    summary = {
        "h5": str(Path(args.h5)),
        "manifest": str(Path(args.manifest)) if args.manifest else None,
        "case_count": len(rows),
        "flagged_count": len(flagged),
        "flagged_cases": [
            {
                "data_index": row["data_index"],
                "case_id": row["case_id"],
                "raw_case_name": row["raw_case_name"],
                "flags": row["flags"],
                "min": row["min"],
                "max": row["max"],
                "mean": row["mean"],
                "std": row["std"],
            }
            for row in flagged
        ],
        "thresholds": {
            "constant_std_threshold": args.constant_std_threshold,
            "high_mean_threshold": args.high_mean_threshold,
            "high_max_threshold": args.high_max_threshold,
            "near_zero_max_threshold": args.near_zero_max_threshold,
        },
    }

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if flagged and args.fail_on_flag:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
