import argparse
import csv
import json
import os
from collections import Counter


SIZE_COLUMNS = ["space_size_x_m", "space_size_y_m", "space_size_z_m"]


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Assess whether current manifests can support size-generalization experiments.")
    parser.add_argument("--manifest_csv", action="append", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    all_rows = []
    manifest_summaries = []
    for path in args.manifest_csv:
        rows = load_rows(path)
        fieldnames = list(rows[0].keys()) if rows else []
        missing_size_cols = [c for c in SIZE_COLUMNS if c not in fieldnames]
        size_counter = Counter()
        if not missing_size_cols:
            for row in rows:
                size_counter[tuple(float(row[c]) for c in SIZE_COLUMNS)] += 1
        manifest_summaries.append(
            {
                "manifest_csv": path,
                "count": len(rows),
                "missing_size_columns": missing_size_cols,
                "size_groups": {",".join(map(str, k)): v for k, v in size_counter.items()},
            }
        )
        all_rows.extend(rows)

    available = all(SIZE_COLUMNS[0] in row for row in all_rows) if all_rows else False
    global_size_counter = Counter()
    if available:
        for row in all_rows:
            global_size_counter[tuple(float(row[c]) for c in SIZE_COLUMNS)] += 1

    report = {
        "size_columns": SIZE_COLUMNS,
        "manifests": manifest_summaries,
        "ready": len(global_size_counter) >= 2,
        "global_size_groups": {",".join(map(str, k)): v for k, v in global_size_counter.items()},
        "message": (
            "Current data can support size-generalization experiments."
            if len(global_size_counter) >= 2
            else "Current manifests do not yet contain at least two distinct space sizes; size-generalization experiments cannot be run meaningfully."
        ),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
