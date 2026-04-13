import argparse
import csv
import json
import os
from collections import defaultdict

import h5py


POSITION_COLUMNS = ["source_x_mm", "source_y_mm", "source_z_mm"]


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    for row in rows:
        row["data_index"] = int(row["data_index"])
        if "leak_rate_ml_min" in row and row["leak_rate_ml_min"] != "":
            row["leak_rate_ml_min"] = int(float(row["leak_rate_ml_min"]))
        for col in POSITION_COLUMNS:
            row[col] = int(float(row[col]))
    return rows


def position_key(row):
    return tuple(int(row[col]) for col in POSITION_COLUMNS)


def subset_h5(input_h5, dataset_key, out_h5, indices):
    with h5py.File(input_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        dset = src[dataset_key]
        sample_shape = dset.shape[1:]
        out = dst.create_dataset(
            dataset_key,
            shape=(len(indices),) + sample_shape,
            dtype=dset.dtype,
            chunks=(1,) + sample_shape,
            compression="gzip",
            compression_opts=1,
        )
        for out_idx, src_idx in enumerate(indices):
            out[out_idx] = dset[src_idx]


def write_manifest(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def enrich_rows(rows):
    enriched = []
    for new_idx, row in enumerate(rows):
        item = dict(row)
        item["original_data_index"] = int(row["data_index"])
        item["data_index"] = int(new_idx)
        item["position_key"] = ",".join(str(item[col]) for col in POSITION_COLUMNS)
        enriched.append(item)
    return enriched


def round_robin_select(rows, size):
    groups = defaultdict(list)
    for row in rows:
        groups[position_key(row)].append(row)

    ordered_groups = []
    for key in sorted(groups.keys()):
        group_rows = sorted(
            groups[key],
            key=lambda r: (
                int(r.get("leak_rate_ml_min", 0)),
                str(r.get("case_id", "")),
                int(r["data_index"]),
            ),
        )
        ordered_groups.append((key, group_rows))

    selected = []
    cursors = {key: 0 for key, _ in ordered_groups}
    while len(selected) < size:
        progress = False
        for key, group_rows in ordered_groups:
            cursor = cursors[key]
            if cursor >= len(group_rows):
                continue
            selected.append(group_rows[cursor])
            cursors[key] = cursor + 1
            progress = True
            if len(selected) >= size:
                break
        if not progress:
            break

    if len(selected) != size:
        raise ValueError(f"Unable to select {size} rows from {len(rows)} available rows")
    return selected


def summarize_rows(rows):
    by_position = defaultdict(int)
    leak_rates = defaultdict(list)
    for row in rows:
        key = ",".join(str(row[col]) for col in POSITION_COLUMNS)
        by_position[key] += 1
        leak_rates[key].append(int(row.get("leak_rate_ml_min", 0)))
    return {
        "count": len(rows),
        "positions": dict(by_position),
        "leak_rates_by_position": {k: sorted(v) for k, v in leak_rates.items()},
        "case_ids": [row.get("case_id", "") for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description="Create deterministic train-size subsets from a training HDF5/manifest pair.")
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_key", default="data")
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--prefix", default="train")
    args = parser.parse_args()

    rows = load_manifest(args.manifest_csv)
    total_count = len(rows)
    sizes = sorted(set(int(s) for s in args.sizes))
    if sizes[0] <= 0:
        raise ValueError("sizes must be positive")
    if sizes[-1] > total_count:
        raise ValueError(f"requested subset size {sizes[-1]} but only {total_count} rows exist")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "input_h5": args.input_h5,
        "manifest_csv": args.manifest_csv,
        "total_count": total_count,
        "subsets": [],
    }

    for size in sizes:
        subset_rows = round_robin_select(rows, size)
        subset_rows = enrich_rows(subset_rows)

        subset_dir = os.path.join(args.out_dir, f"size_{size:03d}")
        os.makedirs(subset_dir, exist_ok=True)
        out_h5 = os.path.join(subset_dir, f"{args.prefix}_{size:03d}.h5")
        out_manifest = os.path.join(subset_dir, f"{args.prefix}_{size:03d}_manifest.csv")

        subset_h5(args.input_h5, args.dataset_key, out_h5, [row["original_data_index"] for row in subset_rows])
        write_manifest(out_manifest, subset_rows)

        item = {
            "size": int(size),
            "h5": out_h5,
            "manifest": out_manifest,
            "summary": summarize_rows(subset_rows),
        }
        summary["subsets"].append(item)
        print(f"saved: {out_h5}")
        print(f"saved: {out_manifest}")

    summary_json = os.path.join(args.out_dir, f"{args.prefix}_subsets_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
