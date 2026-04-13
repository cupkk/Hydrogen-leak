import argparse
import csv
import json
import os

import h5py


POSITION_COLUMNS = ["source_x_mm", "source_y_mm", "source_z_mm"]


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    missing = [c for c in POSITION_COLUMNS + ["case_id", "data_index"] if c not in rows[0]]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    for row in rows:
        row["data_index"] = int(row["data_index"])
        for col in POSITION_COLUMNS:
            row[col] = int(float(row[col]))
    return rows


def position_key(row):
    return tuple(int(row[col]) for col in POSITION_COLUMNS)


def parse_positions(values):
    out = []
    for value in values:
        parts = [x.strip() for x in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"position must be x,y,z in mm, got: {value}")
        out.append(tuple(int(x) for x in parts))
    return out


def write_manifest(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def enrich_rows(rows):
    enriched = []
    for new_idx, row in enumerate(rows):
        item = dict(row)
        item["original_data_index"] = int(row["data_index"])
        item["data_index"] = int(new_idx)
        item["position_key"] = ",".join(str(item[col]) for col in POSITION_COLUMNS)
        enriched.append(item)
    return enriched


def summarize_rows(rows):
    counts = {}
    leak_rates = {}
    for row in rows:
        pos = ",".join(str(row[col]) for col in POSITION_COLUMNS)
        counts[pos] = counts.get(pos, 0) + 1
        q = str(int(float(row.get("leak_rate_ml_min", 0))))
        leak_rates[q] = leak_rates.get(q, 0) + 1
    return {
        "count": len(rows),
        "positions": counts,
        "leak_rates": leak_rates,
        "case_ids": [row["case_id"] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description="Split a multi-case HDF5 dataset into train/val/test subsets by held-out source positions.")
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--dataset_key", default="data")
    parser.add_argument("--test_position", action="append", required=True, help="Held-out test position as x,y,z in mm. Can be repeated.")
    parser.add_argument("--val_position", action="append", required=True, help="Held-out val position as x,y,z in mm. Can be repeated.")
    parser.add_argument("--out_train_h5", required=True)
    parser.add_argument("--out_train_manifest", required=True)
    parser.add_argument("--out_val_h5", required=True)
    parser.add_argument("--out_val_manifest", required=True)
    parser.add_argument("--out_test_h5", required=True)
    parser.add_argument("--out_test_manifest", required=True)
    parser.add_argument("--out_split_json", required=True)
    args = parser.parse_args()

    test_positions = set(parse_positions(args.test_position))
    val_positions = set(parse_positions(args.val_position))
    overlap = test_positions & val_positions
    if overlap:
        raise ValueError(f"val_position and test_position overlap: {sorted(overlap)}")

    rows = load_manifest(args.manifest_csv)
    train_rows = [row for row in rows if position_key(row) not in test_positions and position_key(row) not in val_positions]
    val_rows = [row for row in rows if position_key(row) in val_positions]
    test_rows = [row for row in rows if position_key(row) in test_positions]

    if not train_rows:
        raise ValueError("train split is empty")
    if not val_rows:
        raise ValueError("val split is empty")
    if not test_rows:
        raise ValueError("test split is empty")

    train_rows = enrich_rows(train_rows)
    val_rows = enrich_rows(val_rows)
    test_rows = enrich_rows(test_rows)

    for path in [
        args.out_train_h5,
        args.out_train_manifest,
        args.out_val_h5,
        args.out_val_manifest,
        args.out_test_h5,
        args.out_test_manifest,
        args.out_split_json,
    ]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    subset_h5(args.input_h5, args.dataset_key, args.out_train_h5, [row["original_data_index"] for row in train_rows])
    subset_h5(args.input_h5, args.dataset_key, args.out_val_h5, [row["original_data_index"] for row in val_rows])
    subset_h5(args.input_h5, args.dataset_key, args.out_test_h5, [row["original_data_index"] for row in test_rows])

    write_manifest(args.out_train_manifest, train_rows)
    write_manifest(args.out_val_manifest, val_rows)
    write_manifest(args.out_test_manifest, test_rows)

    summary = {
        "input_h5": args.input_h5,
        "manifest_csv": args.manifest_csv,
        "dataset_key": args.dataset_key,
        "val_positions_mm": [list(pos) for pos in sorted(val_positions)],
        "test_positions_mm": [list(pos) for pos in sorted(test_positions)],
        "train": summarize_rows(train_rows),
        "val": summarize_rows(val_rows),
        "test": summarize_rows(test_rows),
    }
    with open(args.out_split_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.out_train_h5}")
    print(f"saved: {args.out_val_h5}")
    print(f"saved: {args.out_test_h5}")
    print(f"saved: {args.out_split_json}")


if __name__ == "__main__":
    main()
