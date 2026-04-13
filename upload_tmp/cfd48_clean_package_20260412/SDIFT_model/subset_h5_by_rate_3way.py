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
    missing = [c for c in POSITION_COLUMNS + ["case_id", "data_index", "leak_rate_ml_min"] if c not in rows[0]]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    for row in rows:
        row["data_index"] = int(row["data_index"])
        row["leak_rate_ml_min"] = int(float(row["leak_rate_ml_min"]))
        for col in POSITION_COLUMNS:
            row[col] = int(float(row[col]))
    return rows


def rate_key(row):
    return int(row["leak_rate_ml_min"])


def parse_rates(values):
    rates = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                rates.append(int(float(part)))
    if not rates:
        raise ValueError("at least one leak rate must be provided")
    return rates


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
        item["rate_key"] = str(int(item["leak_rate_ml_min"]))
        enriched.append(item)
    return enriched


def summarize_rows(rows):
    counts = {}
    positions = {}
    for row in rows:
        rate = str(int(row["leak_rate_ml_min"]))
        pos = ",".join(str(row[col]) for col in POSITION_COLUMNS)
        counts[rate] = counts.get(rate, 0) + 1
        positions[pos] = positions.get(pos, 0) + 1
    return {
        "count": len(rows),
        "leak_rates": counts,
        "positions": positions,
        "case_ids": [row["case_id"] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split a multi-case HDF5 dataset into train/val/test subsets by held-out leak rates."
    )
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--dataset_key", default="data")
    parser.add_argument(
        "--test_rate",
        action="append",
        required=True,
        help="Held-out test leak rate in mL/min. Can be repeated or passed as a comma-separated list.",
    )
    parser.add_argument(
        "--val_rate",
        action="append",
        required=True,
        help="Held-out validation leak rate in mL/min. Can be repeated or passed as a comma-separated list.",
    )
    parser.add_argument("--out_train_h5", required=True)
    parser.add_argument("--out_train_manifest", required=True)
    parser.add_argument("--out_val_h5", required=True)
    parser.add_argument("--out_val_manifest", required=True)
    parser.add_argument("--out_test_h5", required=True)
    parser.add_argument("--out_test_manifest", required=True)
    parser.add_argument("--out_split_json", required=True)
    args = parser.parse_args()

    test_rates = set(parse_rates(args.test_rate))
    val_rates = set(parse_rates(args.val_rate))
    overlap = test_rates & val_rates
    if overlap:
        raise ValueError(f"val_rate and test_rate overlap: {sorted(overlap)}")

    rows = load_manifest(args.manifest_csv)
    train_rows = [row for row in rows if rate_key(row) not in test_rates and rate_key(row) not in val_rates]
    val_rows = [row for row in rows if rate_key(row) in val_rates]
    test_rows = [row for row in rows if rate_key(row) in test_rates]

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
        "val_leak_rates_ml_min": sorted(int(x) for x in val_rates),
        "test_leak_rates_ml_min": sorted(int(x) for x in test_rates),
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
