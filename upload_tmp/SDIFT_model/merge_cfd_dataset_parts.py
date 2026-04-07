import argparse
import csv
import json
import os

import h5py
import numpy as np


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_fieldnames(rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def main():
    parser = argparse.ArgumentParser(description="Merge partial CFD HDF5 datasets into one dataset.")
    parser.add_argument("--part_h5", action="append", required=True)
    parser.add_argument("--part_manifest", action="append", required=True)
    parser.add_argument("--meta_template", required=True)
    parser.add_argument("--out_h5", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    if len(args.part_h5) != len(args.part_manifest):
        raise ValueError("part_h5 and part_manifest must have the same count")

    for path in [args.out_h5, args.out_meta, args.out_manifest, args.out_report]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    arrays = []
    manifests = []
    for h5_path, manifest_path in zip(args.part_h5, args.part_manifest):
        with h5py.File(h5_path, "r") as f:
            arrays.append(np.asarray(f["data"], dtype=np.float32))
        manifests.extend(load_manifest(manifest_path))

    merged = np.concatenate(arrays, axis=0)

    with h5py.File(args.out_h5, "w") as f:
        f.create_dataset(
            "data",
            data=merged,
            dtype="float32",
            chunks=(1, 1, merged.shape[2], merged.shape[3], merged.shape[4]),
            compression="gzip",
            compression_opts=1,
        )

    meta = np.load(args.meta_template, allow_pickle=True)
    np.save(args.out_meta, meta)

    for i, row in enumerate(manifests):
        row["case_id"] = f"case_{i:04d}"
        row["data_index"] = i

    manifest_fieldnames = collect_fieldnames(manifests)
    for row in manifests:
        for key in manifest_fieldnames:
            row.setdefault(key, "")

    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(manifests)

    report = {
        "part_h5": args.part_h5,
        "part_manifest": args.part_manifest,
        "meta_template": args.meta_template,
        "merged_case_count": int(merged.shape[0]),
        "shape": list(merged.shape),
        "out_h5_size_gb": round(os.path.getsize(args.out_h5) / (1024 ** 3), 4),
        "out_meta_size_mb": round(os.path.getsize(args.out_meta) / (1024 ** 2), 4),
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"merged_cases={merged.shape[0]}")
    print(f"out_h5={args.out_h5}")


if __name__ == "__main__":
    main()
