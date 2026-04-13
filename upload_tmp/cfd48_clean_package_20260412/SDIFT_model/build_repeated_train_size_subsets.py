import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import h5py
import numpy as np


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


def leak_rate_key(row):
    return int(row.get("leak_rate_ml_min", 0))


def is_low_rate(row, low_rates):
    return float(leak_rate_key(row)) in low_rates


def enrich_rows(rows):
    enriched = []
    for new_idx, row in enumerate(rows):
        item = dict(row)
        item["original_data_index"] = int(row["data_index"])
        item["data_index"] = int(new_idx)
        item["position_key"] = ",".join(str(item[col]) for col in POSITION_COLUMNS)
        enriched.append(item)
    return enriched


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


def allocate_counts(counts, total, ensure_nonzero=False):
    keys = list(counts.keys())
    values = np.asarray([counts[k] for k in keys], dtype=np.float64)
    if int(values.sum()) <= 0:
        return {k: 0 for k in keys}
    raw = total * values / values.sum()
    base = np.floor(raw).astype(int)
    if ensure_nonzero and total >= len(keys):
        for i, v in enumerate(values):
            if v > 0 and base[i] == 0:
                base[i] = 1
    diff = int(total - base.sum())
    remainders = raw - np.floor(raw)
    order = list(np.argsort(-remainders))
    while diff > 0:
        progressed = False
        for idx in order:
            if values[idx] <= 0:
                continue
            base[idx] += 1
            diff -= 1
            progressed = True
            if diff <= 0:
                break
        if not progressed:
            break
    while diff < 0:
        order_small = list(np.argsort(remainders))
        progressed = False
        for idx in order_small:
            floor_min = 1 if ensure_nonzero and total >= len(keys) and values[idx] > 0 else 0
            if base[idx] > floor_min:
                base[idx] -= 1
                diff += 1
                progressed = True
                if diff >= 0:
                    break
        if not progressed:
            break
    return {k: int(base[i]) for i, k in enumerate(keys)}


def choose_one(rows, rng, preferred_rates=None):
    if not rows:
        raise ValueError("choose_one received empty rows")
    preferred_rates = preferred_rates or set()
    order = list(range(len(rows)))
    rng.shuffle(order)
    ranked = sorted(
        (rows[i] for i in order),
        key=lambda r: (
            0 if leak_rate_key(r) in preferred_rates else 1,
            leak_rate_key(r),
            str(r.get("case_id", "")),
            int(r["data_index"]),
        ),
    )
    return ranked[0]


def sample_balanced_subset(rows, size, low_rates, seed):
    rng = np.random.default_rng(seed)
    low_rates = set(float(x) for x in low_rates)
    by_position = defaultdict(list)
    for row in rows:
        by_position[position_key(row)].append(row)

    selected = []
    selected_ids = set()
    size = int(size)
    if size <= 0:
        return []

    position_counts = {k: len(v) for k, v in by_position.items()}
    pos_quota = allocate_counts(position_counts, size, ensure_nonzero=True)

    low_pool = [row for row in rows if is_low_rate(row, low_rates)]
    global_low_target = int(round(size * len(low_pool) / max(len(rows), 1)))
    if low_pool and global_low_target <= 0:
        global_low_target = 1

    low_counts_by_pos = {k: sum(1 for row in v if is_low_rate(row, low_rates)) for k, v in by_position.items()}
    low_quota_by_pos = allocate_counts(low_counts_by_pos, min(global_low_target, sum(low_counts_by_pos.values())), ensure_nonzero=False)
    for pos in list(low_quota_by_pos.keys()):
        low_quota_by_pos[pos] = min(low_quota_by_pos[pos], pos_quota.get(pos, 0), low_counts_by_pos.get(pos, 0))

    # Phase 1: ensure position coverage and preserve low-rate cases where possible.
    for pos in sorted(by_position.keys()):
        if pos_quota.get(pos, 0) <= 0:
            continue
        low_rows = [row for row in by_position[pos] if is_low_rate(row, low_rates)]
        other_rows = [row for row in by_position[pos] if not is_low_rate(row, low_rates)]
        preferred_low = low_quota_by_pos.get(pos, 0) > 0 and low_rows
        chosen = choose_one(low_rows if preferred_low else (other_rows or low_rows), rng, preferred_rates=low_rates)
        selected.append(chosen)
        selected_ids.add(int(chosen["data_index"]))

    # Track counts after position seeding.
    pos_selected = Counter(position_key(row) for row in selected)
    rate_selected = Counter(leak_rate_key(row) for row in selected)
    low_selected = sum(1 for row in selected if is_low_rate(row, low_rates))
    rate_counts = Counter(leak_rate_key(row) for row in rows)
    rate_target = allocate_counts(rate_counts, size, ensure_nonzero=False)

    remaining_rows = [row for row in rows if int(row["data_index"]) not in selected_ids]
    while len(selected) < size:
        best_row = None
        best_score = None
        for row in remaining_rows:
            pos = position_key(row)
            rate = leak_rate_key(row)
            low_flag = is_low_rate(row, low_rates)
            pos_need = max(pos_quota.get(pos, 0) - pos_selected.get(pos, 0), 0)
            rate_need = max(rate_target.get(rate, 0) - rate_selected.get(rate, 0), 0)
            low_need = max(global_low_target - low_selected, 0)
            score = 0.0
            score += 8.0 * pos_need
            score += 4.0 * rate_need
            if low_flag:
                score += 6.0 * low_need
            score += float(rng.random()) * 1e-3
            candidate = (
                -score,
                abs(pos_need - 1),
                abs(rate_need - 1),
                0 if low_flag else 1,
                rate,
                str(row.get("case_id", "")),
            )
            if best_score is None or candidate < best_score:
                best_score = candidate
                best_row = row
        if best_row is None:
            break
        selected.append(best_row)
        selected_ids.add(int(best_row["data_index"]))
        pos_selected[position_key(best_row)] += 1
        rate_selected[leak_rate_key(best_row)] += 1
        if is_low_rate(best_row, low_rates):
            low_selected += 1
        remaining_rows = [row for row in remaining_rows if int(row["data_index"]) not in selected_ids]

    if len(selected) != size:
        raise ValueError(f"Unable to sample subset of size {size}; got {len(selected)}")

    selected.sort(
        key=lambda r: (
            POSITION_COLUMNS and tuple(int(r[col]) for col in POSITION_COLUMNS),
            int(r.get("leak_rate_ml_min", 0)),
            str(r.get("case_id", "")),
            int(r["data_index"]),
        )
    )
    return selected


def summarize_rows(rows, low_rates):
    by_position = Counter(",".join(str(row[col]) for col in POSITION_COLUMNS) for row in rows)
    by_rate = Counter(int(row.get("leak_rate_ml_min", 0)) for row in rows)
    low_rates = set(float(x) for x in low_rates)
    return {
        "count": len(rows),
        "positions": dict(by_position),
        "leak_rates": dict(by_rate),
        "low_rate_count": int(sum(1 for row in rows if is_low_rate(row, low_rates))),
        "case_ids": [row.get("case_id", "") for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description="Create repeated, approximately stratified train-size subsets.")
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_key", default="data")
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--low_rates", type=float, nargs="+", default=[50.0, 100.0])
    parser.add_argument("--seed", type=int, default=231)
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
        "repeats": int(args.repeats),
        "low_rates": list(args.low_rates),
        "subsets": [],
    }

    for size in sizes:
        for repeat in range(int(args.repeats)):
            subset_rows = sample_balanced_subset(rows, size=size, low_rates=args.low_rates, seed=args.seed + size * 100 + repeat)
            subset_rows = enrich_rows(subset_rows)
            subset_dir = os.path.join(args.out_dir, f"size_{size:03d}", f"repeat_{repeat:02d}")
            os.makedirs(subset_dir, exist_ok=True)
            out_h5 = os.path.join(subset_dir, f"{args.prefix}_{size:03d}_r{repeat:02d}.h5")
            out_manifest = os.path.join(subset_dir, f"{args.prefix}_{size:03d}_r{repeat:02d}_manifest.csv")

            subset_h5(args.input_h5, args.dataset_key, out_h5, [row["original_data_index"] for row in subset_rows])
            write_manifest(out_manifest, subset_rows)

            item = {
                "size": int(size),
                "repeat": int(repeat),
                "h5": out_h5,
                "manifest": out_manifest,
                "summary": summarize_rows(subset_rows, args.low_rates),
            }
            summary["subsets"].append(item)
            print(f"saved: {out_h5}")
            print(f"saved: {out_manifest}")

    summary_json = os.path.join(args.out_dir, f"{args.prefix}_repeated_subsets_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
