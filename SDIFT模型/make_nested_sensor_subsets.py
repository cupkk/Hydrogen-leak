import argparse
import csv
import json
import math
import os

import numpy as np


def parse_sensor_csv(path):
    data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if np.isnan(data).any() or data.shape[1] < 3:
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32, skip_header=1)
        if data.ndim == 1:
            data = data[None, :]
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] == 0 or data.shape[1] < 3:
        raise ValueError("sensor_csv must contain numeric x,y,z columns")
    return data[:, :3]


def deduplicate_points(points, tol=1e-6):
    if points.size == 0:
        return points, np.zeros((0,), dtype=np.int64)
    keys = np.round(points / max(float(tol), 1e-12)).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(unique_idx)
    return points[keep], keep


def farthest_point_order(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    n = points.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    centroid = points.mean(axis=0)
    d0 = np.linalg.norm(points - centroid[None, :], axis=1)
    order = [int(np.argmin(d0))]
    remaining = set(range(n))
    remaining.remove(order[0])

    min_dist = np.linalg.norm(points - points[order[0]][None, :], axis=1)
    while remaining:
        next_idx = max(remaining, key=lambda idx: (float(min_dist[idx]), -idx))
        order.append(int(next_idx))
        remaining.remove(next_idx)
        dist_new = np.linalg.norm(points - points[next_idx][None, :], axis=1)
        min_dist = np.minimum(min_dist, dist_new)
    return order


def save_csv(path, points):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.asarray(points, dtype=np.float32), delimiter=",", fmt="%.6f")


def main():
    parser = argparse.ArgumentParser(description="Generate nested sensor subset CSVs from a sensor pool.")
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--counts", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="sensors")
    parser.add_argument("--deduplicate", action="store_true", default=False)
    parser.add_argument("--dedup_tol", type=float, default=1e-6)
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    points = parse_sensor_csv(args.sensor_csv)
    original_count = int(points.shape[0])
    kept_indices = np.arange(original_count, dtype=np.int64)
    if args.deduplicate:
        points, kept_indices = deduplicate_points(points, tol=args.dedup_tol)

    order = farthest_point_order(points)
    ordered_points = points[order]
    ordered_source_indices = kept_indices[order]

    counts = sorted(set(int(c) for c in args.counts))
    if counts[0] <= 0:
        raise ValueError("counts must be positive")
    if counts[-1] > ordered_points.shape[0]:
        raise ValueError(f"requested {counts[-1]} sensors but only {ordered_points.shape[0]} are available")

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = []
    for count in counts:
        out_csv = os.path.join(args.out_dir, f"{args.prefix}_{count}.csv")
        save_csv(out_csv, ordered_points[:count])
        outputs.append(
            {
                "count": int(count),
                "csv": out_csv,
                "source_indices": ordered_source_indices[:count].astype(int).tolist(),
                "points": ordered_points[:count].astype(float).tolist(),
            }
        )
        print(f"saved: {out_csv}")

    summary = {
        "sensor_csv": args.sensor_csv,
        "original_count": original_count,
        "deduplicated_count": int(points.shape[0]),
        "counts": counts,
        "selection_order_source_indices": ordered_source_indices.astype(int).tolist(),
        "outputs": outputs,
    }
    summary_json = args.summary_json or os.path.join(args.out_dir, f"{args.prefix}_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
