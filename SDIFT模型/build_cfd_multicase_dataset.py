import argparse
import csv
import glob
import json
import os
import re
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


CASE_NAME_RE = re.compile(r"^(?P<x>-?\d+),(?P<y>-?\d+),(?P<z>-?\d+),(?P<rate>\d+)mlmin$", re.IGNORECASE)
CASE_NAME_QXY_RE = re.compile(r"^Q(?P<rate>\d+)-X(?P<x>-?\d+)-Y(?P<y>-?\d+)-Fraction$", re.IGNORECASE)
CASE_NAME_PREFIX_RE = re.compile(r"^\d+,(?P<x>-?\d+),(?P<y>-?\d+),(?P<rate>\d+)$", re.IGNORECASE)
FRAME_RE = re.compile(r"-(\d+(?:\.\d+)?)(?:\(\d+\))?(?:\.0)?(?:\.baiduyun\.p\.downloading)?$", re.IGNORECASE)
DUPLICATE_COPY_RE = re.compile(r"\(\d+\)(?=\.0(?:\.baiduyun\.p\.downloading)?$|$)", re.IGNORECASE)


def normalize_axis(axis):
    axis = np.asarray(axis, dtype=np.float32)
    axis_min = axis.min()
    axis_max = axis.max()
    if axis_max == axis_min:
        return np.zeros_like(axis, dtype=np.float32)
    return ((axis - axis_min) / (axis_max - axis_min)).astype(np.float32)


def extract_frame_idx(path):
    match = FRAME_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot extract frame number from: {path}")
    return float(match.group(1))


def is_downloading_file(path):
    return os.path.basename(path).lower().endswith(".baiduyun.p.downloading")


def is_duplicate_copy(path):
    return DUPLICATE_COPY_RE.search(os.path.basename(path)) is not None


def read_xyzv(path):
    data = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def read_values(path):
    values = np.loadtxt(path, skiprows=1, usecols=(4,), dtype=np.float32)
    if values.ndim == 0:
        values = values[None]
    return values


def canonicalize_coordinates(x, y, z):
    u = x.astype(np.float32).copy()
    v = z.astype(np.float32).copy()
    w = y.astype(np.float32).copy()

    if u.min() >= -1e-3 and u.max() > 0.9:
        u -= 0.5
    if w.min() >= -1e-3 and w.max() > 0.79:
        w -= 0.4

    return u, v, w


def compute_idw_weights(points, query, k=8, power=2.0):
    tree = cKDTree(points)
    dist, idx = tree.query(query, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    dist = np.maximum(dist, 1e-12)
    weights = 1.0 / (dist ** power)
    zero_mask = dist <= 1e-12
    if np.any(zero_mask):
        rows = np.any(zero_mask, axis=1)
        weights[rows] = zero_mask[rows].astype(np.float32)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return idx.astype(np.int64), weights.astype(np.float32)


def resolve_case_data_dir(root_dir):
    current = root_dir
    visited = set()
    while True:
        current = os.path.abspath(current)
        if current in visited:
            return None
        visited.add(current)

        direct_files = [p for p in glob.glob(os.path.join(current, "*")) if os.path.isfile(p)]
        if direct_files:
            return current

        all_dir = os.path.join(current, "all")
        if os.path.isdir(all_dir):
            all_files = [p for p in glob.glob(os.path.join(all_dir, "*")) if os.path.isfile(p)]
            if all_files:
                return all_dir
            current = all_dir
            continue

        subdirs = sorted([os.path.join(current, name) for name in os.listdir(current) if os.path.isdir(os.path.join(current, name))])
        if len(subdirs) == 1:
            current = subdirs[0]
            continue
        return None


def discover_case_dirs(scan_roots):
    cases = []
    for root in scan_roots:
        if not os.path.isdir(root):
            continue

        direct_files = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isfile(p)]
        if direct_files:
            cases.append((os.path.basename(root), root))
            continue

        for name in sorted(os.listdir(root)):
            subdir = os.path.join(root, name)
            if not os.path.isdir(subdir):
                continue
            data_dir = resolve_case_data_dir(subdir)
            if data_dir:
                cases.append((os.path.basename(subdir), data_dir))
    dedup = {}
    for case_name, data_dir in cases:
        dedup[case_name] = data_dir
    return sorted(dedup.items(), key=lambda x: x[0])


def parse_case_name(case_name):
    normalized_case_name = case_name.replace(".-", ",-")
    match = CASE_NAME_RE.match(normalized_case_name)
    if match:
        return {
            "raw_case_name": normalized_case_name,
            "source_x_mm": int(match.group("x")),
            "source_y_mm": int(match.group("y")),
            "source_z_mm": int(match.group("z")),
            "leak_rate_ml_min": int(match.group("rate")),
        }

    match = CASE_NAME_QXY_RE.match(normalized_case_name)
    if match:
        return {
            "raw_case_name": normalized_case_name,
            "source_x_mm": int(match.group("x")),
            "source_y_mm": int(match.group("y")),
            "source_z_mm": 0,
            "leak_rate_ml_min": int(match.group("rate")),
        }

    match = CASE_NAME_PREFIX_RE.match(normalized_case_name)
    if match:
        return {
            "raw_case_name": normalized_case_name,
            "source_x_mm": int(match.group("x")),
            "source_y_mm": int(match.group("y")),
            "source_z_mm": 0,
            "leak_rate_ml_min": int(match.group("rate")),
        }

    return {
        "raw_case_name": normalized_case_name,
        "source_x_mm": None,
        "source_y_mm": None,
        "source_z_mm": None,
        "leak_rate_ml_min": None,
    }


def choose_preferred_entry(prev, cur):
    def score(item):
        return (
            0 if item["is_duplicate_copy"] else 1,
            0 if item["is_downloading"] else 1,
            item["size"],
            -len(os.path.basename(item["path"])),
        )

    return cur if score(cur) > score(prev) else prev


def collect_frame_entries(file_paths):
    by_time = {}
    ignored_downloading = 0
    ignored_bad_name = []
    discarded_duplicates = 0
    for path in file_paths:
        if is_downloading_file(path):
            ignored_downloading += 1
            continue
        try:
            t = extract_frame_idx(path)
        except Exception:
            ignored_bad_name.append(path)
            continue
        item = {
            "time": float(t),
            "path": path,
            "size": int(os.path.getsize(path)),
            "is_duplicate_copy": is_duplicate_copy(path),
            "is_downloading": False,
        }
        if t in by_time:
            chosen = choose_preferred_entry(by_time[t], item)
            if chosen is not by_time[t]:
                discarded_duplicates += 1
            else:
                discarded_duplicates += 1
            by_time[t] = chosen
        else:
            by_time[t] = item

    entries = [by_time[key] for key in sorted(by_time.keys())]
    return entries, ignored_downloading, discarded_duplicates, ignored_bad_name


def is_integer_time(value, tol=1e-6):
    return abs(value - round(value)) <= tol


def select_frame_entries(entries, selected_steps):
    if len(entries) < selected_steps:
        raise ValueError(f"only {len(entries)} unique readable frames")

    integer_entries = [item for item in entries if is_integer_time(item["time"])]
    integer_map = {int(round(item["time"])): item for item in integer_entries}
    expected_int = list(range(1, selected_steps + 1))
    if all(key in integer_map for key in expected_int):
        selected = [integer_map[key] for key in expected_int]
        return selected, np.asarray(expected_int, dtype=np.float32), "integer_aligned"

    selected = entries[:selected_steps]
    selected_t = np.asarray([item["time"] for item in selected], dtype=np.float32)
    return selected, selected_t, "first_unique"


def ensure_output_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Build a multi-case CFD training tensor from raw ASCII field files.")
    parser.add_argument("--scan_root", action="append", required=True, help="Directory to scan for CFD case folders. Repeat for multiple roots.")
    parser.add_argument("--include_case", action="append", default=[], help="Only convert case(s) with these directory names. Can be repeated.")
    parser.add_argument("--exclude_case", action="append", default=[], help="Exclude case(s) with these directory names. Can be repeated.")
    parser.add_argument("--out_h5", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--selected_steps", type=int, default=120)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--interp_k", type=int, default=8)
    parser.add_argument("--interp_power", type=float, default=2.0)
    parser.add_argument("--compression_level", type=int, default=1)
    parser.add_argument("--u_min", type=float, default=-0.5)
    parser.add_argument("--u_max", type=float, default=0.5)
    parser.add_argument("--v_min", type=float, default=0.0)
    parser.add_argument("--v_max", type=float, default=0.8)
    parser.add_argument("--w_min", type=float, default=-0.4)
    parser.add_argument("--w_max", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--case_start", type=int, default=0)
    parser.add_argument("--case_limit", type=int, default=0)
    args = parser.parse_args()

    for path in [args.out_h5, args.out_meta, args.out_manifest, args.out_report]:
        ensure_output_dir(path)
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    cases = discover_case_dirs(args.scan_root)
    include_cases = set(args.include_case or [])
    exclude_cases = set(args.exclude_case or [])
    if include_cases:
        cases = [item for item in cases if item[0] in include_cases]
    if exclude_cases:
        cases = [item for item in cases if item[0] not in exclude_cases]
    if args.case_start > 0:
        cases = cases[args.case_start:]
    if args.case_limit > 0:
        cases = cases[:args.case_limit]
    if not cases:
        raise SystemExit("No CFD cases discovered.")

    u_axis = np.linspace(args.u_min, args.u_max, args.bins, dtype=np.float32)
    v_axis = np.linspace(args.v_min, args.v_max, args.bins, dtype=np.float32)
    w_axis = np.linspace(args.w_min, args.w_max, args.bins, dtype=np.float32)
    uu, vv, ww = np.meshgrid(u_axis, v_axis, w_axis, indexing="ij")
    query = np.stack([uu.ravel(), vv.ravel(), ww.ravel()], axis=1).astype(np.float32)

    manifest_rows = []
    skipped_cases = []
    repaired_cases = []
    conversion_start = time.time()

    with h5py.File(args.out_h5, "w") as f:
        dset = f.create_dataset(
            "data",
            shape=(len(cases), args.selected_steps, args.bins, args.bins, args.bins),
            dtype="float32",
            chunks=(1, 1, args.bins, args.bins, args.bins),
            compression="gzip",
            compression_opts=args.compression_level,
        )

        valid_case_count = 0
        reference_t = None
        for case_idx, (case_name, data_dir) in enumerate(cases):
            file_paths = [p for p in glob.glob(os.path.join(data_dir, "*")) if os.path.isfile(p)]
            frame_entries, ignored_downloading, discarded_duplicates, ignored_bad_name = collect_frame_entries(file_paths)
            if len(frame_entries) < args.selected_steps:
                skipped_cases.append(
                    {
                        "case_name": case_name,
                        "reason": f"only {len(frame_entries)} unique readable frames",
                        "raw_file_count": len(file_paths),
                        "ignored_downloading_count": ignored_downloading,
                        "discarded_duplicate_count": discarded_duplicates,
                        "ignored_bad_name_count": len(ignored_bad_name),
                    }
                )
                continue

            selected_entries, selected_t, selection_mode = select_frame_entries(frame_entries, args.selected_steps)
            selected_files = [item["path"] for item in selected_entries]
            if reference_t is None:
                reference_t = selected_t
            elif not np.allclose(reference_t, selected_t):
                skipped_cases.append(
                    {
                        "case_name": case_name,
                        "reason": "time steps inconsistent with reference",
                        "selection_mode": selection_mode,
                        "selected_t_head": selected_t[:10].tolist(),
                    }
                )
                continue

            first_valid_idx = None
            last_valid_frame = None
            last_valid_t_idx = None
            pending_bad = []
            repaired_frames = []
            points = None
            u = v = w = None
            idx = None
            weights = None

            frame_iter = tqdm(
                enumerate(selected_files),
                total=len(selected_files),
                desc=f"[{valid_case_count + 1}/{len(cases)}] {case_name}",
                leave=False,
            )
            for t_idx, frame_path in frame_iter:
                try:
                    if points is None:
                        x, y, z, values = read_xyzv(frame_path)
                        u, v, w = canonicalize_coordinates(x, y, z)
                        points = np.stack([u, v, w], axis=1).astype(np.float32)
                        idx, weights = compute_idw_weights(points, query, k=args.interp_k, power=args.interp_power)
                    else:
                        values = read_values(frame_path)

                    if values.shape[0] != points.shape[0]:
                        raise ValueError(f"Value length mismatch in {frame_path}: {values.shape[0]} vs {points.shape[0]}")

                    frame = np.sum(values[idx] * weights, axis=1).reshape(args.bins, args.bins, args.bins).astype(np.float32)
                except Exception as exc:
                    pending_bad.append(
                        {
                            "t_idx": t_idx,
                            "time": float(selected_t[t_idx]),
                            "file": frame_path,
                            "reason": str(exc),
                        }
                    )
                    continue

                if first_valid_idx is None:
                    first_valid_idx = t_idx
                    if pending_bad:
                        for pending in pending_bad:
                            dset[valid_case_count, pending["t_idx"]] = frame
                            repaired_frames.append(
                                {
                                    **pending,
                                    "repair_mode": "copy_first_valid",
                                    "source_left_time": None,
                                    "source_right_time": float(selected_t[t_idx]),
                                }
                            )
                        pending_bad.clear()
                elif pending_bad:
                    gap = len(pending_bad) + 1
                    left_time = float(selected_t[last_valid_t_idx])
                    right_time = float(selected_t[t_idx])
                    for offset, pending in enumerate(pending_bad, start=1):
                        alpha = offset / gap
                        repaired = ((1.0 - alpha) * last_valid_frame + alpha * frame).astype(np.float32)
                        dset[valid_case_count, pending["t_idx"]] = repaired
                        repaired_frames.append(
                            {
                                **pending,
                                "repair_mode": "linear_interp",
                                "source_left_time": left_time,
                                "source_right_time": right_time,
                            }
                        )
                    pending_bad.clear()

                dset[valid_case_count, t_idx] = frame
                last_valid_frame = frame
                last_valid_t_idx = t_idx

            if points is None or first_valid_idx is None or last_valid_frame is None:
                skipped_cases.append({"case_name": case_name, "reason": "no readable frames in selected window"})
                continue

            if pending_bad:
                for pending in pending_bad:
                    dset[valid_case_count, pending["t_idx"]] = last_valid_frame
                    repaired_frames.append(
                        {
                            **pending,
                            "repair_mode": "copy_last_valid",
                            "source_left_time": float(selected_t[last_valid_t_idx]),
                            "source_right_time": None,
                        }
                    )
                pending_bad.clear()

            case_info = parse_case_name(case_name)
            case_info.update(
                {
                    "case_id": f"case_{valid_case_count:04d}",
                    "data_index": valid_case_count,
                    "data_dir": data_dir,
                    "raw_file_count": len(file_paths),
                    "available_unique_frame_count": len(frame_entries),
                    "selected_steps": args.selected_steps,
                    "frame_selection_mode": selection_mode,
                    "frame_time_start": float(selected_t[0]),
                    "frame_time_end": float(selected_t[-1]),
                    "raw_node_count": int(points.shape[0]),
                    "raw_u_min": float(u.min()),
                    "raw_u_max": float(u.max()),
                    "raw_v_min": float(v.min()),
                    "raw_v_max": float(v.max()),
                    "raw_w_min": float(w.min()),
                    "raw_w_max": float(w.max()),
                    "raw_total_bytes_selected": int(sum(os.path.getsize(p) for p in selected_files)),
                    "ignored_downloading_count": int(ignored_downloading),
                    "discarded_duplicate_count": int(discarded_duplicates),
                    "ignored_bad_name_count": int(len(ignored_bad_name)),
                    "repaired_frame_count": len(repaired_frames),
                    "repaired_frame_times": ";".join(str(int(item["time"])) for item in repaired_frames),
                }
            )
            manifest_rows.append(case_info)
            if repaired_frames:
                repaired_cases.append(
                    {
                        "case_name": case_name,
                        "case_id": case_info["case_id"],
                        "repaired_frames": repaired_frames,
                    }
                )
            valid_case_count += 1

        if valid_case_count == 0:
            raise SystemExit("No valid cases converted.")
        if valid_case_count < len(cases):
            dset.resize((valid_case_count, args.selected_steps, args.bins, args.bins, args.bins))

    if reference_t is None:
        raise SystemExit("No valid reference time axis available.")

    meta = {
        "u_ind_uni": normalize_axis(u_axis),
        "v_ind_uni": normalize_axis(v_axis),
        "w_ind_uni": normalize_axis(w_axis),
        "t_ind_uni": normalize_axis(reference_t),
        "u_ind_real": u_axis,
        "v_ind_real": v_axis,
        "w_ind_real": w_axis,
        "t_ind_real": reference_t,
        "mask_tr": np.ones((args.selected_steps, args.bins, args.bins, args.bins), dtype=np.int8),
    }
    np.save(args.out_meta, {"data": meta})

    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "scan_roots": args.scan_root,
        "case_count_discovered": len(cases),
        "case_count_converted": len(manifest_rows),
        "skipped_cases": skipped_cases,
        "repaired_cases": repaired_cases,
        "selected_steps": args.selected_steps,
        "bins": args.bins,
        "interp_k": args.interp_k,
        "interp_power": args.interp_power,
        "compression_level": args.compression_level,
        "canonical_bounds": {
            "u": [args.u_min, args.u_max],
            "v": [args.v_min, args.v_max],
            "w": [args.w_min, args.w_max],
        },
        "output_h5": args.out_h5,
        "output_meta": args.out_meta,
        "output_manifest": args.out_manifest,
        "output_report": args.out_report,
        "output_h5_size_gb": round(os.path.getsize(args.out_h5) / (1024 ** 3), 4) if os.path.exists(args.out_h5) else None,
        "output_meta_size_mb": round(os.path.getsize(args.out_meta) / (1024 ** 2), 4) if os.path.exists(args.out_meta) else None,
        "raw_selected_total_gb": round(sum(r["raw_total_bytes_selected"] for r in manifest_rows) / (1024 ** 3), 4),
        "estimated_float32_uncompressed_gb": round((len(manifest_rows) * args.selected_steps * args.bins * args.bins * args.bins * 4) / (1024 ** 3), 4),
        "conversion_wall_time_sec": round(time.time() - conversion_start, 2),
    }
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"converted_cases={len(manifest_rows)}")
    print(f"output_h5={args.out_h5}")
    print(f"output_meta={args.out_meta}")
    print(f"output_manifest={args.out_manifest}")
    print(f"output_report={args.out_report}")
    print(f"output_h5_size_gb={report['output_h5_size_gb']}")


if __name__ == "__main__":
    main()
