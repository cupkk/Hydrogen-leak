import argparse
import csv
import json
import os

import h5py
import numpy as np
import scipy.io as sio


def point_weights_1d(axis):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1:
        raise ValueError("axis must be 1D")
    if axis.size == 0:
        raise ValueError("axis cannot be empty")
    if axis.size == 1:
        return np.ones((1,), dtype=np.float64)
    if not np.all(np.diff(axis) > 0):
        raise ValueError("axis must be strictly increasing")

    mid = 0.5 * (axis[:-1] + axis[1:])
    left = np.empty_like(axis)
    right = np.empty_like(axis)
    left[1:] = mid
    right[:-1] = mid
    left[0] = axis[0] - (mid[0] - axis[0])
    right[-1] = axis[-1] + (axis[-1] - mid[-1])
    return np.maximum(right - left, 0.0)


def load_truth(path, sample_index=0, dataset_key="data"):
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            return np.asarray(f[dataset_key][sample_index], dtype=np.float64)
    if path.endswith(".npy"):
        d = np.load(path, allow_pickle=True).item()
        data = d["data"]
        return np.asarray(data[sample_index], dtype=np.float64)
    raise ValueError("truth_path must be .h5 or .npy")


def load_recon(path, sample_index=0, recon_key="recon_list"):
    d = sio.loadmat(path)
    if recon_key not in d:
        raise KeyError(f"{recon_key} not found in {path}")
    arr = np.asarray(d[recon_key][sample_index], dtype=np.float64)
    return arr


def load_axes(meta_path, use_real_coords=True):
    meta = np.load(meta_path, allow_pickle=True).item()["data"]
    if use_real_coords:
        u = np.asarray(meta.get("u_ind_real", meta["u_ind_uni"]), dtype=np.float64)
        v = np.asarray(meta.get("v_ind_real", meta["v_ind_uni"]), dtype=np.float64)
        w = np.asarray(meta.get("w_ind_real", meta["w_ind_uni"]), dtype=np.float64)
        t = np.asarray(meta.get("t_ind_real", meta["t_ind_uni"]), dtype=np.float64)
    else:
        u = np.asarray(meta["u_ind_uni"], dtype=np.float64)
        v = np.asarray(meta["v_ind_uni"], dtype=np.float64)
        w = np.asarray(meta["w_ind_uni"], dtype=np.float64)
        t = np.asarray(meta["t_ind_uni"], dtype=np.float64)
    return u, v, w, t


def compute_mass_series(field, voxel_weight):
    return np.sum(field * voxel_weight[None, ...], axis=(1, 2, 3))


def masked_mean(values, mask):
    if mask is None:
        return float(np.mean(values))
    count = int(np.sum(mask))
    if count == 0:
        return None
    return float(np.mean(values[mask]))


def compute_metrics(pred, truth, voxel_weight=None, eps=1e-8, truth_threshold=1e-5):
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs truth {truth.shape}")

    diff = pred - truth
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2
    abs_truth = np.abs(truth)

    per_time_rmse = np.sqrt(np.mean(sq_diff, axis=(1, 2, 3)))
    per_time_mae = np.mean(abs_diff, axis=(1, 2, 3))
    per_time_rel_l1 = np.mean(abs_diff / (abs_truth + eps), axis=(1, 2, 3))
    active_mask = abs_truth >= truth_threshold

    per_time_rel_l1_active = []
    active_ratio_per_time = []
    rel_field = abs_diff / (abs_truth + eps)
    for i in range(pred.shape[0]):
        mask_t = active_mask[i]
        active_ratio_per_time.append(float(np.mean(mask_t)))
        per_time_rel_l1_active.append(masked_mean(rel_field[i], mask_t))

    out = {
        "global_rmse": float(np.sqrt(np.mean(sq_diff))),
        "global_mae": float(np.mean(abs_diff)),
        "global_rel_l1_mean": float(np.mean(abs_diff / (abs_truth + eps))),
        "global_rel_l2": float(np.linalg.norm(diff.ravel()) / (np.linalg.norm(truth.ravel()) + eps)),
        "truth_threshold": float(truth_threshold),
        "active_ratio_mean": float(np.mean(active_mask)),
        "per_time_rmse": per_time_rmse.tolist(),
        "per_time_mae": per_time_mae.tolist(),
        "per_time_rel_l1": per_time_rel_l1.tolist(),
        "per_time_rel_l1_active": per_time_rel_l1_active,
        "active_ratio_per_time": active_ratio_per_time,
        "peak_time_rmse": float(np.max(per_time_rmse)),
        "peak_time_rel_l1": float(np.max(per_time_rel_l1)),
        "mean_time_rmse": float(np.mean(per_time_rmse)),
        "mean_time_rel_l1": float(np.mean(per_time_rel_l1)),
    }
    global_rel_l1_active = masked_mean(rel_field, active_mask)
    if global_rel_l1_active is not None:
        out["global_rel_l1_active_mean"] = global_rel_l1_active
        out["peak_time_rel_l1_active"] = float(np.nanmax([x for x in per_time_rel_l1_active if x is not None]))
        valid = [x for x in per_time_rel_l1_active if x is not None]
        out["mean_time_rel_l1_active"] = float(np.mean(valid)) if valid else None

    if voxel_weight is not None:
        truth_mass = compute_mass_series(truth, voxel_weight)
        pred_mass = compute_mass_series(pred, voxel_weight)
        mass_abs = np.abs(pred_mass - truth_mass)
        mass_rel = mass_abs / (np.abs(truth_mass) + eps)
        out["mass"] = {
            "truth_series": truth_mass.tolist(),
            "pred_series": pred_mass.tolist(),
            "abs_error_series": mass_abs.tolist(),
            "rel_error_series": mass_rel.tolist(),
            "mean_rel_error": float(np.mean(mass_rel)),
            "peak_rel_error": float(np.max(mass_rel)),
        }

    return out


def write_per_time_csv(path, t_axis, metrics):
    rows = zip(
        range(len(metrics["per_time_rmse"])),
        t_axis,
        metrics["per_time_rmse"],
        metrics["per_time_mae"],
        metrics["per_time_rel_l1"],
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_index", "time_value", "rmse", "mae", "rel_l1"])
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation for reconstructed 3D concentration fields.")
    parser.add_argument("--recon_mat", required=True)
    parser.add_argument("--truth_path", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--recon_key", default="recon_list")
    parser.add_argument("--truth_dataset_key", default="data")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--use_real_coords", action="store_true", default=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--truth_threshold", type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    pred = load_recon(args.recon_mat, sample_index=args.sample_index, recon_key=args.recon_key)
    truth = load_truth(args.truth_path, sample_index=args.sample_index, dataset_key=args.truth_dataset_key)
    u, v, w, t = load_axes(args.meta, use_real_coords=args.use_real_coords)

    wu = point_weights_1d(u)
    wv = point_weights_1d(v)
    ww = point_weights_1d(w)
    voxel_weight = wu[:, None, None] * wv[None, :, None] * ww[None, None, :]

    metrics = compute_metrics(pred, truth, voxel_weight=voxel_weight, eps=args.eps, truth_threshold=args.truth_threshold)
    result = {
        "recon_mat": args.recon_mat,
        "truth_path": args.truth_path,
        "meta": args.meta,
        "sample_index": int(args.sample_index),
        "shape": list(pred.shape),
        "metrics": metrics,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.out_csv:
        write_per_time_csv(args.out_csv, t, metrics)

    print(f"saved: {args.out_json}")
    if args.out_csv:
        print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
