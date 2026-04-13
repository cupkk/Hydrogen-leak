import argparse
import glob
import os
import re

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def extract_idx(path):
    base = os.path.basename(path)
    match = re.search(r"--([0-9]+(?:\.[0-9]+)?)$", base)
    if not match:
        raise ValueError(f"Filename does not end with --<number>: {base}")
    return float(match.group(1))


def read_ascii(path, skiprows=1):
    data = np.loadtxt(path, skiprows=skiprows, usecols=(1, 2, 3, 4), dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 4:
        raise ValueError(f"Unexpected column count in {path}: {data.shape}")
    x, y, z, value = data.T
    return x, y, z, value


def normalize_axis(axis):
    axis_min = axis.min()
    axis_max = axis.max()
    if axis_max == axis_min:
        return np.zeros_like(axis, dtype=np.float32)
    return ((axis - axis_min) / (axis_max - axis_min)).astype(np.float32)


def build_axis(axis):
    return np.unique(axis)


def map_axis_indices(axis, values, tol=1e-6):
    idx = np.searchsorted(axis, values)
    if np.any(idx < 0) or np.any(idx >= axis.size):
        raise ValueError("Coordinate index out of range.")
    if not np.allclose(axis[idx], values, rtol=0.0, atol=tol):
        raise ValueError("Coordinate mismatch exceeds tolerance.")
    return idx.astype(np.int64)


def build_uniform_axis(min_val, max_val, count):
    return np.linspace(min_val, max_val, count, dtype=np.float32)


def idw_interpolate(tree, values, query_points, k=8, power=2.0):
    dist, idx = tree.query(query_points, k=k, workers=-1)
    if k == 1:
        return values[idx]
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
    vals = values[idx]
    return np.sum(vals * weights, axis=1)


def interpolate_to_grid(x, y, z, value, x_axis, y_axis, z_axis, method, k, power, chunk_size):
    if cKDTree is None:
        raise ImportError("scipy is required for interpolation; install scipy first.")
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    tree = cKDTree(points)
    gx, gy, gz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    query = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    out = np.empty(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        qp = query[start:end]
        if method == "nearest":
            _, idx = tree.query(qp, k=1, workers=-1)
            out[start:end] = value[idx]
        else:
            out[start:end] = idw_interpolate(tree, value, qp, k=k, power=power)
    return out.reshape(gx.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--pattern", default="400mlpermin--*")
    parser.add_argument("--out_h5", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--skiprows", type=int, default=1)
    parser.add_argument("--time_start", type=float, default=0.0)
    parser.add_argument("--time_step", type=float, default=1.0)
    parser.add_argument("--time_start_idx", type=int, default=0)
    parser.add_argument("--time_end_idx", type=int, default=-1)
    parser.add_argument("--time_stride", type=int, default=1)
    parser.add_argument("--mask_ratio", type=float, default=1.0)
    parser.add_argument("--grid_mode", choices=["rect", "bin", "interp"], default="rect")
    parser.add_argument("--axis_stride", type=int, default=1)
    parser.add_argument("--stride_x", type=int, default=0)
    parser.add_argument("--stride_y", type=int, default=0)
    parser.add_argument("--stride_z", type=int, default=0)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--bins_x", type=int, default=0)
    parser.add_argument("--bins_y", type=int, default=0)
    parser.add_argument("--bins_z", type=int, default=0)
    parser.add_argument("--interp_method", choices=["idw", "nearest"], default="idw")
    parser.add_argument("--interp_k", type=int, default=8)
    parser.add_argument("--interp_power", type=float, default=2.0)
    parser.add_argument("--interp_chunk", type=int, default=200000)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)), key=extract_idx)
    if not files:
        raise SystemExit("No input files found.")

    start = max(args.time_start_idx, 0)
    end = len(files) if args.time_end_idx < 0 else min(args.time_end_idx, len(files))
    files = files[start:end:args.time_stride]
    if not files:
        raise SystemExit("No input files selected after slicing.")

    x0, y0, z0, v0 = read_ascii(files[0], skiprows=args.skiprows)

    x_axis_full = y_axis_full = z_axis_full = None
    if args.grid_mode == "rect":
        stride_x = args.stride_x or args.axis_stride
        stride_y = args.stride_y or args.axis_stride
        stride_z = args.stride_z or args.axis_stride
        x_axis_full = build_axis(x0)
        y_axis_full = build_axis(y0)
        z_axis_full = build_axis(z0)
        x_axis = x_axis_full[::stride_x]
        y_axis = y_axis_full[::stride_y]
        z_axis = z_axis_full[::stride_z]
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        if nx * ny * nz > v0.size:
            raise ValueError("Grid is not rectilinear or has missing points.")
        x_edges = y_edges = z_edges = None
    elif args.grid_mode == "bin":
        bins_x = args.bins_x or args.bins
        bins_y = args.bins_y or args.bins
        bins_z = args.bins_z or args.bins
        x_edges = np.linspace(x0.min(), x0.max(), bins_x + 1, dtype=np.float32)
        y_edges = np.linspace(y0.min(), y0.max(), bins_y + 1, dtype=np.float32)
        z_edges = np.linspace(z0.min(), z0.max(), bins_z + 1, dtype=np.float32)
        x_axis = (x_edges[:-1] + x_edges[1:]) / 2
        y_axis = (y_edges[:-1] + y_edges[1:]) / 2
        z_axis = (z_edges[:-1] + z_edges[1:]) / 2
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    else:
        bins_x = args.bins_x or args.bins
        bins_y = args.bins_y or args.bins
        bins_z = args.bins_z or args.bins
        x_edges = y_edges = z_edges = None
        x_axis = build_uniform_axis(x0.min(), x0.max(), bins_x)
        y_axis = build_uniform_axis(y0.min(), y0.max(), bins_y)
        z_axis = build_uniform_axis(z0.min(), z0.max(), bins_z)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)

    t_count = len(files)
    t_real = args.time_start + np.arange(t_count, dtype=np.float32) * args.time_step
    t_uni = normalize_axis(t_real)

    u_real = x_axis.astype(np.float32)
    v_real = y_axis.astype(np.float32)
    w_real = z_axis.astype(np.float32)
    u_uni = normalize_axis(u_real)
    v_uni = normalize_axis(v_real)
    w_uni = normalize_axis(w_real)

    if args.mask_ratio >= 1.0:
        mask_tr = np.ones((t_count, nx, ny, nz), dtype=np.int8)
    else:
        mask_tr = np.zeros((t_count, nx, ny, nz), dtype=np.int8)
        num = int(mask_tr.size * args.mask_ratio)
        idx = np.random.choice(mask_tr.size, num, replace=False)
        mask_tr.flat[idx] = 1

    with h5py.File(args.out_h5, "w") as f:
        dset = f.create_dataset(
            "data",
            shape=(1, t_count, nx, ny, nz),
            dtype="float32",
            chunks=(1, 1, nx, ny, nz),
            compression="gzip",
        )

        if args.grid_mode == "rect":
            ix_full = map_axis_indices(x_axis_full, x0)
            iy_full = map_axis_indices(y_axis_full, y0)
            iz_full = map_axis_indices(z_axis_full, z0)
            mask = (ix_full % stride_x == 0) & (iy_full % stride_y == 0) & (iz_full % stride_z == 0)
            grid = np.zeros((nx, ny, nz), dtype=np.float32)
            grid[(ix_full[mask] // stride_x).astype(np.int64), (iy_full[mask] // stride_y).astype(np.int64), (iz_full[mask] // stride_z).astype(np.int64)] = v0[mask]
        elif args.grid_mode == "bin":
            ix = np.clip(np.digitize(x0, x_edges) - 1, 0, nx - 1)
            iy = np.clip(np.digitize(y0, y_edges) - 1, 0, ny - 1)
            iz = np.clip(np.digitize(z0, z_edges) - 1, 0, nz - 1)
            grid_sum = np.zeros((nx, ny, nz), dtype=np.float32)
            grid_cnt = np.zeros((nx, ny, nz), dtype=np.float32)
            np.add.at(grid_sum, (ix, iy, iz), v0)
            np.add.at(grid_cnt, (ix, iy, iz), 1.0)
            grid = grid_sum / np.maximum(grid_cnt, 1.0)
        else:
            grid = interpolate_to_grid(
                x0,
                y0,
                z0,
                v0,
                x_axis,
                y_axis,
                z_axis,
                method=args.interp_method,
                k=args.interp_k,
                power=args.interp_power,
                chunk_size=args.interp_chunk,
            )
        dset[0, 0] = grid

        for t, path in enumerate(files[1:], start=1):
            x, y, z, value = read_ascii(path, skiprows=args.skiprows)
            if args.grid_mode == "rect":
                ix_full = map_axis_indices(x_axis_full, x)
                iy_full = map_axis_indices(y_axis_full, y)
                iz_full = map_axis_indices(z_axis_full, z)
                mask = (ix_full % stride_x == 0) & (iy_full % stride_y == 0) & (iz_full % stride_z == 0)
                grid = np.zeros((nx, ny, nz), dtype=np.float32)
                grid[(ix_full[mask] // stride_x).astype(np.int64), (iy_full[mask] // stride_y).astype(np.int64), (iz_full[mask] // stride_z).astype(np.int64)] = value[mask]
            elif args.grid_mode == "bin":
                ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
                iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
                iz = np.clip(np.digitize(z, z_edges) - 1, 0, nz - 1)
                grid_sum = np.zeros((nx, ny, nz), dtype=np.float32)
                grid_cnt = np.zeros((nx, ny, nz), dtype=np.float32)
                np.add.at(grid_sum, (ix, iy, iz), value)
                np.add.at(grid_cnt, (ix, iy, iz), 1.0)
                grid = grid_sum / np.maximum(grid_cnt, 1.0)
            else:
                grid = interpolate_to_grid(
                    x,
                    y,
                    z,
                    value,
                    x_axis,
                    y_axis,
                    z_axis,
                    method=args.interp_method,
                    k=args.interp_k,
                    power=args.interp_power,
                    chunk_size=args.interp_chunk,
                )
            dset[0, t] = grid

    meta = {
        "u_ind_uni": u_uni,
        "v_ind_uni": v_uni,
        "w_ind_uni": w_uni,
        "t_ind_uni": t_uni,
        "u_ind_real": u_real,
        "v_ind_real": v_real,
        "w_ind_real": w_real,
        "t_ind_real": t_real,
        "mask_tr": mask_tr,
    }
    np.save(args.out_meta, {"data": meta})


if __name__ == "__main__":
    main()
