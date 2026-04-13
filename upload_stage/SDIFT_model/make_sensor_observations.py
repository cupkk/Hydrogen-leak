import argparse

import h5py
import numpy as np


def load_meta(meta_path):
    d = np.load(meta_path, allow_pickle=True).item()["data"]
    u_uni = d["u_ind_uni"]
    v_uni = d["v_ind_uni"]
    w_uni = d["w_ind_uni"]
    t_uni = d["t_ind_uni"]
    u_real = d.get("u_ind_real", u_uni)
    v_real = d.get("v_ind_real", v_uni)
    w_real = d.get("w_ind_real", w_uni)
    t_real = d.get("t_ind_real", t_uni)
    return u_uni, v_uni, w_uni, t_uni, u_real, v_real, w_real, t_real


def nearest_index(axis, value):
    return int(np.argmin(np.abs(axis - value)))


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
        raise ValueError("sensor_csv must contain numeric x,y,z columns.")
    return data[:, :3]


def deduplicate_points(points, tol=1e-6):
    if points.size == 0:
        return points, np.zeros((0,), dtype=np.int64), 0
    scale = np.maximum(tol, 1e-12)
    keys = np.round(points / scale).astype(np.int64)
    _, unique_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    keep = np.sort(unique_idx)
    dup_count = int(points.shape[0] - keep.shape[0])
    return points[keep], inverse, dup_count


def compute_trilinear_weights(x, y, z, x_axis, y_axis, z_axis, clip_outside=False):
    if not (np.all(np.diff(x_axis) > 0) and np.all(np.diff(y_axis) > 0) and np.all(np.diff(z_axis) > 0)):
        raise ValueError("Axes must be strictly increasing for interpolation.")

    if clip_outside:
        x = np.clip(x, x_axis[0], x_axis[-1])
        y = np.clip(y, y_axis[0], y_axis[-1])
        z = np.clip(z, z_axis[0], z_axis[-1])
    else:
        if np.any(x < x_axis[0]) or np.any(x > x_axis[-1]):
            raise ValueError("Sensor x outside grid bounds.")
        if np.any(y < y_axis[0]) or np.any(y > y_axis[-1]):
            raise ValueError("Sensor y outside grid bounds.")
        if np.any(z < z_axis[0]) or np.any(z > z_axis[-1]):
            raise ValueError("Sensor z outside grid bounds.")

    ix1 = np.searchsorted(x_axis, x, side="right")
    iy1 = np.searchsorted(y_axis, y, side="right")
    iz1 = np.searchsorted(z_axis, z, side="right")
    ix1 = np.clip(ix1, 1, len(x_axis) - 1)
    iy1 = np.clip(iy1, 1, len(y_axis) - 1)
    iz1 = np.clip(iz1, 1, len(z_axis) - 1)

    ix0 = ix1 - 1
    iy0 = iy1 - 1
    iz0 = iz1 - 1

    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    y0 = y_axis[iy0]
    y1 = y_axis[iy1]
    z0 = z_axis[iz0]
    z1 = z_axis[iz1]

    tx = np.zeros_like(x, dtype=np.float32)
    ty = np.zeros_like(y, dtype=np.float32)
    tz = np.zeros_like(z, dtype=np.float32)
    np.divide((x - x0), (x1 - x0), out=tx, where=(x1 != x0))
    np.divide((y - y0), (y1 - y0), out=ty, where=(y1 != y0))
    np.divide((z - z0), (z1 - z0), out=tz, where=(z1 != z0))
    return ix0, ix1, iy0, iy1, iz0, iz1, tx, ty, tz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_h5", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset_key", default="data")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sensor_csv", default="")
    parser.add_argument("--num_sensors", type=int, default=0)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--sensor_unit", choices=["m", "mm"], default="m")
    parser.add_argument("--deduplicate", action="store_true", default=False)
    parser.add_argument("--dedup_tol", type=float, default=1e-6)
    parser.add_argument("--save_clean_sensor_csv", default="")
    parser.add_argument("--check_bounds", action="store_true", default=False)
    parser.add_argument("--sample_mode", choices=["nearest", "trilinear"], default="trilinear")
    parser.add_argument("--clip_outside", action="store_true", default=False)
    args = parser.parse_args()

    u_uni, v_uni, w_uni, t_uni, u_real, v_real, w_real, t_real = load_meta(args.metadata_path)

    if args.sensor_csv:
        sensor_xyz_real = parse_sensor_csv(args.sensor_csv)
        if args.sensor_unit == "mm":
            sensor_xyz_real = sensor_xyz_real / 1000.0
        if args.deduplicate:
            sensor_xyz_real, _, dup_count = deduplicate_points(sensor_xyz_real, tol=args.dedup_tol)
            print(f"deduplicate: removed {dup_count} duplicated sensors")
        if args.check_bounds:
            x_ok = (sensor_xyz_real[:, 0] >= u_real.min()) & (sensor_xyz_real[:, 0] <= u_real.max())
            y_ok = (sensor_xyz_real[:, 1] >= v_real.min()) & (sensor_xyz_real[:, 1] <= v_real.max())
            z_ok = (sensor_xyz_real[:, 2] >= w_real.min()) & (sensor_xyz_real[:, 2] <= w_real.max())
            ok = x_ok & y_ok & z_ok
            bad = np.where(~ok)[0]
            if bad.size > 0:
                raise ValueError(f"{bad.size} sensors are outside grid bounds. First bad index: {int(bad[0])}")
        if args.save_clean_sensor_csv:
            np.savetxt(args.save_clean_sensor_csv, sensor_xyz_real, delimiter=",", fmt="%.6f")
    else:
        if args.num_sensors <= 0:
            raise ValueError("Provide --sensor_csv or --num_sensors.")
        rng = np.random.default_rng(args.seed)
        idx_u = rng.integers(0, len(u_real), size=args.num_sensors)
        idx_v = rng.integers(0, len(v_real), size=args.num_sensors)
        idx_w = rng.integers(0, len(w_real), size=args.num_sensors)
        sensor_xyz_real = np.stack([u_real[idx_u], v_real[idx_v], w_real[idx_w]], axis=1)

    sensor_idx = np.empty((sensor_xyz_real.shape[0], 3), dtype=np.int64)
    for i, (x, y, z) in enumerate(sensor_xyz_real):
        sensor_idx[i, 0] = nearest_index(u_real, x)
        sensor_idx[i, 1] = nearest_index(v_real, y)
        sensor_idx[i, 2] = nearest_index(w_real, z)

    sensor_xyz_uni = np.stack(
        [
            np.interp(sensor_xyz_real[:, 0], u_real, u_uni),
            np.interp(sensor_xyz_real[:, 1], v_real, v_uni),
            np.interp(sensor_xyz_real[:, 2], w_real, w_uni),
        ],
        axis=1,
    ).astype(np.float32)

    with h5py.File(args.field_h5, "r") as f:
        dset = f[args.dataset_key]
        if args.sample_index >= dset.shape[0]:
            raise ValueError("sample_index out of range")
        t_count = dset.shape[1]
        y = np.zeros((t_count, sensor_xyz_real.shape[0]), dtype=np.float32)
        if args.sample_mode == "trilinear":
            ix0, ix1, iy0, iy1, iz0, iz1, tx, ty, tz = compute_trilinear_weights(
                sensor_xyz_real[:, 0],
                sensor_xyz_real[:, 1],
                sensor_xyz_real[:, 2],
                u_real,
                v_real,
                w_real,
                clip_outside=args.clip_outside,
            )
        for t in range(t_count):
            frame = dset[args.sample_index, t]
            if args.sample_mode == "trilinear":
                f000 = frame[ix0, iy0, iz0]
                f100 = frame[ix1, iy0, iz0]
                f010 = frame[ix0, iy1, iz0]
                f110 = frame[ix1, iy1, iz0]
                f001 = frame[ix0, iy0, iz1]
                f101 = frame[ix1, iy0, iz1]
                f011 = frame[ix0, iy1, iz1]
                f111 = frame[ix1, iy1, iz1]
                c00 = f000 * (1 - tx) + f100 * tx
                c10 = f010 * (1 - tx) + f110 * tx
                c01 = f001 * (1 - tx) + f101 * tx
                c11 = f011 * (1 - tx) + f111 * tx
                c0 = c00 * (1 - ty) + c10 * ty
                c1 = c01 * (1 - ty) + c11 * ty
                y[t] = c0 * (1 - tz) + c1 * tz
            else:
                y[t] = frame[sensor_idx[:, 0], sensor_idx[:, 1], sensor_idx[:, 2]]

    out = {
        "sensor_xyz": sensor_xyz_uni,
        "sensor_xyz_real": sensor_xyz_real.astype(np.float32),
        "sensor_idx": sensor_idx,
        "t": t_uni.astype(np.float32),
        "t_real": t_real.astype(np.float32),
        "y": y,
    }
    np.save(args.out, out)


if __name__ == "__main__":
    main()
