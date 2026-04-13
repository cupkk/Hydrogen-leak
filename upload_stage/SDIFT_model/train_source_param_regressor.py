import argparse
import csv
import json
import os

import h5py
import numpy as np
import scipy.io as sio

from param_regressor_utils import (
    build_core_features,
    build_sensor_features,
    fit_ridge_multioutput,
    predict_ridge_multioutput,
    regression_metrics,
    target_from_manifest_row,
)
from make_sensor_observations import compute_trilinear_weights, load_meta, parse_sensor_csv


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    for row in rows:
        row["data_index"] = int(row["data_index"])
        if "original_data_index" in row and row["original_data_index"] != "":
            row["original_data_index"] = int(row["original_data_index"])
        row["source_x_mm"] = int(float(row["source_x_mm"]))
        row["source_y_mm"] = int(float(row["source_y_mm"]))
        row["source_z_mm"] = int(float(row["source_z_mm"]))
        row["leak_rate_ml_min"] = float(row["leak_rate_ml_min"])
    return rows


def row_lookup_index(row):
    if "original_data_index" in row and row["original_data_index"] != "":
        return int(row["original_data_index"])
    return int(row["data_index"])


def choose_index_column(rows, dataset_length):
    if not rows:
        return "data_index"
    has_original = all(("original_data_index" in row and row["original_data_index"] != "") for row in rows)
    if not has_original:
        return "data_index"
    original_max = max(int(row["original_data_index"]) for row in rows)
    if original_max < int(dataset_length):
        return "original_data_index"
    return "data_index"


def load_case_field(field_h5_path, case_index):
    with h5py.File(field_h5_path, "r") as f:
        dset = f["data"]
        if case_index >= dset.shape[0]:
            raise ValueError("case index out of range")
        return np.asarray(dset[case_index], dtype=np.float64)


def extract_sensor_observation(field_3d, sensor_xyz_real, u_real, v_real, w_real, observed_time_steps=0, sample_mode="trilinear"):
    field_3d = np.asarray(field_3d, dtype=np.float64)
    if field_3d.ndim != 4:
        raise ValueError("field must be [T, U, V, W]")
    t_count = field_3d.shape[0]
    if observed_time_steps and observed_time_steps > 0:
        t_count = min(int(observed_time_steps), t_count)
        field_3d = field_3d[:t_count]

    if sample_mode != "trilinear":
        raise ValueError("only trilinear observation extraction is supported in this trainer")

    ix0, ix1, iy0, iy1, iz0, iz1, tx, ty, tz = compute_trilinear_weights(
        sensor_xyz_real[:, 0],
        sensor_xyz_real[:, 1],
        sensor_xyz_real[:, 2],
        u_real,
        v_real,
        w_real,
        clip_outside=False,
    )
    obs = np.zeros((t_count, sensor_xyz_real.shape[0]), dtype=np.float64)
    for t in range(t_count):
        frame = field_3d[t]
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
        obs[t] = c0 * (1 - tz) + c1 * tz
    return obs


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight source-position and leak-rate regressor on FTM core features.")
    parser.add_argument("--core_path", default="")
    parser.add_argument("--test_core_path", default="")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--input_mode", choices=["core", "sensor", "hybrid"], default="core")
    parser.add_argument("--field_h5", default="")
    parser.add_argument("--test_field_h5", default="")
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--sensor_csv", default="")
    parser.add_argument("--observed_time_steps", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--early_steps", type=int, default=20)
    parser.add_argument("--test_manifest", default="")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    train_rows = load_manifest(args.train_manifest)
    test_rows = load_manifest(args.test_manifest) if args.test_manifest else []

    sensor_xyz_real = None
    u_real = v_real = w_real = None
    core = None
    test_core = None
    if args.input_mode in {"sensor", "hybrid"}:
        if not args.field_h5 or not args.metadata_path or not args.sensor_csv:
            raise ValueError("--field_h5, --metadata_path, and --sensor_csv are required when --input_mode sensor/hybrid")
        _, _, _, _, u_real, v_real, w_real, _ = load_meta(args.metadata_path)
        sensor_xyz_real = parse_sensor_csv(args.sensor_csv)
        if sensor_xyz_real.shape[1] > 3:
            sensor_xyz_real = sensor_xyz_real[:, :3]
    if args.input_mode in {"core", "hybrid"}:
        if not args.core_path:
            raise ValueError("--core_path is required when --input_mode core/hybrid")
        core = np.asarray(sio.loadmat(args.core_path)["core"], dtype=np.float64)
        test_core_path = args.test_core_path or args.core_path
        test_core = np.asarray(sio.loadmat(test_core_path)["core"], dtype=np.float64)

    train_field_h5 = args.field_h5
    test_field_h5 = args.test_field_h5 or args.field_h5

    def build_xy(rows, dataset_kind):
        feats = []
        targets = []
        csv_rows = []
        core_array = core if dataset_kind == "train" else test_core
        field_h5 = train_field_h5 if dataset_kind == "train" else test_field_h5
        dataset_length = core_array.shape[0] if core_array is not None else None
        if dataset_length is None and field_h5:
            with h5py.File(field_h5, "r") as f:
                dataset_length = int(f["data"].shape[0])
        index_col = choose_index_column(rows, dataset_length)
        for row in rows:
            idx = int(row[index_col])
            if args.input_mode == "core":
                feat = build_core_features(core_array[idx], early_steps=args.early_steps)
            elif args.input_mode == "sensor":
                field = load_case_field(field_h5, idx)
                obs = extract_sensor_observation(
                    field,
                    sensor_xyz_real,
                    u_real,
                    v_real,
                    w_real,
                    observed_time_steps=args.observed_time_steps,
                )
                feat = build_sensor_features(obs, sensor_xyz=sensor_xyz_real, early_steps=args.early_steps)
            else:
                field = load_case_field(field_h5, idx)
                obs = extract_sensor_observation(
                    field,
                    sensor_xyz_real,
                    u_real,
                    v_real,
                    w_real,
                    observed_time_steps=args.observed_time_steps,
                )
                core_feat = build_core_features(core_array[idx], early_steps=args.early_steps)
                sensor_feat = build_sensor_features(obs, sensor_xyz=sensor_xyz_real, early_steps=args.early_steps)
                feat = np.concatenate([core_feat, sensor_feat], axis=0)
            target = target_from_manifest_row(row)
            feats.append(feat)
            targets.append(target)
            csv_rows.append(dict(row))
        return np.asarray(feats, dtype=np.float64), np.asarray(targets, dtype=np.float64), csv_rows

    x_train, y_train, train_csv_rows = build_xy(train_rows, "train")
    model = fit_ridge_multioutput(x_train, y_train, alpha=args.alpha)
    if args.input_mode == "core":
        model["feature_type"] = "core_stats_v2"
    elif args.input_mode == "sensor":
        model["feature_type"] = "sensor_stats_v1"
    else:
        model["feature_type"] = "hybrid_stats_v1"
    model["early_steps"] = int(args.early_steps)
    model["input_mode"] = args.input_mode
    model["observed_time_steps"] = int(args.observed_time_steps)

    train_pred = predict_ridge_multioutput(x_train, model)
    train_metrics = regression_metrics(y_train, train_pred)

    payload = dict(model)
    payload["train_metrics"] = train_metrics

    if test_rows:
        x_test, y_test, _ = build_xy(test_rows, "test")
        test_pred = predict_ridge_multioutput(x_test, model)
        payload["test_metrics"] = regression_metrics(y_test, test_pred)
        payload["test_size"] = int(len(test_rows))
    payload["train_size"] = int(len(train_rows))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    out_rows = []
    for row, pred, target in zip(train_csv_rows, train_pred, y_train):
        row["pred_source_u_m"] = float(pred[0])
        row["pred_source_v_m"] = float(pred[1])
        row["pred_source_w_m"] = float(pred[2])
        row["pred_leak_rate_ml_min"] = float(pred[3])
        row["target_source_u_m"] = float(target[0])
        row["target_source_v_m"] = float(target[1])
        row["target_source_w_m"] = float(target[2])
        row["target_leak_rate_ml_min"] = float(target[3])
        row["source_l2_error_mm"] = float(np.linalg.norm(pred[:3] - target[:3]) * 1000.0)
        row["leak_rate_abs_error_ml_min"] = float(abs(pred[3] - target[3]))
        row["leak_rate_rel_error"] = float(abs(pred[3] - target[3]) / max(abs(target[3]), 1e-12))
        row["input_mode"] = args.input_mode
        out_rows.append(row)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
