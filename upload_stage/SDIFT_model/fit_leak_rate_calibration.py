import argparse
import csv
import json
import os

import h5py
import numpy as np

from source_param_utils import (
    build_leak_rate_feature_dict,
    estimate_leak_source,
)


DEFAULT_FEATURES = ["mass_slope_early", "source_strength"]


def load_meta(meta_path):
    meta = np.load(meta_path, allow_pickle=True).item()["data"]
    u = np.asarray(meta.get("u_ind_real", meta["u_ind_uni"]), dtype=np.float64)
    v = np.asarray(meta.get("v_ind_real", meta["v_ind_uni"]), dtype=np.float64)
    w = np.asarray(meta.get("w_ind_real", meta["w_ind_uni"]), dtype=np.float64)
    return u, v, w


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    for row in rows:
        row["data_index"] = int(row["data_index"])
        row["source_x_mm"] = int(float(row["source_x_mm"]))
        row["source_y_mm"] = int(float(row["source_y_mm"]))
        row["source_z_mm"] = int(float(row["source_z_mm"]))
        row["leak_rate_ml_min"] = float(row["leak_rate_ml_min"])
    rows.sort(key=lambda r: r["data_index"])
    return rows


def fit_linear_regression(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    design = np.concatenate([ones, x], axis=1)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(beta[0])
    coef = beta[1:].astype(np.float64)
    pred = intercept + x @ coef
    return intercept, coef, pred


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    rel = np.abs(err) / np.maximum(np.abs(y_true), 1e-12)
    y_mean = float(np.mean(y_true))
    denom = float(np.sum((y_true - y_mean) ** 2))
    r2 = None if denom <= 0 else float(1.0 - np.sum(err ** 2) / denom)
    return {
        "rmse": rmse,
        "mae": mae,
        "mean_rel_error": float(np.mean(rel)),
        "max_rel_error": float(np.max(rel)),
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit a leak-rate calibration model from CFD fields and known leak rates.")
    parser.add_argument("--field_h5", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--feature_names", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--time_window", type=int, default=5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--source_abs_threshold", type=float, default=1e-6)
    parser.add_argument("--source_rel_threshold", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows = load_manifest(args.manifest_csv)
    u_axis, v_axis, w_axis = load_meta(args.meta)

    output_rows = []
    x_rows = []
    y_rows = []
    with h5py.File(args.field_h5, "r") as f:
        data = f["data"]
        for row in rows:
            field = np.asarray(data[row["data_index"]], dtype=np.float64)
            source = estimate_leak_source(
                field,
                u_axis,
                v_axis,
                w_axis,
                time_window=args.time_window,
                radius=args.radius,
                abs_threshold=args.source_abs_threshold,
                rel_threshold=args.source_rel_threshold,
            )
            features = build_leak_rate_feature_dict(source)
            pred_pos = np.asarray(source["position"], dtype=np.float64)
            # The tensor axes are normalized as (u, v, w) = (x, z, y).
            gt_pos_m = np.asarray(
                [
                    row["source_x_mm"] / 1000.0,
                    row["source_z_mm"] / 1000.0,
                    row["source_y_mm"] / 1000.0,
                ],
                dtype=np.float64,
            )
            source_l2_m = float(np.linalg.norm(pred_pos - gt_pos_m))
            feature_vector = [float(features.get(name, 0.0)) for name in args.feature_names]
            x_rows.append(feature_vector)
            y_rows.append(float(row["leak_rate_ml_min"]))

            item = dict(row)
            item["pred_source_x_m"] = float(pred_pos[0])
            item["pred_source_y_m"] = float(pred_pos[1])
            item["pred_source_z_m"] = float(pred_pos[2])
            item["source_l2_error_m"] = source_l2_m
            item["source_l2_error_mm"] = source_l2_m * 1000.0
            for key, value in features.items():
                item[f"feature_{key}"] = value
            output_rows.append(item)

    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    intercept, coef, pred = fit_linear_regression(x, y)
    metrics = regression_metrics(y, pred)
    source_err_mm = [float(r["source_l2_error_mm"]) for r in output_rows]

    for row, q_pred in zip(output_rows, pred):
        row["pred_leak_rate_ml_min"] = float(q_pred)
        row["leak_rate_abs_error_ml_min"] = float(abs(q_pred - float(row["leak_rate_ml_min"])))
        row["leak_rate_rel_error"] = float(abs(q_pred - float(row["leak_rate_ml_min"])) / max(float(row["leak_rate_ml_min"]), 1e-12))

    payload = {
        "type": "linear_features",
        "feature_names": list(args.feature_names),
        "coef": coef.astype(float).tolist(),
        "intercept": float(intercept),
        "output_unit": "mL/min",
        "fit_metrics": metrics,
        "source_position_metrics": {
            "mean_l2_error_mm": float(np.mean(source_err_mm)),
            "median_l2_error_mm": float(np.median(source_err_mm)),
            "max_l2_error_mm": float(np.max(source_err_mm)),
        },
        "estimation_config": {
            "time_window": int(args.time_window),
            "radius": int(args.radius),
            "source_abs_threshold": float(args.source_abs_threshold),
            "source_rel_threshold": float(args.source_rel_threshold),
        },
        "note": "Calibration was fit on available CFD fields. Use this as a baseline calibration model for reconstructed-field leak-rate estimation, and validate on held-out conditions before claiming final performance.",
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
