import argparse
import csv
import json
import os

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
from train_source_param_regressor import (
    choose_index_column,
    extract_sensor_observation,
    load_case_field,
    load_manifest,
)
from make_sensor_observations import load_meta, parse_sensor_csv
from sample_weight_utils import compute_sample_weights_from_manifest


def parse_alpha_grid(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("alpha grid cannot be empty")
    return vals


def load_rows_with_groups(path):
    rows = load_manifest(path)
    groups = []
    for row in rows:
        groups.append(row.get("position_key") or f'{row["source_x_mm"]},{row["source_y_mm"]},{row["source_z_mm"]}')
    return rows, np.asarray(groups)


def build_features(mode, rows, core_array, field_h5, sensor_xyz_real, u_real, v_real, w_real, early_steps, observed_time_steps):
    feats = []
    targets = []
    case_ids = []
    dataset_length = core_array.shape[0] if core_array is not None else None
    if dataset_length is None and field_h5:
        import h5py

        with h5py.File(field_h5, "r") as f:
            dataset_length = int(f["data"].shape[0])
    index_col = choose_index_column(rows, dataset_length)
    for row in rows:
        idx = int(row[index_col])
        if mode == "core":
            feat = build_core_features(core_array[idx], early_steps=early_steps)
        elif mode == "sensor":
            field = load_case_field(field_h5, idx)
            obs = extract_sensor_observation(
                field,
                sensor_xyz_real,
                u_real,
                v_real,
                w_real,
                observed_time_steps=observed_time_steps,
            )
            feat = build_sensor_features(obs, sensor_xyz=sensor_xyz_real, early_steps=early_steps)
        elif mode == "hybrid":
            field = load_case_field(field_h5, idx)
            obs = extract_sensor_observation(
                field,
                sensor_xyz_real,
                u_real,
                v_real,
                w_real,
                observed_time_steps=observed_time_steps,
            )
            core_feat = build_core_features(core_array[idx], early_steps=early_steps)
            sensor_feat = build_sensor_features(obs, sensor_xyz=sensor_xyz_real, early_steps=early_steps)
            feat = np.concatenate([core_feat, sensor_feat], axis=0)
        else:
            raise ValueError(f"unsupported mode: {mode}")
        feats.append(feat)
        targets.append(target_from_manifest_row(row))
        case_ids.append(row["case_id"])
    return np.asarray(feats, dtype=np.float64), np.asarray(targets, dtype=np.float64), case_ids


def leave_one_group_out_indices(groups):
    unique_groups = []
    for g in groups:
        if g not in unique_groups:
            unique_groups.append(g)
    for group in unique_groups:
        val_mask = groups == group
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        yield group, train_idx, val_idx


def metric_score(metrics):
    return float(metrics["source_l2_error_mm_mean"] / 100.0 + metrics["leak_rate_rel_error_mean"])


def cross_validate_alpha(x_train, y_train, groups, alpha_grid, sample_weight=None):
    rows = []
    for alpha in alpha_grid:
        fold_metrics = []
        for group, train_idx, val_idx in leave_one_group_out_indices(groups):
            train_weight = None if sample_weight is None else sample_weight[train_idx]
            model = fit_ridge_multioutput(x_train[train_idx], y_train[train_idx], alpha=alpha, sample_weight=train_weight)
            pred = predict_ridge_multioutput(x_train[val_idx], model)
            metrics = regression_metrics(y_train[val_idx], pred)
            fold_metrics.append({
                "group": group,
                "alpha": float(alpha),
                "score": metric_score(metrics),
                **metrics,
            })
        mean_score = float(np.mean([x["score"] for x in fold_metrics]))
        rows.append({
            "alpha": float(alpha),
            "mean_score": mean_score,
            "mean_source_l2_error_mm": float(np.mean([x["source_l2_error_mm_mean"] for x in fold_metrics])),
            "mean_leak_rate_rel_error": float(np.mean([x["leak_rate_rel_error_mean"] for x in fold_metrics])),
            "folds": fold_metrics,
        })
    rows.sort(key=lambda x: x["mean_score"])
    return rows


def save_case_predictions(path, rows, preds, targets, mode):
    out_rows = []
    for row, pred, target in zip(rows, preds, targets):
        item = dict(row)
        item["mode"] = mode
        item["pred_source_u_m"] = float(pred[0])
        item["pred_source_v_m"] = float(pred[1])
        item["pred_source_w_m"] = float(pred[2])
        item["pred_leak_rate_ml_min"] = float(pred[3])
        item["target_source_u_m"] = float(target[0])
        item["target_source_v_m"] = float(target[1])
        item["target_source_w_m"] = float(target[2])
        item["target_leak_rate_ml_min"] = float(target[3])
        item["source_l2_error_mm"] = float(np.linalg.norm(pred[:3] - target[:3]) * 1000.0)
        item["leak_rate_abs_error_ml_min"] = float(abs(pred[3] - target[3]))
        item["leak_rate_rel_error"] = float(abs(pred[3] - target[3]) / max(abs(target[3]), 1e-12))
        out_rows.append(item)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def main():
    parser = argparse.ArgumentParser(description="Train and compare source/leak parameter regressors for core, sensor, and hybrid inputs.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--train_field_h5", required=True)
    parser.add_argument("--test_field_h5", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--sensor_csv", required=True)
    parser.add_argument("--train_core_path", default="")
    parser.add_argument("--test_core_path", default="")
    parser.add_argument("--modes", default="sensor,core,hybrid")
    parser.add_argument("--alpha_grid", default="0.01,0.1,1,10,100")
    parser.add_argument("--early_steps", type=int, default=20)
    parser.add_argument("--observed_time_steps", type=int, default=120)
    parser.add_argument(
        "--sample_weight_mode",
        type=str,
        default="none",
        choices=["none", "balanced_by_rate", "lowflow_focus_v1", "lowflow_balanced_v1"],
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_rows, train_groups = load_rows_with_groups(args.train_manifest)
    test_rows, _ = load_rows_with_groups(args.test_manifest)
    _, _, _, _, u_real, v_real, w_real, _ = load_meta(args.metadata_path)
    sensor_xyz_real = parse_sensor_csv(args.sensor_csv)
    if sensor_xyz_real.shape[1] > 3:
        sensor_xyz_real = sensor_xyz_real[:, :3]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    alpha_grid = parse_alpha_grid(args.alpha_grid)

    train_core = None
    test_core = None
    if any(m in {"core", "hybrid"} for m in modes):
        if not args.train_core_path or not args.test_core_path:
            raise ValueError("train_core_path and test_core_path are required for core/hybrid modes")
        train_core = np.asarray(sio.loadmat(args.train_core_path)["core"], dtype=np.float64)
        test_core = np.asarray(sio.loadmat(args.test_core_path)["core"], dtype=np.float64)

    ranking = []
    for mode in modes:
        x_train, y_train, _ = build_features(
            mode,
            train_rows,
            train_core,
            args.train_field_h5,
            sensor_xyz_real,
            u_real,
            v_real,
            w_real,
            args.early_steps,
            args.observed_time_steps,
        )
        x_test, y_test, _ = build_features(
            mode,
            test_rows,
            test_core,
            args.test_field_h5,
            sensor_xyz_real,
            u_real,
            v_real,
            w_real,
            args.early_steps,
            args.observed_time_steps,
        )

        sample_weight = None
        if args.sample_weight_mode != "none":
            sample_weight = compute_sample_weights_from_manifest(train_rows, mode=args.sample_weight_mode)
        cv_rows = cross_validate_alpha(x_train, y_train, train_groups, alpha_grid, sample_weight=sample_weight)
        best_alpha = cv_rows[0]["alpha"]
        model = fit_ridge_multioutput(x_train, y_train, alpha=best_alpha, sample_weight=sample_weight)
        model["feature_type"] = f"{mode}_stats_formal_v1"
        model["input_mode"] = mode
        model["early_steps"] = int(args.early_steps)
        model["observed_time_steps"] = int(args.observed_time_steps)
        model["selected_alpha"] = float(best_alpha)
        model["alpha_grid"] = alpha_grid
        model["cv_summary"] = cv_rows
        model["sample_weight_mode"] = args.sample_weight_mode

        train_pred = predict_ridge_multioutput(x_train, model)
        test_pred = predict_ridge_multioutput(x_test, model)
        train_metrics = regression_metrics(y_train, train_pred)
        test_metrics = regression_metrics(y_test, test_pred)
        model["train_metrics"] = train_metrics
        model["test_metrics"] = test_metrics
        model["train_size"] = int(len(train_rows))
        model["test_size"] = int(len(test_rows))

        model_path = os.path.join(args.out_dir, f"{mode}_param_model.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
        save_case_predictions(os.path.join(args.out_dir, f"{mode}_test_predictions.csv"), test_rows, test_pred, y_test, mode)

        ranking.append({
            "mode": mode,
            "selected_alpha": float(best_alpha),
            "feature_type": model["feature_type"],
            "sample_weight_mode": args.sample_weight_mode,
            "test_score": metric_score(test_metrics),
            "source_l2_error_mm_mean": float(test_metrics["source_l2_error_mm_mean"]),
            "leak_rate_rel_error_mean": float(test_metrics["leak_rate_rel_error_mean"]),
            "leak_rate_mae_ml_min": float(test_metrics["leak_rate_mae_ml_min"]),
            "model_path": model_path,
        })

    ranking.sort(key=lambda x: x["test_score"])
    with open(os.path.join(args.out_dir, "ranking.json"), "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "ranking.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        writer.writerows(ranking)

    print(f"saved: {os.path.join(args.out_dir, 'ranking.json')}")
    print(f"saved: {os.path.join(args.out_dir, 'ranking.csv')}")


if __name__ == "__main__":
    main()
