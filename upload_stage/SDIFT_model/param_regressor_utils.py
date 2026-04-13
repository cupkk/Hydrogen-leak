import json

import numpy as np


def target_from_manifest_row(row):
    return np.array(
        [
            float(row["source_x_mm"]) / 1000.0,
            float(row["source_z_mm"]) / 1000.0,
            float(row["source_y_mm"]) / 1000.0,
            float(row["leak_rate_ml_min"]),
        ],
        dtype=np.float64,
    )


def build_core_features(core, early_steps=20):
    core = np.asarray(core, dtype=np.float64)
    if core.ndim != 4:
        raise ValueError("core must be [T, R1, R2, R3]")
    t = core.shape[0]
    k = int(max(1, min(int(early_steps), t)))
    early = core[:k]
    feat_parts = [
        core.mean(axis=0).ravel(),
        core.max(axis=0).ravel(),
        core.std(axis=0).ravel(),
        early.mean(axis=0).ravel(),
    ]
    if k > 1:
        feat_parts.append(np.diff(early, axis=0).mean(axis=0).ravel())
    else:
        feat_parts.append(np.zeros_like(early[0].ravel()))
    feat_parts.extend(
        [
            core.mean(axis=(1, 2, 3)),
            core.max(axis=(1, 2, 3)),
            core.std(axis=(1, 2, 3)),
        ]
    )
    return np.concatenate(feat_parts, axis=0)


def _fit_line_slope(y):
    y = np.asarray(y, dtype=np.float64)
    if y.size <= 1:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def build_sensor_features(observations, sensor_xyz=None, early_steps=20):
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("observations must be [T, S]")
    t_count, s_count = observations.shape
    k = int(max(1, min(int(early_steps), t_count)))
    early = observations[:k]
    late = observations[-k:]

    per_sensor_mean = observations.mean(axis=0)
    per_sensor_max = observations.max(axis=0)
    per_sensor_std = observations.std(axis=0)
    per_sensor_slope = np.array([_fit_line_slope(observations[:, j]) for j in range(s_count)], dtype=np.float64)
    peak_idx = np.argmax(observations, axis=0).astype(np.float64)
    if t_count > 1:
        peak_idx /= float(t_count - 1)

    global_mean_series = observations.mean(axis=1)
    global_max_series = observations.max(axis=1)
    global_std_series = observations.std(axis=1)

    feature_parts = [
        observations.ravel(),
        per_sensor_mean,
        per_sensor_max,
        per_sensor_std,
        per_sensor_slope,
        peak_idx,
        early.mean(axis=0),
        late.mean(axis=0),
        global_mean_series,
        global_max_series,
        global_std_series,
        np.array([
            float(global_mean_series.mean()),
            float(global_mean_series.max()),
            float(global_mean_series.min()),
            float(_fit_line_slope(global_mean_series)),
            float(np.trapz(global_mean_series)),
        ], dtype=np.float64),
    ]

    if sensor_xyz is not None:
        sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64)
        if sensor_xyz.ndim != 2 or sensor_xyz.shape[0] != s_count:
            raise ValueError("sensor_xyz must match observations shape")
        feature_parts.extend(
            [
                sensor_xyz.ravel(),
                sensor_xyz.mean(axis=0),
                sensor_xyz.std(axis=0),
            ]
        )

    return np.concatenate(feature_parts, axis=0)


def standardize_train(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std, mean, std


def standardize_apply(x, mean, std):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return (x - mean) / std


def fit_ridge_multioutput(x, y, alpha=1.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_std, x_mean, x_scale = standardize_train(x)
    y_std, y_mean, y_scale = standardize_train(y)

    xtx = x_std.T @ x_std
    eye = np.eye(xtx.shape[0], dtype=np.float64)
    beta = np.linalg.solve(xtx + float(alpha) * eye, x_std.T @ y_std)

    return {
        "type": "ridge_core_features_v1",
        "alpha": float(alpha),
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "coef": beta.tolist(),
        "target_names": ["source_u_m", "source_v_m", "source_w_m", "leak_rate_ml_min"],
    }


def predict_ridge_multioutput(features, model):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x_std = standardize_apply(x, model["x_mean"], model["x_scale"])
    beta = np.asarray(model["coef"], dtype=np.float64)
    y_std = x_std @ beta
    y = y_std * np.asarray(model["y_scale"], dtype=np.float64) + np.asarray(model["y_mean"], dtype=np.float64)
    return y


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    out = {
        "per_target_mae": mae.astype(float).tolist(),
        "per_target_rmse": rmse.astype(float).tolist(),
        "target_names": ["source_u_m", "source_v_m", "source_w_m", "leak_rate_ml_min"],
    }
    source_l2 = np.linalg.norm(err[:, :3], axis=1)
    leak_rel = np.abs(err[:, 3]) / np.maximum(np.abs(y_true[:, 3]), 1e-12)
    out["source_l2_error_m_mean"] = float(np.mean(source_l2))
    out["source_l2_error_mm_mean"] = float(np.mean(source_l2) * 1000.0)
    out["source_l2_error_mm_max"] = float(np.max(source_l2) * 1000.0)
    out["leak_rate_mae_ml_min"] = float(mae[3])
    out["leak_rate_rmse_ml_min"] = float(rmse[3])
    out["leak_rate_rel_error_mean"] = float(np.mean(leak_rel))
    return out


def source_position_l2_mm(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape[-1] < 3 or y_pred.shape[-1] < 3:
        raise ValueError("source position arrays must have at least 3 columns")
    return np.linalg.norm(y_true[:, :3] - y_pred[:, :3], axis=1) * 1000.0


def load_model(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
