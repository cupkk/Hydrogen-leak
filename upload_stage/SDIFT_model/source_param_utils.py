import json

import numpy as np


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


def voxel_weights_from_axes(u_axis, v_axis, w_axis):
    wu = point_weights_1d(u_axis)
    wv = point_weights_1d(v_axis)
    ww = point_weights_1d(w_axis)
    return wu[:, None, None] * wv[None, :, None] * ww[None, None, :]


def compute_mass_series(field, voxel_weight):
    return np.sum(field * voxel_weight[None, ...], axis=(1, 2, 3))


def fit_line_slope(y):
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


def _interp_axis(axis, index_value):
    axis = np.asarray(axis, dtype=np.float64)
    grid = np.arange(axis.size, dtype=np.float64)
    return float(np.interp(float(index_value), grid, axis))


def estimate_leak_source(
    field,
    u_axis,
    v_axis,
    w_axis,
    time_window=5,
    radius=2,
    abs_threshold=1e-6,
    rel_threshold=0.1,
):
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 4:
        raise ValueError("field must be [T, U, V, W]")

    t_window = int(max(1, min(int(time_window), field.shape[0])))
    radius = int(max(1, radius))
    early_field = np.maximum(field[:t_window], 0.0)
    early_mean = early_field.mean(axis=0)
    early_peak = early_field.max(axis=0)

    peak_max = float(early_peak.max())
    threshold = max(float(abs_threshold), float(rel_threshold) * peak_max)
    active = early_field >= threshold
    any_active = active.any(axis=0)
    first_idx = np.argmax(active, axis=0).astype(np.float64)
    first_idx[~any_active] = float(t_window)

    if t_window > 1:
        arrival_weight = 1.0 - np.clip(first_idx, 0.0, t_window - 1) / float(t_window - 1)
    else:
        arrival_weight = np.ones_like(first_idx, dtype=np.float64)
    arrival_weight[~any_active] = 0.0

    if peak_max > 0:
        peak_norm = early_peak / peak_max
    else:
        peak_norm = np.zeros_like(early_peak)

    score = early_mean * (0.5 + 0.5 * peak_norm) * np.maximum(arrival_weight, 0.0)
    if not np.any(score > 0):
        score = early_mean.copy()
    if not np.any(score > 0):
        score = early_peak.copy()

    coarse_idx = np.unravel_index(np.argmax(score), score.shape)
    u0 = max(0, coarse_idx[0] - radius)
    u1 = min(score.shape[0], coarse_idx[0] + radius + 1)
    v0 = max(0, coarse_idx[1] - radius)
    v1 = min(score.shape[1], coarse_idx[1] + radius + 1)
    w0 = max(0, coarse_idx[2] - radius)
    w1 = min(score.shape[2], coarse_idx[2] + radius + 1)

    patch_score = score[u0:u1, v0:v1, w0:w1]
    weights = np.maximum(patch_score, 0.0)
    if weights.sum() <= 0:
        weights = np.maximum(early_peak[u0:u1, v0:v1, w0:w1], 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(patch_score, dtype=np.float64)

    grid_u, grid_v, grid_w = np.meshgrid(
        np.arange(u0, u1, dtype=np.float64),
        np.arange(v0, v1, dtype=np.float64),
        np.arange(w0, w1, dtype=np.float64),
        indexing="ij",
    )
    denom = float(weights.sum())
    centroid_u = float(np.sum(weights * grid_u) / denom)
    centroid_v = float(np.sum(weights * grid_v) / denom)
    centroid_w = float(np.sum(weights * grid_w) / denom)

    voxel_weight = voxel_weights_from_axes(u_axis, v_axis, w_axis)
    mass_series = compute_mass_series(np.maximum(field, 0.0), voxel_weight)
    early_mass = mass_series[:t_window]
    early_mass_slope = fit_line_slope(early_mass)
    early_mass_mean = float(np.mean(early_mass)) if early_mass.size > 0 else 0.0

    arrival_idx_local = first_idx[coarse_idx]
    arrival_time_fraction = None
    if np.isfinite(arrival_idx_local) and t_window > 1:
        arrival_time_fraction = float(arrival_idx_local / float(t_window - 1))

    return {
        "method": "early_arrival_weighted_centroid_v1",
        "index": [int(coarse_idx[0]), int(coarse_idx[1]), int(coarse_idx[2])],
        "index_float": [centroid_u, centroid_v, centroid_w],
        "position": [
            _interp_axis(u_axis, centroid_u),
            _interp_axis(v_axis, centroid_v),
            _interp_axis(w_axis, centroid_w),
        ],
        "coarse_position": [
            float(u_axis[coarse_idx[0]]),
            float(v_axis[coarse_idx[1]]),
            float(w_axis[coarse_idx[2]]),
        ],
        "strength": float(np.mean(early_mean[u0:u1, v0:v1, w0:w1])),
        "peak": float(early_peak[coarse_idx]),
        "score_peak": float(score[coarse_idx]),
        "arrival_index": float(arrival_idx_local),
        "arrival_time_fraction": arrival_time_fraction,
        "mass_slope_early": float(early_mass_slope),
        "mass_mean_early": float(early_mass_mean),
        "mass_last": float(mass_series[-1]),
        "threshold": float(threshold),
        "time_window": int(t_window),
        "radius": int(radius),
    }


def build_leak_rate_feature_dict(source_summary):
    return {
        "source_strength": float(source_summary.get("strength", 0.0)),
        "strength": float(source_summary.get("strength", 0.0)),
        "peak": float(source_summary.get("peak", 0.0)),
        "score_peak": float(source_summary.get("score_peak", 0.0)),
        "mass_slope_early": float(source_summary.get("mass_slope_early", 0.0)),
        "mass_mean_early": float(source_summary.get("mass_mean_early", 0.0)),
        "mass_last": float(source_summary.get("mass_last", 0.0)),
        "arrival_time_fraction": float(source_summary.get("arrival_time_fraction", 1.0) if source_summary.get("arrival_time_fraction") is not None else 1.0),
    }


def load_leak_rate_calibration(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "type" not in cfg:
        raise ValueError("leak-rate calibration must contain 'type'")
    return cfg


def estimate_leak_rate(feature_dict, calibration):
    if calibration is None:
        return None
    ctype = calibration.get("type", "").lower()

    if ctype == "linear":
        input_key = calibration.get("input", "source_strength")
        x = float(feature_dict.get(input_key, 0.0))
        a = float(calibration.get("a", 1.0))
        b = float(calibration.get("b", 0.0))
        return a * x + b

    if ctype == "power":
        input_key = calibration.get("input", "source_strength")
        x = float(feature_dict.get(input_key, 0.0))
        a = float(calibration.get("a", 1.0))
        p = float(calibration.get("p", 1.0))
        b = float(calibration.get("b", 0.0))
        return a * (max(x, 0.0) ** p) + b

    if ctype == "linear_features":
        names = calibration.get("feature_names", [])
        coef = calibration.get("coef", [])
        intercept = float(calibration.get("intercept", 0.0))
        if len(names) != len(coef):
            raise ValueError("linear_features calibration requires matching feature_names and coef lengths")
        x = np.array([float(feature_dict.get(name, 0.0)) for name in names], dtype=np.float64)
        beta = np.array(coef, dtype=np.float64)
        return float(intercept + np.dot(x, beta))

    if ctype == "log_linear_features":
        names = calibration.get("feature_names", [])
        coef = calibration.get("coef", [])
        intercept = float(calibration.get("intercept", 0.0))
        eps = float(calibration.get("eps", 1e-12))
        if len(names) != len(coef):
            raise ValueError("log_linear_features calibration requires matching feature_names and coef lengths")
        x = np.array([np.log(max(float(feature_dict.get(name, 0.0)), eps)) for name in names], dtype=np.float64)
        beta = np.array(coef, dtype=np.float64)
        return float(np.exp(intercept + np.dot(x, beta)))

    raise ValueError(f"Unsupported leak-rate calibration type: {calibration.get('type')}")
