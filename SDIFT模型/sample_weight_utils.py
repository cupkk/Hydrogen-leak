import csv
from typing import Iterable, List, Sequence

import numpy as np


def load_manifest_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def leak_rate_from_row(row: dict) -> float:
    return float(row["leak_rate_ml_min"])


def _normalize_mean_one(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    mean = float(np.mean(weights))
    if mean <= 0:
        return np.ones_like(weights, dtype=np.float64)
    return weights / mean


def compute_sample_weights_from_rates(
    leak_rates: Sequence[float],
    mode: str = "none",
) -> np.ndarray:
    leak_rates = np.asarray(leak_rates, dtype=np.float64)
    if leak_rates.ndim != 1:
        raise ValueError("leak_rates must be a 1D sequence")
    if mode in {"", "none", None}:
        return np.ones_like(leak_rates, dtype=np.float64)

    if mode == "balanced_by_rate":
        unique, counts = np.unique(leak_rates, return_counts=True)
        lookup = {float(k): 1.0 / float(v) for k, v in zip(unique, counts)}
        weights = np.asarray([lookup[float(x)] for x in leak_rates], dtype=np.float64)
        return _normalize_mean_one(weights)

    if mode == "lowflow_focus_v1":
        weights = np.ones_like(leak_rates, dtype=np.float64)
        weights[leak_rates <= 50.0] = 4.0
        weights[(leak_rates > 50.0) & (leak_rates <= 100.0)] = 3.0
        weights[(leak_rates > 100.0) & (leak_rates <= 200.0)] = 2.0
        weights[(leak_rates > 200.0) & (leak_rates <= 400.0)] = 1.5
        return _normalize_mean_one(weights)

    if mode == "lowflow_balanced_v1":
        balanced = compute_sample_weights_from_rates(leak_rates, mode="balanced_by_rate")
        focus = compute_sample_weights_from_rates(leak_rates, mode="lowflow_focus_v1")
        return _normalize_mean_one(balanced * focus)

    raise ValueError(f"unsupported sample_weight_mode: {mode}")


def compute_sample_weights_from_manifest(rows: Iterable[dict], mode: str = "none") -> np.ndarray:
    rates = [leak_rate_from_row(row) for row in rows]
    return compute_sample_weights_from_rates(rates, mode=mode)

