"""Prediction intervals from out-of-sample backtest residuals (split-conformal style).

For a series and model, the residuals actual − forecast from earlier backtest folds give empirical quantiles.
The interval for a new forecast f is [f + q_lo, f + q_hi], with q_lo/q_hi the (1 − level)/2 and (1 + level)/2
residual quantiles, widened with the split-conformal finite-sample correction (rank ⌈(n + 1)·p⌉ out of n), and
the lower bound clipped at 0. Coverage is measured on folds that were not used for
calibration, so it is an honest estimate rather than an in-sample fit.
"""
from __future__ import annotations

import numpy as np


def residual_offsets(residuals: np.ndarray, levels) -> dict[float, tuple[float, float]]:
    """Empirical quantile offsets per level from a 1-D array of residuals (NaNs ignored)."""
    r = np.asarray(residuals, float)
    r = r[~np.isnan(r)]
    out = {}
    for lv in levels:
        if len(r) == 0:
            out[float(lv)] = (0.0, 0.0)
            continue
        r_sorted = np.sort(r)
        n = len(r_sorted)
        k_hi = min(n, int(np.ceil((n + 1) * (1 + lv) / 2 - 1e-9)))     # 1-based ranks (tolerant of float error)
        k_lo = max(1, int(np.floor((n + 1) * (1 - lv) / 2 + 1e-9)))
        out[float(lv)] = (float(r_sorted[k_lo - 1]), float(r_sorted[k_hi - 1]))
    return out


def apply_offsets(point: np.ndarray, offsets: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.clip(point + offsets[0], 0, None)
    hi = np.maximum(point + offsets[1], lo)
    return lo, hi
