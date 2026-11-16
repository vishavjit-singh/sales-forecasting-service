"""Forecast accuracy metrics. All take 1-D arrays of actuals and forecasts over the test window."""
from __future__ import annotations

import numpy as np


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2)))


def wape(y, f) -> float:
    """Sum of absolute errors / sum of actuals. Volume-weighted, so it is defined when single days are zero."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - f).sum() / denom) if denom > 0 else float("nan")


def smape(y, f) -> float:
    """Symmetric MAPE in [0, 2]; a day where both actual and forecast are 0 counts as a perfect 0."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y) + np.abs(f)
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(denom > 0, 2 * np.abs(y - f) / denom, 0.0)
    return float(terms.mean())


def mase_scale(history, season: int = 7) -> float:
    """In-sample MAE of the seasonal naive forecast on the training data (the MASE denominator)."""
    h = np.asarray(history, float)
    if len(h) <= season:
        return float("nan")
    s = np.mean(np.abs(h[season:] - h[:-season]))
    return float(s) if s > 0 else float("nan")


def mase(y, f, scale: float) -> float:
    return mae(y, f) / scale if scale and np.isfinite(scale) else float("nan")


def bias(y, f) -> float:
    """(sum of forecasts − sum of actuals) / sum of actuals: positive means over-forecasting."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y).sum()
    return float((f - y).sum() / denom) if denom > 0 else float("nan")


def all_metrics(y, f, scale: float) -> dict:
    return {"mae": mae(y, f), "rmse": rmse(y, f), "wape": wape(y, f), "smape": smape(y, f),
            "mase": mase(y, f, scale), "bias": bias(y, f)}


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
