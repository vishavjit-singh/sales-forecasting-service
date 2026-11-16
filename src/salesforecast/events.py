"""The holiday and festival calendar from config.yaml, shared by the generator and the models.

These dates are known in advance, so models may use them for future days without leaking anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTSIDE = 99  # festival_days value for dates outside every festival window


def festival_days(dates: pd.DatetimeIndex, cal: dict) -> np.ndarray:
    """Days until the festival peak (negative after it) inside a festival window, else OUTSIDE."""
    f = cal["festival"]
    out = np.full(len(dates), OUTSIDE, dtype=int)
    d = dates.values.astype("datetime64[D]")
    for p in f["peaks"]:
        delta = (np.datetime64(p, "D") - d).astype(int)
        inside = (delta <= f["days_before"]) & (delta >= -f["days_after"])
        out[inside] = delta[inside]
    return out


def festival_ramp(dates: pd.DatetimeIndex, cal: dict) -> np.ndarray:
    """0 outside the window, rising linearly to 1 on the peak day, 0.4 on the days just after it."""
    dd = festival_days(dates, cal)
    before = cal["festival"]["days_before"]
    ramp = np.zeros(len(dates))
    pre = (dd >= 0) & (dd != OUTSIDE)
    ramp[pre] = 1.0 - dd[pre] / (before + 1)
    ramp[(dd < 0)] = 0.4
    return ramp


def is_holiday(dates: pd.DatetimeIndex, cal: dict) -> np.ndarray:
    md = dates.strftime("%m-%d")
    return np.isin(md, cal["holidays"]["dates"]).astype(int)


def is_year_end(dates: pd.DatetimeIndex, cal: dict) -> np.ndarray:
    md = dates.strftime("%m-%d")
    return ((md >= cal["year_end"]["start"]) & (md <= "12-31")).astype(int)


def calendar_frame(dates: pd.DatetimeIndex, cal: dict) -> pd.DataFrame:
    """One row per date: the calendar features every model may use."""
    doy = dates.dayofyear.values
    fd = festival_days(dates, cal)
    return pd.DataFrame({
        "dow": dates.dayofweek.values,
        "month": dates.month.values,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "festival_days": fd,
        "festival_ramp": festival_ramp(dates, cal),
        "holiday": is_holiday(dates, cal),
        "year_end": is_year_end(dates, cal),
        "event": ((fd != OUTSIDE) | (is_holiday(dates, cal) == 1) | (is_year_end(dates, cal) == 1)).astype(int),
    }, index=dates)
