"""Load the raw files into a wide panel (series × days), fill gaps and replace outliers.

Every cleaning step is causal: the cleaned value for day t uses only raw values up to day t. That keeps the
backtest honest, because cleaning the full history cannot pull information from after a cutoff.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, events
from .generate import file_sha256, series_id


@dataclass
class Panel:
    series: list[str]
    stores: np.ndarray
    categories: np.ndarray
    dates: pd.DatetimeIndex          # observed days
    y_raw: np.ndarray                # (n_series, n_days), NaN where the row was missing
    y: np.ndarray                    # cleaned
    imputed: np.ndarray              # bool mask: gap filled
    replaced: np.ndarray             # bool mask: outlier replaced
    plan_dates: pd.DatetimeIndex     # observed days + the planned future
    price_index: np.ndarray          # (n_series, n_plan_days) price / list price
    promo: np.ndarray                # (n_series, n_plan_days)
    calendar: pd.DataFrame           # one row per plan date
    data_sha256: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def index(self, sid: str) -> int:
        return self.series.index(sid)


def _same_weekday_median(x: np.ndarray, weeks: int) -> np.ndarray:
    """For each day, the median of the same weekday over the previous `weeks` weeks (NaNs ignored)."""
    n, T = x.shape
    stack = np.full((weeks, n, T), np.nan)
    for w in range(1, weeks + 1):
        stack[w - 1, :, 7 * w:] = x[:, :T - 7 * w]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN slices → NaN, handled by callers
        return np.nanmedian(stack, axis=0)


def fill_gaps(y_raw: np.ndarray, weeks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Missing day → same-weekday median of the previous weeks, else the last observed value, else 0."""
    est = _same_weekday_median(y_raw, weeks)
    last = pd.DataFrame(y_raw.T).ffill().shift(1).to_numpy().T
    est = np.where(np.isnan(est), last, est)
    est = np.where(np.isnan(est), 0.0, est)
    mask = np.isnan(y_raw)
    return np.where(mask, est, y_raw), mask


def replace_outliers(x: np.ndarray, exempt: np.ndarray, window: int = 28, k: float = 6.0,
                     weeks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Trailing median/IQR check: flag x > median + k·sigma of the previous `window` days.

    sigma = max(IQR / 1.349, sqrt(median) + 1), so low-count series are not flagged for a sale of 3.
    Flagged values become the same-weekday median of the previous weeks (capped at the threshold).
    Promotion days and calendar events are exempt: their spikes are real demand.
    """
    df = pd.DataFrame(x.T)
    roll = df.rolling(window, min_periods=14)
    med = roll.median().shift(1).to_numpy().T
    iqr = (roll.quantile(0.75) - roll.quantile(0.25)).shift(1).to_numpy().T
    sigma = np.maximum(iqr / 1.349, np.sqrt(np.maximum(med, 0)) + 1)
    thr = med + k * sigma
    flag = (x > thr) & ~np.isnan(thr) & ~exempt
    est = _same_weekday_median(x, weeks)
    repl = np.minimum(np.where(np.isnan(est), med, est), thr)
    return np.where(flag, repl, x), flag


def load_panel(cfg: dict, root: Path | None = None, write: bool = False) -> Panel:
    root = Path(root or config.ROOT)
    raw = root / "data" / "raw"
    sales = pd.read_csv(raw / "sales.csv", parse_dates=["date"])
    plan = pd.read_csv(raw / "plan.csv", parse_dates=["date"])
    for df in (sales, plan):
        df["series"] = df["store"] + "-" + df["category"]

    end = pd.Timestamp(cfg["data"]["end"])
    dates = pd.date_range(sales["date"].min(), end, freq="D")
    keys = plan[["store", "category"]].drop_duplicates().sort_values(["store", "category"])
    series = [series_id(s, c) for s, c in keys.itertuples(index=False)]
    y_raw = sales.pivot_table(index="series", columns="date", values="units", aggfunc="sum")
    y_raw = y_raw.reindex(index=series, columns=dates).to_numpy(float)

    plan_dates = pd.date_range(dates[0], plan["date"].max(), freq="D")
    price = plan.pivot(index="series", columns="date", values="price").reindex(index=series, columns=plan_dates)
    promo = plan.pivot(index="series", columns="date", values="promo").reindex(index=series, columns=plan_dates)
    list_price = np.array([cfg["categories"][c]["list_price"] for c in keys["category"]], float)[:, None]
    cal = events.calendar_frame(plan_dates, cfg["calendar"])

    p = cfg["prep"]
    filled, imputed = fill_gaps(y_raw, p["gap_fill_weeks"])
    T = len(dates)
    exempt = (promo.to_numpy()[:, :T] == 1) | (cal["event"].to_numpy()[None, :T] == 1)
    y, replaced = replace_outliers(filled, exempt, p["outlier_window"], p["outlier_k"], p["gap_fill_weeks"])

    panel = Panel(series=series, stores=keys["store"].to_numpy(), categories=keys["category"].to_numpy(),
                  dates=dates, y_raw=y_raw, y=y, imputed=imputed, replaced=replaced, plan_dates=plan_dates,
                  price_index=price.to_numpy(float) / list_price, promo=promo.to_numpy(int), calendar=cal,
                  data_sha256=file_sha256(raw / "sales.csv"))
    panel.meta = {"n_series": len(series), "n_days": T, "start": str(dates[0].date()), "end": str(end.date()),
                  "cells_imputed": int(imputed.sum()), "cells_replaced": int(replaced.sum()),
                  "data_sha256": panel.data_sha256}
    if write:
        out = root / "data" / "prepared"
        out.mkdir(parents=True, exist_ok=True)
        long = pd.DataFrame({"series": np.repeat(series, T), "date": np.tile(dates.strftime("%Y-%m-%d"), len(series)),
                             "units_raw": y_raw.ravel(), "units_clean": y.ravel().round(3),
                             "imputed": imputed.ravel().astype(int), "replaced": replaced.ravel().astype(int)})
        long.to_csv(out / "sales_clean.csv", index=False)
        (out / "prep_report.json").write_text(json.dumps(panel.meta, indent=2))
    return panel
