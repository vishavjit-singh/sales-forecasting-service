"""SYNTHETIC daily unit sales for stores × categories, with every true component written to data/truth/.

expected units = base × store size × store trend × weekday × yearly cycle × festival × holiday × year-end
                 × (price / list price)^elasticity × promo display lift × structural break
observed units = Poisson(expected × Gamma(shape, 1/shape)), then outliers are injected and some rows are dropped.

Outputs
  data/raw/sales.csv        date, store, category, units           (rows missing on purpose)
  data/raw/plan.csv         date, store, category, price, promo    (complete, runs past the end of sales)
  data/truth/components.csv the expected value and each multiplier, per row (for checking findings)
  data/truth/generator.json counts of injected problems and the series that are intermittent by construction
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, events

WEEKDAY = np.array([0.93, 0.90, 0.92, 0.96, 1.04, 1.0, 1.0])  # Mon..Fri; Sat/Sun set per category


def series_id(store: str, category: str) -> str:
    return f"{store}-{category}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _weekday_profile(weekend: float) -> np.ndarray:
    w = WEEKDAY.copy()
    w[5], w[6] = weekend, weekend * 0.95
    return w / w.mean()


def generate(cfg: dict, root: Path | None = None) -> dict:
    root = Path(root or config.ROOT)
    d, cats, cal = cfg["data"], cfg["categories"], cfg["calendar"]
    rng = np.random.default_rng(d["seed"])
    end = pd.Timestamp(d["end"])
    dates = pd.date_range(d["start"], end + pd.Timedelta(days=d["plan_days_after_end"]), freq="D")
    n_obs = int((dates <= end).sum())
    stores = [f"S{i + 1:02d}" for i in range(d["n_stores"])]
    cat_names = list(cats)
    S, C, T = len(stores), len(cat_names), len(dates)
    t_years = (np.arange(T) / 365.25)[None, None, :]

    size = rng.lognormal(0.0, 0.35, S)
    growth = rng.uniform(-0.03, 0.12, S)
    trend = 1 + growth[:, None, None] * t_years

    dow, doy = dates.dayofweek.values, dates.dayofyear.values
    weekly = np.stack([_weekday_profile(cats[c]["weekend"])[dow] for c in cat_names])[None]
    yearly = np.stack([1 + cats[c]["yearly"] * np.cos(2 * np.pi * (doy - cats[c]["peak_doy"]) / 365.25)
                       for c in cat_names])[None]
    ramp = events.festival_ramp(dates, cal)
    festival = 1 + np.array([cats[c]["festival"] for c in cat_names])[None, :, None] * ramp[None, None, :]
    holiday = np.where(events.is_holiday(dates, cal) == 1, cal["holidays"]["factor"], 1.0)[None, None, :]
    year_end = np.where(events.is_year_end(dates, cal) == 1, cal["year_end"]["factor"], 1.0)[None, None, :]

    # prices: list price, one chain-wide increase, promotional discounts
    list_price = np.array([cats[c]["list_price"] for c in cat_names], float)[None, :, None]
    pi = d["price_increase"]
    step = np.ones((1, C, T))
    step[0, cat_names.index(pi["category"]), dates >= pd.Timestamp(pi["date"])] = pi["factor"]
    discount = np.zeros((S, C, T))
    n_promos = int(round(d["promotions_per_year"] * T / 365.25))
    for s in range(S):
        for c in range(C):
            for start in rng.choice(T - 7, n_promos, replace=False):
                discount[s, c, start:start + rng.integers(3, 8)] = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30])
    promo = (discount > 0).astype(int)
    price = list_price * step * (1 - discount)
    elasticity = np.array([cats[c]["elasticity"] for c in cat_names])[None, :, None]
    price_mult = (price / list_price) ** elasticity
    display = np.where(promo == 1, d["promo_display_lift"], 1.0)

    br = d["structural_break"]
    brk = np.ones((S, 1, T))
    brk[stores.index(br["store"]), 0, dates >= pd.Timestamp(br["date"])] = br["factor"]

    base = np.array([cats[c]["base"] for c in cat_names])[None, :, None] * size[:, None, None]
    expected = base * trend * weekly * yearly * festival * holiday * year_end * price_mult * display * brk

    k = d["noise_shape"]
    units_true = rng.poisson(expected * rng.gamma(k, 1 / k, expected.shape)).astype(float)

    # data problems, only inside the observed period
    units = units_true.copy()
    obs = np.zeros((S, C, T), bool)
    obs[:, :, :n_obs] = True
    outlier = np.zeros((S, C, T), bool)
    cand = np.argwhere(obs & (units >= 1))
    pick = cand[rng.choice(len(cand), int(d["outlier_rate"] * obs.sum()), replace=False)]
    outlier[tuple(pick.T)] = True
    units[outlier] = np.round(units[outlier] * rng.uniform(4, 8, outlier.sum()))
    missing = np.zeros((S, C, T), bool)
    for s in range(S):
        missing[s, :, rng.choice(np.arange(30, n_obs), d["missing_store_days"], replace=False)] = True
    missing |= obs & (rng.random((S, C, T)) < d["missing_cell_rate"])
    outlier &= ~missing

    idx = pd.MultiIndex.from_product([stores, cat_names, dates], names=["store", "category", "date"])
    full = pd.DataFrame({
        "units": units.ravel(), "units_true": units_true.ravel(), "expected": expected.ravel().round(4),
        "price": price.ravel().round(2), "promo": promo.ravel(),
        "festival_mult": np.broadcast_to(festival, (S, C, T)).ravel().round(4),
        "price_mult": price_mult.ravel().round(4), "break_mult": np.broadcast_to(brk, (S, C, T)).ravel(),
        "is_outlier": outlier.ravel().astype(int), "is_missing": missing.ravel().astype(int),
        "observed_period": obs.ravel().astype(int),
    }, index=idx).reset_index()
    full["date"] = full["date"].dt.strftime("%Y-%m-%d")

    raw, truth = root / "data" / "raw", root / "data" / "truth"
    raw.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    sales = full[(full["observed_period"] == 1) & (full["is_missing"] == 0)]
    sales[["date", "store", "category", "units"]].astype({"units": int}).to_csv(raw / "sales.csv", index=False)
    full[["date", "store", "category", "price", "promo"]].to_csv(raw / "plan.csv", index=False)
    in_obs = full[full["observed_period"] == 1]
    in_obs.drop(columns="observed_period").to_csv(truth / "components.csv", index=False)

    mean_true = in_obs.groupby(["store", "category"])["units_true"].agg(lambda u: (u == 0).mean())
    intermittent = sorted(series_id(s, c) for (s, c), z in mean_true.items() if z >= 0.3)
    summary = {
        "label": "SYNTHETIC",
        "seed": d["seed"], "start": d["start"], "end": d["end"], "stores": stores, "categories": cat_names,
        "n_series": S * C, "n_days": n_obs, "rows_possible": int(obs.sum()), "rows_written": int(len(sales)),
        "rows_missing": int(missing.sum()), "outliers_injected": int(outlier.sum()),
        "store_size": dict(zip(stores, size.round(3).tolist())),
        "store_growth_per_year": dict(zip(stores, growth.round(4).tolist())),
        "structural_break": br, "price_increase": pi,
        "intermittent_series": intermittent,
        "sales_sha256": file_sha256(raw / "sales.csv"),
    }
    (truth / "generator.json").write_text(json.dumps(summary, indent=2))
    return summary
