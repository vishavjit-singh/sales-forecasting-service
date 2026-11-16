"""Forecasting models behind one interface.

Every model is fitted on a ``TrainView`` (history up to a cutoff, plus the calendar and the price/promotion plan
that are known in advance) and returns a (n_series, horizon) array of non-negative point forecasts. Prediction
intervals are added afterwards from backtest residuals (``intervals.py``), the same way for every model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .ets import HoltWintersBatch
from .prep import Panel


@dataclass
class TrainView:
    """What a model may see when fitted at `cutoff`: no sales on or after the cutoff day."""
    y: np.ndarray                 # (n_series, cutoff)
    cutoff: int                   # index of the first forecast day
    stores: np.ndarray
    categories: np.ndarray
    price_index: np.ndarray       # (n_series, n_plan_days): known in advance
    promo: np.ndarray
    calendar: pd.DataFrame

    @classmethod
    def from_panel(cls, panel: Panel, cutoff: int) -> "TrainView":
        return cls(y=panel.y[:, :cutoff].copy(), cutoff=cutoff, stores=panel.stores, categories=panel.categories,
                   price_index=panel.price_index, promo=panel.promo, calendar=panel.calendar)


class Forecaster(ABC):
    name = "base"

    def fit(self, view: TrainView) -> "Forecaster":
        self._cache = {}
        self._fit(view)
        self.cutoff_, self.fitted_on_ = view.cutoff, view
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Non-negative point forecasts for the `horizon` days from the cutoff (cached until the next fit)."""
        if horizon not in self._cache:
            self._cache[horizon] = np.clip(self._predict(horizon), 0, None)
        return self._cache[horizon].copy()

    @abstractmethod
    def _fit(self, view: TrainView) -> None:
        ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        ...

    def describe(self, i: int) -> dict:
        """Parameters of series i, stored in the registry."""
        return {}


class SeasonalNaive(Forecaster):
    """Repeat the last full week."""
    name = "seasonal_naive"

    def __init__(self, season: int = 7):
        self.m = season

    def _fit(self, view):
        self.last_ = view.y[:, -self.m:]

    def _predict(self, horizon):
        reps = int(np.ceil(horizon / self.m))
        return np.tile(self.last_, reps)[:, :horizon]


class MovingAverage(Forecaster):
    """Flat forecast at the mean of the last `window` days."""
    name = "moving_average"

    def __init__(self, window: int = 28):
        self.window = window

    def _fit(self, view):
        self.level_ = view.y[:, -self.window:].mean(axis=1)

    def _predict(self, horizon):
        return np.repeat(self.level_[:, None], horizon, axis=1)


class ETS(Forecaster):
    """Additive damped Holt-Winters with weekly seasonality (NumPy, see ets.py)."""
    name = "ets"

    def __init__(self, season: int = 7, fit_window: int = 730):
        self.season, self.fit_window = season, fit_window

    def _fit(self, view):
        y = view.y[:, -self.fit_window:]
        y = y[:, (y.shape[1] % self.season):]           # start the window on a whole number of weeks
        self.hw_ = HoltWintersBatch(self.season).fit(y)

    def _predict(self, horizon):
        return self.hw_.forecast(horizon)

    def describe(self, i):
        return self.hw_.params_dict(i)


def croston(y: np.ndarray, alpha: float = 0.1, sba: bool = True) -> float:
    """Croston's method on one series; with `sba` the Syntetos-Boylan bias correction (1 − α/2)."""
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    z, p = float(y[nz[0]]), float(nz[0] + 1)
    for prev, cur in zip(nz[:-1], nz[1:]):
        z += alpha * (y[cur] - z)
        p += alpha * ((cur - prev) - p)
    return (1 - alpha / 2 if sba else 1.0) * z / p


class CrostonSBA(Forecaster):
    """Croston / SBA for intermittent demand: smoothed demand size over smoothed interval between demands."""
    name = "croston_sba"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def _fit(self, view):
        self.level_ = np.array([croston(row, self.alpha) for row in view.y])

    def _predict(self, horizon):
        return np.repeat(self.level_[:, None], horizon, axis=1)


CALENDAR_FEATURES = ["dow", "doy_sin", "doy_cos", "festival_days", "festival_ramp", "holiday", "year_end"]


def lag_features(H: np.ndarray, ts: np.ndarray, lags, windows) -> dict[str, np.ndarray]:
    """Lag and trailing-mean features for target days `ts`, from history H (n_series, ≥ max(ts)).

    Each feature for day t uses only H[:, :t]. The same function builds the training matrix and the
    features at every recursive forecast step, so the two cannot drift apart.
    """
    cs = np.concatenate([np.zeros((H.shape[0], 1)), np.cumsum(H, axis=1)], axis=1)
    out = {f"lag_{k}": H[:, ts - k] for k in lags}
    for w in windows:
        out[f"mean_{w}"] = (cs[:, ts] - cs[:, ts - w]) / w
    out["same_dow_mean_4w"] = sum(H[:, ts - 7 * j] for j in range(1, 5)) / 4
    return out


class GlobalGBM(Forecaster):
    """One gradient-boosting model for all series (Poisson loss), forecasting recursively day by day.

    Features: lags and trailing means of the series, same-weekday mean, store and category (categorical),
    calendar (weekday, annual cycle, festival window, holidays) and the planned promotion and price index.
    """
    name = "gbm"

    def __init__(self, lags=(1, 2, 7, 14, 21, 28), rolling=(7, 28), max_iter=250, learning_rate=0.06,
                 max_leaf_nodes=31, min_samples_leaf=40, train_window=730, n_threads=1, random_state=0):
        self.n_threads = n_threads           # OpenMP threads; 1 is fastest on a small shared machine
        self.lags, self.rolling = tuple(lags), tuple(rolling)
        self.train_window = train_window
        self.params = dict(loss="poisson", max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           random_state=random_state, early_stopping=False)

    @property
    def min_history(self) -> int:
        return max(max(self.lags), max(self.rolling), 28)

    def _frame(self, H, ts, view) -> pd.DataFrame:
        n = H.shape[0]
        feats = lag_features(H, ts, self.lags, self.rolling)
        cols = {k: v.ravel() for k, v in feats.items()}
        cols["store"] = np.repeat(self.store_codes_, len(ts))
        cols["category"] = np.repeat(self.cat_codes_, len(ts))
        cols["promo"] = view.promo[:, ts].ravel()
        cols["price_index"] = view.price_index[:, ts].ravel()
        cal = view.calendar.iloc[ts]
        for c in CALENDAR_FEATURES:
            cols[c] = np.tile(cal[c].to_numpy(), n)
        return pd.DataFrame(cols)

    def _fit(self, view):
        self.view_ = view
        self.stores_ = sorted(set(view.stores))
        self.cats_ = sorted(set(view.categories))
        self.store_codes_ = np.array([self.stores_.index(s) for s in view.stores])
        self.cat_codes_ = np.array([self.cats_.index(c) for c in view.categories])
        c = view.cutoff
        start = max(self.min_history, c - self.train_window)
        ts = np.arange(start, c)
        X = self._frame(view.y, ts, view)
        target = view.y[:, ts].ravel()
        self.features_ = list(X.columns)
        cat_mask = [f in ("store", "category") for f in self.features_]
        self.model_ = HistGradientBoostingRegressor(categorical_features=cat_mask, **self.params)
        with threadpool_limits(self.n_threads):
            self.model_.fit(X, target)
        self.train_rows_ = len(X)

    def _predict(self, horizon):
        view, c = self.view_, self.view_.cutoff
        H = np.concatenate([view.y, np.zeros((view.y.shape[0], horizon))], axis=1)
        with threadpool_limits(self.n_threads):
            for step in range(horizon):
                t = c + step
                X = self._frame(H, np.array([t]), view)
                H[:, t] = np.clip(self.model_.predict(X), 0, None)
        return H[:, c:c + horizon]

    def describe(self, i):
        return {"strategy": "recursive", "train_rows": int(self.train_rows_), "loss": "poisson",
                "max_iter": self.params["max_iter"]}


class Ensemble(Forecaster):
    """Mean of member forecasts. Members already fitted on the same view are reused, not refitted."""
    name = "ensemble"

    def __init__(self, members: list[Forecaster]):
        self.members = members

    def _fit(self, view):
        for m in self.members:
            if getattr(m, "fitted_on_", None) is not view:
                m.fit(view)

    def _predict(self, horizon):
        return np.mean([m.predict(horizon) for m in self.members], axis=0)

    def describe(self, i):
        return {"members": [m.name for m in self.members]}


def build_models(cfg: dict, names: list[str] | None = None) -> dict[str, Forecaster]:
    """Instantiate the configured models (the ensemble shares its members' instances)."""
    mc = cfg["models"]
    names = names or cfg["backtest"]["models"]
    factories = {
        "seasonal_naive": lambda: SeasonalNaive(),
        "moving_average": lambda: MovingAverage(**mc["moving_average"]),
        "ets": lambda: ETS(**mc["ets"]),
        "croston_sba": lambda: CrostonSBA(**mc["croston_sba"]),
        "gbm": lambda: GlobalGBM(**mc["gbm"]),
    }
    out: dict[str, Forecaster] = {}
    needed = list(names)
    if "ensemble" in names:
        needed += [m for m in mc["ensemble"]["members"] if m not in needed]
    for n in needed:
        if n != "ensemble":
            out[n] = factories[n]()
    if "ensemble" in names:
        out["ensemble"] = Ensemble([out[m] for m in mc["ensemble"]["members"]])
    return {n: out[n] for n in needed}
