"""Additive damped Holt-Winters, ETS(A,Ad,A), in NumPy.

statsmodels could not be installed where this was built, so the recursion is written out here with the same
equations and parameter conventions as ``statsmodels.tsa.holtwinters.ExponentialSmoothing(trend="add",
damped_trend=True, seasonal="add")``:

    level    l_t = α (y_t − s_{t−m}) + (1 − α)(l_{t−1} + φ b_{t−1})
    trend    b_t = β (l_t − l_{t−1}) + (1 − β) φ b_{t−1}
    season   s_t = γ (y_t − l_{t−1} − φ b_{t−1}) + (1 − γ) s_{t−m}
    forecast ŷ_{n+h} = l_n + (φ + φ² + … + φ^h) b_n + s_{n+h−m(k+1)}

The CI job ``statsmodels-parity`` checks the recursion against statsmodels with fixed parameters and compares
fitted SSE and forecasts after optimisation.

The recursion runs on many series (and many parameter sets) at once: each time step is a handful of vector
operations. Parameters (α, β, γ, φ) are chosen by a coarse grid followed by two local grid rounds on in-sample
SSE, for all series in one pass (about a second for 80 series × 730 days on one CPU). Initial states come from
the first four weeks and are not optimised.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

GRID = {"alpha": [0.03, 0.1, 0.25, 0.5], "beta": [0.001, 0.02, 0.1], "gamma": [0.01, 0.05, 0.15, 0.35],
        "phi": [0.85, 0.94, 0.98]}
BOUNDS = {"alpha": (1e-4, 0.9999), "beta": (1e-4, 0.9999), "gamma": (1e-4, 0.9999), "phi": (0.8, 0.995)}
NAMES = ("alpha", "beta", "gamma", "phi")


@dataclass
class HWState:
    level: np.ndarray        # (k,)
    trend: np.ndarray        # (k,)
    season: np.ndarray       # (k, m): season[:, j] applies to the next observation with (t % m) == j
    n_obs: int               # observations filtered (to know where the seasonal cycle is)


def heuristic_init(y: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial level, trend and seasonals from the first few cycles. y: (k, n)."""
    cycles = max(2, min(4, y.shape[1] // m))
    first = y[:, :m * cycles].reshape(y.shape[0], cycles, m)
    cyc_mean = first.mean(axis=2)
    level0 = cyc_mean[:, 0]
    trend0 = (cyc_mean[:, 1] - cyc_mean[:, 0]) / m
    season0 = (first - cyc_mean[:, :, None]).mean(axis=1)
    return level0, trend0, season0


def hw_filter(y: np.ndarray, alpha, beta, gamma, phi, level0, trend0, season0, m: int):
    """Run the recursion. y: (k, n); parameters and states broadcast over k.

    Returns one-step-ahead fitted values (k, n) and the final state.
    """
    y = np.atleast_2d(np.asarray(y, float))
    k, n = y.shape
    a, b_, g, p = (np.broadcast_to(np.asarray(v, float), (k,)) for v in (alpha, beta, gamma, phi))
    lvl = np.broadcast_to(np.asarray(level0, float), (k,)).copy()
    trd = np.broadcast_to(np.asarray(trend0, float), (k,)).copy()
    s = np.broadcast_to(np.asarray(season0, float), (k, m)).copy()
    fitted = np.empty((k, n))
    ac, bc, gc = 1 - a, 1 - b_, 1 - g
    for t in range(n):
        j = t % m
        sj = s[:, j]
        base = lvl + p * trd
        fitted[:, t] = base + sj
        yt = y[:, t]
        new_lvl = a * (yt - sj) + ac * base
        trd = b_ * (new_lvl - lvl) + bc * p * trd
        s[:, j] = g * (yt - base) + gc * sj
        lvl = new_lvl
    return fitted, HWState(lvl, trd, s, n)


def hw_forecast(state: HWState, phi, horizon: int) -> np.ndarray:
    m = state.season.shape[1]
    phi = np.broadcast_to(np.asarray(phi, float), state.level.shape)
    h = np.arange(1, horizon + 1)
    damp = np.cumsum(phi[:, None] ** h[None, :], axis=1)
    seas = state.season[:, (state.n_obs + h - 1) % m]
    return state.level[:, None] + damp * state.trend[:, None] + seas


def _grid_search(y, init, m, grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate parameter sets for every series at once. grids: (G, 4) shared, or (k, G, 4) per series.

    Returns the best parameter set and its in-sample SSE per series.
    """
    k, n = y.shape
    P = np.broadcast_to(grids, (k,) + grids.shape[-2:]) if grids.ndim == 2 else grids
    G = P.shape[1]
    flat = P.reshape(k * G, 4)
    yy = np.repeat(y, G, axis=0)                         # (k·G, n)
    l0, b0, s0 = (np.repeat(v, G, axis=0) for v in init)
    fitted, _ = hw_filter(yy, *flat.T, l0, b0, s0, m)
    sse = ((yy - fitted) ** 2).sum(axis=1).reshape(k, G)
    best = sse.argmin(axis=1)
    return P[np.arange(k), best], sse[np.arange(k), best]


def _refine_grids(centers: np.ndarray) -> np.ndarray:
    """A 3×3×3×3 local grid around each series' current best point: (k, 81, 4)."""
    axes = []
    for i, name in enumerate(NAMES):
        lo, hi = BOUNDS[name]
        factors = np.array((0.97, 1.0, 1.01) if name == "phi" else (0.55, 1.0, 1.6))
        axes.append(np.clip(centers[:, i:i + 1] * factors[None, :], lo, hi))      # (k, 3)
    idx = np.array(list(itertools.product(range(3), repeat=4)))                   # (81, 4)
    return np.stack([axes[i][:, idx[:, i]] for i in range(4)], axis=2)


class HoltWintersBatch:
    """Fit ETS(A,Ad,A) to many series at once: a coarse grid, then two local grid rounds, on in-sample SSE."""

    def __init__(self, season: int = 7):
        self.m = season

    def fit(self, y: np.ndarray) -> "HoltWintersBatch":
        y = np.atleast_2d(np.asarray(y, float))
        m = self.m
        init = heuristic_init(y, m)
        coarse = np.array(list(itertools.product(*(GRID[n] for n in NAMES))))
        best, _ = _grid_search(y, init, m, coarse)
        params = best
        for _ in range(2):                                               # two local refinement rounds
            params, sse_best = _grid_search(y, init, m, _refine_grids(params))
        self.params_ = params                             # (k, 4): alpha, beta, gamma, phi
        self.sse_ = sse_best
        _, self.state_ = hw_filter(y, *params.T, *init, m)
        self.init_ = init
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        return hw_forecast(self.state_, self.params_[:, 3], horizon)

    def params_dict(self, i: int) -> dict:
        return {n: round(float(v), 5) for n, v in zip(NAMES, self.params_[i])}
