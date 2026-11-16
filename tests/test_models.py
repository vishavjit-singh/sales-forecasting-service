import numpy as np
import pandas as pd
import pytest

from salesforecast.ets import HoltWintersBatch, heuristic_init, hw_filter, hw_forecast
from salesforecast.models import (CrostonSBA, Ensemble, GlobalGBM, MovingAverage, SeasonalNaive, TrainView,
                                  build_models, croston, lag_features)


def _toy_view(y: np.ndarray, extra_days: int = 60) -> TrainView:
    n, T = y.shape
    dates = pd.date_range("2024-01-01", periods=T + extra_days)
    cal = pd.DataFrame({"dow": dates.dayofweek, "doy_sin": 0.0, "doy_cos": 0.0, "festival_days": 99,
                        "festival_ramp": 0.0, "holiday": 0, "year_end": 0}, index=dates)
    return TrainView(y=y, cutoff=T, stores=np.array(["S01"] * n), categories=np.array([f"c{i}" for i in range(n)]),
                     price_index=np.ones((n, T + extra_days)), promo=np.zeros((n, T + extra_days), int),
                     calendar=cal)


def reference_hw(y, a, b, g, phi, l0, b0, s0, m):
    """Textbook ETS(A,Ad,A) written out with scalars, independent of the vectorised implementation."""
    level, trend, season = l0, b0, list(s0)
    fitted = []
    for t, yt in enumerate(y):
        s_old = season[t % m]
        fitted.append(level + phi * trend + s_old)
        new_level = a * (yt - s_old) + (1 - a) * (level + phi * trend)
        new_trend = b * (new_level - level) + (1 - b) * phi * trend
        season[t % m] = g * (yt - level - phi * trend) + (1 - g) * s_old
        level, trend = new_level, new_trend
    fc = [level + sum(phi ** j for j in range(1, h + 1)) * trend + season[(len(y) + h - 1) % m] for h in range(1, 15)]
    return np.array(fitted), np.array(fc)


def test_vectorised_holt_winters_matches_the_scalar_reference():
    rng = np.random.default_rng(3)
    y = 40 + 0.05 * np.arange(200) + 6 * np.sin(np.arange(200) * 2 * np.pi / 7) + rng.normal(0, 2, 200)
    l0, b0, s0 = (v[0] for v in heuristic_init(y[None, :], 7))
    params = (0.3, 0.05, 0.2, 0.93)
    fitted, state = hw_filter(y[None, :], *params, l0, b0, s0, 7)
    ref_fit, ref_fc = reference_hw(y, *params, l0, b0, s0, 7)
    assert np.allclose(fitted[0], ref_fit) and np.allclose(hw_forecast(state, params[3], 14)[0], ref_fc)


def test_holt_winters_fit_recovers_a_clean_seasonal_series():
    t = np.arange(364 + 28)
    pattern = np.array([0, -5, -3, 2, 6, 12, 8], float)
    truth = 100 + 0.04 * t + pattern[t % 7]
    y = truth + np.random.default_rng(0).normal(0, 1.5, len(t))
    hw = HoltWintersBatch(7).fit(y[None, :364])
    fc = hw.forecast(28)[0]
    assert np.abs(fc - truth[364:]).mean() / truth[364:].mean() < 0.03
    default_fit, _ = hw_filter(y[None, :364], 0.5, 0.1, 0.1, 0.9, *heuristic_init(y[None, :364], 7), 7)
    assert hw.sse_[0] <= ((y[:364] - default_fit[0]) ** 2).sum()          # the search beats a fixed guess


def test_seasonal_naive_and_moving_average():
    y = np.tile(np.arange(7, dtype=float), 8)[None, :]
    v = _toy_view(y)
    assert SeasonalNaive().fit(v).predict(10)[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
    assert np.allclose(MovingAverage(window=28).fit(v).predict(5), 3.0)


def test_croston_sba_on_a_hand_worked_series():
    y = np.array([0, 0, 3, 0, 0, 0, 3, 0, 5], float)
    # demands at t=2,6,8 → z: 3 → 3 → 3.2; intervals: 3 → 3.1 → 2.99
    z, p = 3 + 0.1 * (5 - 3), (3 + 0.1 * (4 - 3)) + 0.1 * (2 - (3 + 0.1 * (4 - 3)))
    assert croston(y, alpha=0.1, sba=False) == pytest.approx(z / p)
    assert croston(y, alpha=0.1) == pytest.approx(0.95 * z / p)
    assert croston(np.zeros(10)) == 0.0
    f = CrostonSBA(0.1).fit(_toy_view(y[None, :])).predict(4)
    assert np.allclose(f, 0.95 * z / p)


def test_gbm_training_features_equal_the_recursive_features():
    rng = np.random.default_rng(1)
    H = rng.poisson(20, (3, 120)).astype(float)
    lags, windows = (1, 2, 7, 14, 21, 28), (7, 28)
    ts = np.arange(40, 120)
    train = lag_features(H, ts, lags, windows)
    for t in (40, 77, 119):
        # at a recursive step only H[:, :t] exists; pad the future with junk to prove it is not read
        padded = np.concatenate([H[:, :t], np.full((3, 5), 1e9)], axis=1)
        one = lag_features(padded, np.array([t]), lags, windows)
        for k in train:
            assert np.allclose(train[k][:, t - 40], one[k][:, 0]), k


def test_every_model_returns_nonnegative_finite_forecasts(project, cfg):
    panel = project["panel"]
    view = TrainView.from_panel(panel, panel.n_days - 28)
    for name, model in build_models(cfg).items():
        f = model.fit(view).predict(14)
        assert f.shape == (len(panel.series), 14), name
        assert np.isfinite(f).all() and (f >= 0).all(), name


def test_ensemble_is_the_mean_of_members_without_refitting(project, cfg):
    panel = project["panel"]
    view = TrainView.from_panel(panel, panel.n_days - 28)
    ms = build_models(cfg, ["ets", "gbm", "ensemble"])
    ms["ets"].fit(view)
    ms["gbm"].fit(view)
    gbm_model = ms["gbm"].model_
    ens = ms["ensemble"]
    assert isinstance(ens, Ensemble) and ens.members == [ms["ets"], ms["gbm"]]
    ens.fit(view)
    assert ms["gbm"].model_ is gbm_model                                     # reused, not refitted
    assert np.allclose(ens.predict(7), (ms["ets"].predict(7) + ms["gbm"].predict(7)) / 2)
    assert isinstance(ms["gbm"], GlobalGBM) and ms["gbm"].describe(0)["strategy"] == "recursive"
