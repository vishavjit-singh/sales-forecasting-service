"""Parity of the NumPy Holt-Winters with statsmodels. statsmodels is installed in the CI `statsmodels-parity`
job; elsewhere these tests are skipped."""
import numpy as np
import pytest

from salesforecast.ets import HoltWintersBatch, heuristic_init, hw_filter, hw_forecast

sm = pytest.importorskip("statsmodels.tsa.holtwinters")


def _series(seed: int, n: int = 364) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    week = rng.normal(0, 8, 7)
    return 80 + 0.03 * t + week[t % 7] + rng.normal(0, 4, n)


@pytest.mark.parametrize("seed", [0, 1])
def test_fixed_parameters_give_identical_fits_and_forecasts(seed):
    y = _series(seed)
    l0, b0, s0 = (v[0] for v in heuristic_init(y[None, :], 7))
    a, b, g, phi = 0.25, 0.04, 0.15, 0.92
    model = sm.ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                    initialization_method="known", initial_level=l0, initial_trend=b0,
                                    initial_seasonal=s0)
    fit = model.fit(smoothing_level=a, smoothing_trend=b, smoothing_seasonal=g, damping_trend=phi,
                    optimized=False)
    ours, state = hw_filter(y[None, :], a, b, g, phi, l0, b0, s0, 7)
    assert np.allclose(ours[0], fit.fittedvalues, rtol=1e-8, atol=1e-6)
    # Same final states as statsmodels.
    assert np.allclose([state.level[0], state.trend[0]], [fit.level[-1], fit.trend[-1]], rtol=1e-8, atol=1e-6)
    # fit.season[-7:][i] is the seasonal of observation n−7+i, which our state keeps in slot (n + i) % 7.
    assert np.allclose(state.season[0, (len(y) + np.arange(7)) % 7], fit.season[-7:], rtol=1e-8, atol=1e-6)

    h = np.arange(1, 29)
    forecast = hw_forecast(state, phi, 28)[0]
    # ŷ_{n+h} = l_n + (φ + … + φ^h) b_n + s_{n+h−m(k+1)}, from statsmodels' own final states.
    textbook = fit.level[-1] + np.cumsum(phi ** h) * fit.trend[-1] + fit.season[-7:][(h - 1) % 7]
    assert np.allclose(forecast, textbook, rtol=1e-8, atol=1e-6)
    # statsmodels' forecast() agrees except at h = m, 2m, …: HoltWintersResults._predict overwrites the
    # seasonal updated by the last observation with the one from a cycle earlier
    # (s[nobs + m - 1:] = [s[(nobs - 1) + j % m] ...]), so those steps are compared to the textbook only.
    off_cycle = h % 7 != 0
    assert np.allclose(forecast[off_cycle], fit.forecast(28)[off_cycle], rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_grid_search_fit_is_close_to_statsmodels_optimiser(seed):
    y = _series(seed)
    ours = HoltWintersBatch(7).fit(y[None, :])
    ref = sm.ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                  initialization_method="estimated").fit()
    print(f"seed {seed}: SSE ours {ours.sse_[0]:.1f} vs statsmodels {ref.sse:.1f}")
    assert ours.sse_[0] <= ref.sse * 1.15
    diff = np.abs(ours.forecast(28)[0] - ref.forecast(28)).mean() / y[-28:].mean()
    assert diff < 0.05
