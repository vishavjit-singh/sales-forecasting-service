import numpy as np
import pytest

from salesforecast import metrics
from salesforecast.intervals import apply_offsets, residual_offsets


def test_point_metrics_on_known_values():
    y, f = np.array([10, 0, 30, 20]), np.array([12, 1, 24, 20])
    assert metrics.mae(y, f) == pytest.approx(9 / 4)
    assert metrics.rmse(y, f) == pytest.approx(np.sqrt((4 + 1 + 36) / 4))
    assert metrics.wape(y, f) == pytest.approx(9 / 60)
    assert metrics.bias(y, f) == pytest.approx(-3 / 60)
    assert np.isnan(metrics.wape([0, 0], [1, 1]))


def test_smape_handles_zero_days_and_mase_scale():
    assert metrics.smape([0, 0], [0, 0]) == 0.0
    assert metrics.smape([0, 10], [0, 10]) == 0.0
    assert metrics.smape([0], [5]) == pytest.approx(2.0)
    hist = np.tile([1, 2, 3, 4, 5, 6, 7], 3) + np.r_[np.zeros(7), np.ones(7), 2 * np.ones(7)]
    assert metrics.mase_scale(hist, 7) == pytest.approx(1.0)        # each week is one unit above the last
    assert metrics.mase([3, 3], [4, 2], 2.0) == pytest.approx(0.5)
    assert np.isnan(metrics.mase_scale(np.ones(30)))                  # constant history: undefined


def test_residual_offsets_use_conformal_ranks():
    r = np.arange(1, 20, dtype=float)                                  # n = 19 residuals 1..19
    (lo80, hi80), (lo95, hi95) = residual_offsets(r, [0.8, 0.95]).values()
    assert (lo80, hi80) == (2.0, 18.0)                                 # ranks floor(20·0.1)=2, ceil(20·0.9)=18
    assert (lo95, hi95) == (1.0, 19.0)
    assert residual_offsets([np.nan], [0.8])[0.8] == (0.0, 0.0)


def test_apply_offsets_clips_at_zero_and_coverage():
    lo, hi = apply_offsets(np.array([1.0, 5.0]), (-3.0, 2.0))
    assert lo.tolist() == [0.0, 2.0] and hi.tolist() == [3.0, 7.0]
    assert metrics.coverage([0, 4, 8], [0, 3, 9], [1, 5, 10]) == pytest.approx(2 / 3)
