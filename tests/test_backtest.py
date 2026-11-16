import numpy as np
import pandas as pd
import pytest

from salesforecast.backtest import SELECTED, classify_demand, make_cutoffs
from salesforecast.intervals import residual_offsets
from salesforecast.models import TrainView, build_models
from salesforecast.prep import load_panel


def test_cutoffs_form_an_expanding_window_ending_on_the_last_day():
    cuts = make_cutoffs(n_days=1096, horizon=28, n_cutoffs=6, step=28)
    assert cuts[-1] + 28 == 1096
    assert np.diff(cuts).tolist() == [28] * 5
    with pytest.raises(ValueError):
        make_cutoffs(n_days=400, horizon=28, n_cutoffs=6, step=28)


def test_train_view_holds_no_data_from_the_cutoff_on(project):
    p = project["panel"]
    v = TrainView.from_panel(p, 500)
    assert v.y.shape == (len(p.series), 500)
    v.y[:] = -1                                                   # a copy: the panel is untouched
    assert (p.y[:, :500] >= 0).all()


def test_no_leakage_future_sales_cannot_change_any_forecast(project, cfg, tmp_path):
    """Corrupt every raw sale from the cutoff on, rebuild the panel and refit: forecasts must not move."""
    root, panel = project["root"], project["panel"]
    cutoff = panel.n_days - 56
    cut_date = panel.dates[cutoff]
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/plan.csv").write_text((root / "data/raw/plan.csv").read_text())
    sales = pd.read_csv(root / "data/raw/sales.csv", parse_dates=["date"])
    late = sales["date"] >= cut_date
    sales.loc[late, "units"] = sales.loc[late, "units"] * 50 + 999
    sales.to_csv(tmp_path / "data/raw/sales.csv", index=False, date_format="%Y-%m-%d")
    corrupted = load_panel(cfg, tmp_path)
    assert not np.array_equal(corrupted.y[:, cutoff:], panel.y[:, cutoff:])

    a = build_models(cfg)
    b = build_models(cfg)
    va, vb = TrainView.from_panel(panel, cutoff), TrainView.from_panel(corrupted, cutoff)
    for name in a:
        fa, fb = a[name].fit(va).predict(28), b[name].fit(vb).predict(28)
        assert np.array_equal(fa, fb), f"{name} saw data after the cutoff"


def test_rolling_selection_uses_only_earlier_folds(project, cfg):
    res = project["res"]
    mt = res.metrics
    min_cal = cfg["backtest"]["min_calibration_folds"]
    assert (res.chosen_rolling[:min_cal] == -1).all() and np.isnan(res.preds[SELECTED][:min_cal]).all()
    for k in range(min_cal, len(res.cutoffs)):
        past = mt[(mt["fold"] < k) & (mt["model"] != SELECTED)]
        best = past.groupby(["series", "model"])["wape"].mean().unstack()[res.models].idxmin(axis=1)
        chosen = [res.models[j] for j in res.chosen_rolling[k]]
        assert chosen == [best[s] for s in project["panel"].series]


def test_intervals_are_calibrated_on_earlier_folds_only(project, cfg):
    res = project["res"]
    k, i, m, lv = len(res.cutoffs) - 1, 5, "gbm", 0.8
    offs = residual_offsets((res.actual[:k, i] - res.preds[m][:k, i]).ravel(), [lv])[lv]
    expected_lo = np.clip(res.preds[m][k, i] + offs[0], 0, None)
    assert np.allclose(res.lo[(m, lv)][k, i], expected_lo)
    assert np.isnan(res.lo[(m, lv)][0]).all()                     # calibration folds get no interval


def test_backtest_summary_is_sane(project):
    s = project["res"].summary
    ov = s["overall_scored_folds"]
    assert s["label"] == "SYNTHETIC"
    assert ov["gbm"]["wape"] < ov["seasonal_naive"]["wape"]
    assert ov["ets"]["wape"] < ov["seasonal_naive"]["wape"]
    for lv, (lo, hi) in {"0.8": (0.65, 0.92), "0.95": (0.85, 0.995)}.items():
        assert lo <= s["interval_coverage"][SELECTED][lv] <= hi
    assert len(s["wape_by_horizon"]["gbm"]) == s["horizon"]
    assert sum(s["selection_share"].values()) == s["n_series"]


def test_hierarchy_bottom_up_beats_naive_totals(project):
    h = project["res"].summary["hierarchy"]
    assert h["bottom_up_selected"]["wape"] < h["direct_seasonal_naive_on_totals"]["wape"]
    assert set(h["store_total_offsets"]) == set(project["panel"].stores)


def test_demand_classes():
    assert classify_demand(np.full(100, 10.0))[2] == "smooth"
    sparse = np.zeros(100)
    sparse[::4] = 3
    adi, cv2, cls = classify_demand(sparse)
    assert adi == pytest.approx(4.0) and cls == "intermittent"
    lumpy = np.zeros(100)
    lumpy[::5] = [1, 20] * 10
    assert classify_demand(lumpy)[2] == "lumpy"
    assert classify_demand(np.zeros(10))[2] == "no_demand"


def test_backtest_files(project):
    rep = project["root"] / "reports"
    preds = pd.read_csv(rep / "backtest_predictions.csv")
    assert {"lo_80", "hi_80", "lo_95", "hi_95", "model_used"} <= set(preds.columns)
    sel = preds[preds["model"] == SELECTED].dropna(subset=["lo_80"])
    assert (sel["lo_95"] <= sel["lo_80"] + 1e-9).all() and (sel["hi_80"] <= sel["hi_95"] + 1e-9).all()
    first_day = preds[preds["h"] == 1].groupby("fold")["date"].first()
    assert first_day.tolist() == project["res"].cutoff_dates
