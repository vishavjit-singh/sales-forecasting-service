import json

import numpy as np
import pandas as pd
import pytest

from salesforecast import events
from salesforecast.generate import file_sha256, generate
from salesforecast.prep import fill_gaps, load_panel, replace_outliers


def test_generator_is_deterministic(tmp_path, cfg, project):
    generate(cfg, tmp_path)
    assert file_sha256(tmp_path / "data/raw/sales.csv") == project["gen"]["sales_sha256"]


def test_generator_truth_accounts_for_every_row(project, cfg):
    g, root = project["gen"], project["root"]
    assert g["label"] == "SYNTHETIC"
    assert g["rows_written"] + g["rows_missing"] == g["rows_possible"]
    assert len(pd.read_csv(root / "data/raw/sales.csv")) == g["rows_written"]
    truth = pd.read_csv(root / "data/truth/components.csv")
    assert truth["is_outlier"].sum() == g["outliers_injected"] and truth["is_missing"].sum() == g["rows_missing"]
    # the structural break is applied exactly from its date, only to its store
    br = cfg["data"]["structural_break"]
    after = (truth["store"] == br["store"]) & (truth["date"] >= br["date"])
    assert set(truth.loc[after, "break_mult"]) == {br["factor"]} and set(truth.loc[~after, "break_mult"]) == {1.0}


def test_intermittent_series_come_from_the_low_volume_categories(project):
    inter = project["gen"]["intermittent_series"]
    assert inter and all(s.split("-", 1)[1] in ("electronics", "home_decor") for s in inter)


def test_plan_runs_past_the_sales_and_carries_the_price_rise(project, cfg):
    p = project["panel"]
    assert len(p.plan_dates) == p.n_days + cfg["data"]["plan_days_after_end"]
    i = p.series.index("S01-dairy")
    rise = p.plan_dates >= pd.Timestamp(cfg["data"]["price_increase"]["date"])
    normal = p.promo[i] == 0
    assert np.allclose(p.price_index[i, rise & normal], cfg["data"]["price_increase"]["factor"])
    assert np.allclose(p.price_index[i, ~rise & normal], 1.0)


def test_fill_gaps_uses_the_same_weekday_of_previous_weeks():
    y = np.arange(1, 43, dtype=float)[None, :]           # 6 weeks, strictly increasing
    y[0, 35] = np.nan                                     # day 35: previous same weekdays are days 28, 21, 14, 7
    filled, mask = fill_gaps(y, weeks=4)
    assert mask.sum() == 1 and mask[0, 35]
    assert filled[0, 35] == np.median([29, 22, 15, 8])
    assert not np.isnan(filled).any()


def test_outlier_replacement_respects_exempt_days():
    rng = np.random.default_rng(0)
    x = rng.poisson(50, (2, 120)).astype(float)
    x[:, 100] = 400
    exempt = np.zeros_like(x, bool)
    exempt[1, 100] = True                                 # e.g. a promotion day
    out, flag = replace_outliers(x, exempt, window=28, k=6.0)
    assert flag[0, 100] and out[0, 100] < 120
    assert not flag[1, 100] and out[1, 100] == 400
    assert flag.sum() == 1


def test_cleaning_is_causal(project, cfg, tmp_path):
    """Changing raw sales after a day must not change any cleaned value before it."""
    root = project["root"]
    cut = project["panel"].dates[-60]
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/plan.csv").write_text((root / "data/raw/plan.csv").read_text())
    sales = pd.read_csv(root / "data/raw/sales.csv", parse_dates=["date"])
    late = sales["date"] >= cut
    sales.loc[late, "units"] = sales.loc[late, "units"] * 10 + 7
    sales = sales.drop(sales[late].sample(frac=0.2, random_state=1).index)
    sales.to_csv(tmp_path / "data/raw/sales.csv", index=False, date_format="%Y-%m-%d")
    other = load_panel(cfg, tmp_path)
    c = project["panel"].dates.get_loc(cut)
    assert np.array_equal(other.y[:, :c], project["panel"].y[:, :c])


def test_outlier_check_finds_most_injected_outliers(project):
    p, root = project["panel"], project["root"]
    truth = pd.read_csv(root / "data/truth/components.csv")
    truth["series"] = truth["store"] + "-" + truth["category"]
    o = truth.pivot(index="series", columns="date", values="is_outlier").reindex(p.series).to_numpy().astype(bool)
    T = p.n_days
    exempt = (p.promo[:, :T] == 1) | (p.calendar["event"].to_numpy()[None, :T] == 1)
    big = np.isin(p.categories, ["electronics", "home_decor"], invert=True)[:, None]
    target = o & ~exempt & big
    assert (p.replaced & target).sum() / target.sum() >= 0.6
    assert (p.replaced & ~o).sum() <= 0.002 * p.y.size        # false flags stay rare
    meta = json.loads((root / "data/prepared/prep_report.json").read_text())
    assert meta["cells_imputed"] == project["gen"]["rows_missing"]


def test_calendar_features(cfg):
    cal = cfg["calendar"]
    peak = pd.Timestamp(cal["festival"]["peaks"][1])
    dates = pd.date_range(peak - pd.Timedelta(days=30), peak + pd.Timedelta(days=5))
    f = events.calendar_frame(dates, cal)
    assert f.loc[peak, "festival_days"] == 0 and f.loc[peak, "festival_ramp"] == pytest.approx(1.0)
    assert f.loc[peak - pd.Timedelta(days=10), "festival_days"] == 10
    assert f.loc[peak - pd.Timedelta(days=25), "festival_days"] == events.OUTSIDE
    assert f.loc[peak + pd.Timedelta(days=1), "festival_ramp"] == pytest.approx(0.4)
    hol = events.calendar_frame(pd.DatetimeIndex(["2024-08-15", "2024-08-16", "2024-12-27"]), cal)
    assert hol["holiday"].tolist() == [1, 0, 0] and hol["year_end"].tolist() == [0, 0, 1]
