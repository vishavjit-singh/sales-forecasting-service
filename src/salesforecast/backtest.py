"""Rolling-origin backtest: expanding window, several cutoffs, one fixed horizon.

For each cutoff every model is refitted on the history before it (``TrainView``) and forecasts the next
`horizon` days. Forecasts are scored against the cleaned actuals; days that were missing in the raw data are
left out of the scores.

On top of the raw per-model scores:
- **per-series selection** ("selected"): for fold k, each series uses the model with the lowest mean WAPE on
  folds before k, so the selection itself is scored out of sample;
- **prediction intervals** for fold k come from residuals of folds before k (``intervals.py``);
- **hierarchy**: store totals forecast bottom-up (sum of the selected category forecasts) against a direct
  Holt-Winters model on the store totals.
Folds from `min_calibration_folds` onward are the "scored folds" where all of the above exist.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics
from .ets import HoltWintersBatch
from .intervals import apply_offsets, residual_offsets
from .models import TrainView, build_models
from .prep import Panel

log = logging.getLogger(__name__)
SELECTED = "selected"


def make_cutoffs(n_days: int, horizon: int, n_cutoffs: int, step: int) -> list[int]:
    """Cutoff indices (first forecast day), oldest first. The last test window ends on the last day."""
    last = n_days - horizon
    cuts = [last - step * (n_cutoffs - 1 - k) for k in range(n_cutoffs)]
    if cuts[0] < 365:
        raise ValueError("not enough history for the first cutoff (need a year)")
    return cuts


def classify_demand(y: np.ndarray) -> tuple[float, float, str]:
    """Syntetos-Boylan classes from the average demand interval (ADI) and squared CV of non-zero sizes."""
    nz = y[y > 0]
    if len(nz) == 0:
        return float("inf"), 0.0, "no_demand"
    adi = len(y) / len(nz)
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    if adi < 1.32:
        cls = "smooth" if cv2 < 0.49 else "erratic"
    else:
        cls = "intermittent" if cv2 < 0.49 else "lumpy"
    return float(adi), cv2, cls


def _masked(a, mask):
    return np.where(mask, a, np.nan)


def _wape_nan(y, f):
    ok = ~np.isnan(y)
    return metrics.wape(y[ok], f[ok]) if ok.any() else float("nan")


@dataclass
class BacktestResult:
    cutoffs: list[int]
    cutoff_dates: list[str]
    horizon: int
    models: list[str]
    actual: np.ndarray                    # (K, n, h), NaN where the raw day was missing
    preds: dict[str, np.ndarray]          # model → (K, n, h); "selected" is NaN on calibration folds
    lo: dict[tuple[str, float], np.ndarray] = field(default_factory=dict)
    hi: dict[tuple[str, float], np.ndarray] = field(default_factory=dict)
    scales: np.ndarray | None = None      # (K, n) MASE denominators
    chosen_rolling: np.ndarray | None = None   # (K, n) model index used by "selected" (−1 on calibration folds)
    metrics: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)
    selection: pd.DataFrame | None = None


def run_backtest(cfg: dict, panel: Panel) -> BacktestResult:
    bt = cfg["backtest"]
    h, levels, min_cal = bt["horizon"], bt["interval_levels"], bt["min_calibration_folds"]
    cuts = make_cutoffs(panel.n_days, h, bt["n_cutoffs"], bt["step"])
    K, n = len(cuts), len(panel.series)
    names = list(bt["models"])
    preds = {m: np.full((K, n, h), np.nan) for m in names}
    actual = np.full((K, n, h), np.nan)
    scales = np.zeros((K, n))
    timings = {m: 0.0 for m in names}
    for k, c in enumerate(cuts):
        view = TrainView.from_panel(panel, c)
        actual[k] = _masked(panel.y[:, c:c + h], ~panel.imputed[:, c:c + h])
        scales[k] = [metrics.mase_scale(row) for row in view.y]
        models = build_models(cfg, names)
        for m in names:
            t0 = time.perf_counter()
            models[m].fit(view)
            preds[m][k] = models[m].predict(h)
            timings[m] += time.perf_counter() - t0
        log.info("fold %d/%d cutoff %s done", k + 1, K, panel.dates[c].date())

    res = BacktestResult(cutoffs=cuts, cutoff_dates=[str(panel.dates[c].date()) for c in cuts], horizon=h,
                         models=names, actual=actual, preds=preds, scales=scales)
    _rolling_selection(res, min_cal)
    _intervals(res, levels, min_cal)
    res.metrics = _metrics_table(res, panel)
    res.selection = _final_selection(res, panel, levels)
    res.summary = _summary(res, panel, cfg, timings)
    return res


def _fold_wape(res: BacktestResult, model: str) -> np.ndarray:
    """(K, n) WAPE per fold and series."""
    p, a = res.preds[model], res.actual
    out = np.full(p.shape[:2], np.nan)
    for k in range(p.shape[0]):
        for i in range(p.shape[1]):
            if not np.isnan(p[k, i]).all():
                out[k, i] = _wape_nan(a[k, i], p[k, i])
    return out


def _rolling_selection(res: BacktestResult, min_cal: int) -> None:
    K, n, h = res.actual.shape
    W = np.stack([_fold_wape(res, m) for m in res.models])          # (M, K, n)
    chosen = np.full((K, n), -1)
    sel = np.full((K, n, h), np.nan)
    for k in range(min_cal, K):
        with np.errstate(invalid="ignore"):
            score = np.nanmean(np.where(np.isnan(W[:, :k]), np.inf, W[:, :k]), axis=1)    # (M, n)
        chosen[k] = score.argmin(axis=0)
        for i in range(n):
            sel[k, i] = res.preds[res.models[chosen[k, i]]][k, i]
    res.preds[SELECTED] = sel
    res.chosen_rolling = chosen


def _intervals(res: BacktestResult, levels, min_cal: int) -> None:
    K, n, h = res.actual.shape
    resid = {m: res.actual - res.preds[m] for m in res.models}
    for m in res.models + [SELECTED]:
        for lv in levels:
            res.lo[(m, lv)] = np.full((K, n, h), np.nan)
            res.hi[(m, lv)] = np.full((K, n, h), np.nan)
    for k in range(min_cal, K):
        for i in range(n):
            for m in res.models:
                offs = residual_offsets(resid[m][:k, i].ravel(), levels)
                for lv in levels:
                    res.lo[(m, lv)][k, i], res.hi[(m, lv)][k, i] = apply_offsets(res.preds[m][k, i], offs[lv])
            m_sel = res.models[res.chosen_rolling[k, i]]
            for lv in levels:
                res.lo[(SELECTED, lv)][k, i] = res.lo[(m_sel, lv)][k, i]
                res.hi[(SELECTED, lv)][k, i] = res.hi[(m_sel, lv)][k, i]


def _metrics_table(res: BacktestResult, panel: Panel) -> pd.DataFrame:
    rows = []
    for m in res.models + [SELECTED]:
        for k, c in enumerate(res.cutoffs):
            for i, sid in enumerate(panel.series):
                f, y = res.preds[m][k, i], res.actual[k, i]
                ok = ~np.isnan(y) & ~np.isnan(f)
                if not ok.any():
                    continue
                r = {"series": sid, "store": panel.stores[i], "category": panel.categories[i], "model": m,
                     "fold": k, "cutoff": res.cutoff_dates[k], "actual_sum": float(y[ok].sum()),
                     **metrics.all_metrics(y[ok], f[ok], res.scales[k, i])}
                rows.append(r)
    return pd.DataFrame(rows)


def _final_selection(res: BacktestResult, panel: Panel, levels) -> pd.DataFrame:
    """Production choice per series: lowest mean WAPE over all folds; offsets from that model's residuals."""
    W = np.stack([_fold_wape(res, m) for m in res.models])          # (M, K, n)
    mean_w = np.nanmean(np.where(np.isnan(W), np.inf, W), axis=1)    # (M, n)
    rows = []
    y_hist = panel.y[:, :res.cutoffs[-1]]
    for i, sid in enumerate(panel.series):
        j = int(mean_w[:, i].argmin())
        m = res.models[j]
        offs = residual_offsets((res.actual[:, i] - res.preds[m][:, i]).ravel(), levels)
        adi, cv2, cls = classify_demand(y_hist[i, -365:])
        row = {"series": sid, "store": panel.stores[i], "category": panel.categories[i], "model": m,
               "demand_class": cls, "adi": round(adi, 3), "cv2": round(cv2, 3)}
        row.update({f"wape_{name}": round(float(mean_w[jj, i]), 4) for jj, name in enumerate(res.models)})
        for lv in levels:
            row[f"q_lo_{int(lv * 100)}"], row[f"q_hi_{int(lv * 100)}"] = (round(v, 3) for v in offs[lv])
        rows.append(row)
    return pd.DataFrame(rows)


def _scored(res: BacktestResult, min_cal: int):
    return slice(min_cal, len(res.cutoffs))


def _agg(a: np.ndarray, f: np.ndarray) -> dict:
    ok = ~np.isnan(a) & ~np.isnan(f)
    y, p = a[ok], f[ok]
    return {"wape": round(metrics.wape(y, p), 4), "bias": round(metrics.bias(y, p), 4),
            "smape": round(metrics.smape(y, p), 4), "rmse": round(metrics.rmse(y, p), 3)}


def _hierarchy(res: BacktestResult, panel: Panel, cfg: dict, sl: slice, levels) -> dict:
    stores = sorted(set(panel.stores))
    groups = [np.flatnonzero(panel.stores == s) for s in stores]
    K, n, h = res.actual.shape
    complete = ~np.isnan(res.actual)

    def totals(arr):
        return np.stack([arr[:, g].sum(axis=1) for g in groups], axis=1)    # (K, S, h)

    act = totals(np.nan_to_num(res.actual))
    ok = np.stack([complete[:, g].all(axis=1) for g in groups], axis=1)
    act = np.where(ok, act, np.nan)
    bu_sel = totals(np.nan_to_num(res.preds[SELECTED]))
    bu_gbm = totals(res.preds["gbm"]) if "gbm" in res.preds else None
    direct = np.full_like(act, np.nan)
    snaive = np.full_like(act, np.nan)
    tot_hist = np.stack([panel.y[g].sum(axis=0) for g in groups])
    m = cfg["models"]["ets"]
    for k, c in enumerate(res.cutoffs):
        y = tot_hist[:, max(0, c - m["fit_window"]):c]
        y = y[:, y.shape[1] % m["season"]:]
        direct[k] = np.clip(HoltWintersBatch(m["season"]).fit(y).forecast(h), 0, None)
        last = tot_hist[:, c - 7:c]
        snaive[k] = np.tile(last, int(np.ceil(h / 7)))[:, :h]
    out = {"stores": len(stores),
           "bottom_up_selected": _agg(act[sl], bu_sel[sl]),
           "direct_ets_on_totals": _agg(act[sl], direct[sl]),
           "direct_seasonal_naive_on_totals": _agg(act[sl], snaive[sl])}
    if bu_gbm is not None:
        out["bottom_up_gbm"] = _agg(act[sl], bu_gbm[sl])
    # intervals for store totals: residuals on earlier folds of the bottom-up total built with fold k's choices
    cov = {}
    hits = {lv: [] for lv in levels}
    for k in range(sl.start, K):
        choice = res.chosen_rolling[k]
        past = np.stack([res.preds[res.models[choice[i]]][:k, i] for i in range(n)], axis=1)    # (k, n, h)
        resid = act[:k] - totals(past)
        for s in range(len(stores)):
            offs = residual_offsets(resid[:, s].ravel(), levels)
            a = act[k, s]
            good = ~np.isnan(a)
            for lv in levels:
                lo, hi = apply_offsets(bu_sel[k, s], offs[lv])
                hits[lv].append(((a >= lo) & (a <= hi))[good])
    for lv in levels:
        cov[str(lv)] = round(float(np.concatenate(hits[lv]).mean()), 4)
    out["interval_coverage_store_totals"] = cov
    # production offsets for store totals: all folds, bottom-up with the final per-series choice
    final = [res.models.index(m) for m in res.selection["model"]]
    allp = np.stack([res.preds[res.models[final[i]]][:, i] for i in range(n)], axis=1)
    resid_all = act - totals(allp)
    out["store_total_offsets"] = {
        st: {str(lv): [round(v, 3) for v in o] for lv, o in residual_offsets(resid_all[:, s].ravel(), levels).items()}
        for s, st in enumerate(stores)}
    return out


def _summary(res: BacktestResult, panel: Panel, cfg: dict, timings: dict) -> dict:
    bt = cfg["backtest"]
    min_cal, levels = bt["min_calibration_folds"], bt["interval_levels"]
    sl = _scored(res, min_cal)
    allm = res.models + [SELECTED]
    a = res.actual
    overall = {m: _agg(a[sl], res.preds[m][sl]) for m in allm}
    overall_all = {m: _agg(a, res.preds[m]) for m in res.models}
    mt = res.metrics[res.metrics["fold"] >= min_cal]
    for m in allm:
        sub = mt[mt["model"] == m]
        overall[m]["mase_mean"] = round(float(sub["mase"].mean()), 4)
        overall[m]["series_wape_median"] = round(float(sub.groupby("series")["wape"].mean().median()), 4)

    # error by horizon day (all series, scored folds)
    by_h = {}
    for m in allm:
        p = res.preds[m][sl]
        by_h[m] = [round(_wape_nan(a[sl][:, :, j].ravel(), p[:, :, j].ravel()), 4) for j in range(res.horizon)]

    # coverage per model and level on scored folds
    cov = {}
    for m in allm:
        cov[m] = {}
        for lv in levels:
            lo, hi = res.lo[(m, lv)][sl], res.hi[(m, lv)][sl]
            ok = ~np.isnan(a[sl])
            cov[m][str(lv)] = round(metrics.coverage(a[sl][ok], lo[ok], hi[ok]), 4)

    sel = res.selection
    classes = dict(zip(sel["series"], sel["demand_class"]))
    cls_arr = np.array([classes[s] for s in panel.series])
    cov_by_class = {}
    for cl in sorted(set(cls_arr)):
        idx = cls_arr == cl
        cov_by_class[cl] = {}
        for lv in levels:
            aa = a[sl][:, idx]
            ok = ~np.isnan(aa)
            lo, hi = res.lo[(SELECTED, lv)][sl][:, idx], res.hi[(SELECTED, lv)][sl][:, idx]
            cov_by_class[cl][str(lv)] = round(metrics.coverage(aa[ok], lo[ok], hi[ok]), 4)
    width = {}
    for lv in levels:
        lo, hi = res.lo[(SELECTED, lv)][sl], res.hi[(SELECTED, lv)][sl]
        width[str(lv)] = round(float(np.nansum(hi - lo) / np.nansum(res.preds[SELECTED][sl])), 4)

    # festival window and the structural-break store
    fd = panel.calendar["festival_days"].to_numpy()
    in_fest = np.stack([[fd[c:c + res.horizon] != 99 for _ in panel.series] for c in res.cutoffs])
    festival = {}
    for m in allm:
        p = res.preds[m][sl]
        festival[m] = {"inside": round(_wape_nan(a[sl][in_fest[sl]], p[in_fest[sl]]), 4),
                       "outside": round(_wape_nan(a[sl][~in_fest[sl]], p[~in_fest[sl]]), 4)}
    br = cfg["data"]["structural_break"]["store"]
    bidx = panel.stores == br
    break_store = {m: _agg(a[sl][:, bidx], res.preds[m][sl][:, bidx])["wape"] for m in allm}

    share = sel["model"].value_counts().to_dict()
    share_by_class = sel.groupby("demand_class")["model"].value_counts().unstack(fill_value=0)
    best_single = min(res.models, key=lambda m: overall[m]["wape"])
    return {
        "label": "SYNTHETIC",
        "n_series": len(panel.series), "horizon": res.horizon, "cutoffs": res.cutoff_dates,
        "scored_folds": res.cutoff_dates[min_cal:], "calibration_folds": res.cutoff_dates[:min_cal],
        "data_sha256": panel.data_sha256,
        "overall_scored_folds": overall, "overall_all_folds": overall_all,
        "best_single_model": best_single,
        "selected_vs_best_single_wape": [overall[SELECTED]["wape"], overall[best_single]["wape"]],
        "wape_by_horizon": by_h,
        "interval_coverage": cov, "interval_coverage_selected_by_class": cov_by_class,
        "interval_relative_width_selected": width,
        "festival_window_wape": festival, "break_store": {"store": br, "wape": break_store},
        "selection_share": {k: int(v) for k, v in share.items()},
        "selection_by_class": {c: {k: int(v) for k, v in r.items() if v} for c, r in share_by_class.iterrows()},
        "demand_classes": {k: int(v) for k, v in sel["demand_class"].value_counts().items()},
        "hierarchy": _hierarchy(res, panel, cfg, sl, levels),
        "fit_seconds": {m: round(v, 2) for m, v in timings.items()},
    }


def predictions_frame(res: BacktestResult, panel: Panel) -> pd.DataFrame:
    """Long table: one row per series × model × fold × horizon day, with intervals where available."""
    K, n, h = res.actual.shape
    frames = []
    for m in res.models + [SELECTED]:
        for k, c in enumerate(res.cutoffs):
            d = panel.dates[c:c + h].strftime("%Y-%m-%d")
            df = pd.DataFrame({"series": np.repeat(panel.series, h), "model": m, "fold": k,
                               "date": np.tile(d, n), "h": np.tile(np.arange(1, h + 1), n),
                               "actual": res.actual[k].ravel(), "forecast": res.preds[m][k].ravel()})
            if m == SELECTED:
                used = [res.models[j] if j >= 0 else "" for j in res.chosen_rolling[k]]
                df["model_used"] = np.repeat(used, h)
            else:
                df["model_used"] = m
            for (mm, lv), arr in res.lo.items():
                if mm == m:
                    df[f"lo_{int(lv * 100)}"] = arr[k].ravel()
                    df[f"hi_{int(lv * 100)}"] = res.hi[(mm, lv)][k].ravel()
            frames.append(df.dropna(subset=["forecast"]))
    return pd.concat(frames, ignore_index=True).round(3)


def write_backtest(res: BacktestResult, panel: Panel, root: Path | None = None) -> Path:
    out = Path(root or config.ROOT) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    res.metrics.round(5).to_csv(out / "backtest_metrics.csv", index=False)
    predictions_frame(res, panel).to_csv(out / "backtest_predictions.csv", index=False)
    res.selection.to_csv(out / "selection.csv", index=False)
    (out / "backtest_summary.json").write_text(json.dumps(res.summary, indent=2))
    return out
