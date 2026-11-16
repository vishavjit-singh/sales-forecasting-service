"""Train the chosen model per series on all data and publish a versioned registry entry.

models/registry/<version>/
    metadata.json         version, data SHA-256, training cutoff, forecast window, backtest headline, versions
    registry.json         per series: chosen model, its parameters, backtest WAPE per model, interval offsets
    forecasts.csv         per series and day: point forecast and 80% / 95% intervals
    store_totals.csv      bottom-up store totals with their own intervals
    backtest_metrics.csv  per series × model × fold scores (served by GET /backtest/{series})
    gbm.joblib            the fitted global model, when any series uses it
models/registry/LATEST    the version the API serves
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config
from .intervals import apply_offsets
from .models import TrainView, build_models
from .prep import load_panel

log = logging.getLogger(__name__)


def _config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def train(cfg: dict, root: Path | None = None) -> Path:
    root = Path(root or config.ROOT)
    reports = root / "reports"
    summary_path = reports / "backtest_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run `python -m salesforecast backtest` first: model selection comes from it")
    summary = json.loads(summary_path.read_text())
    selection = pd.read_csv(reports / "selection.csv")
    panel = load_panel(cfg, root)
    if summary["data_sha256"] != panel.data_sha256:
        raise RuntimeError("data/raw/sales.csv changed since the backtest; run the backtest again")
    selection = selection.set_index("series").loc[panel.series].reset_index()

    H = cfg["api"]["max_horizon"]
    levels = cfg["backtest"]["interval_levels"]
    T = panel.n_days
    view = TrainView.from_panel(panel, T)
    needed = sorted(set(selection["model"]))
    models = build_models(cfg, needed)
    for m in models.values():
        m.fit(view)
    preds = {name: m.predict(H) for name, m in models.items()}

    fdates = panel.plan_dates[T:T + H]
    version = f"{panel.dates[-1].date()}_{panel.data_sha256[:8]}"
    out = root / "models" / "registry" / version
    out.mkdir(parents=True, exist_ok=True)

    rows, entries = [], {}
    for i, sid in enumerate(panel.series):
        sel = selection.iloc[i]
        point = preds[sel["model"]][i]
        rec = {"series": sid, "store": panel.stores[i], "category": panel.categories[i], "model": sel["model"],
               "date": fdates.strftime("%Y-%m-%d"), "h": np.arange(1, H + 1), "forecast": point}
        offsets = {}
        for lv in levels:
            tag = int(lv * 100)
            offsets[str(lv)] = [float(sel[f"q_lo_{tag}"]), float(sel[f"q_hi_{tag}"])]
            rec[f"lo_{tag}"], rec[f"hi_{tag}"] = apply_offsets(point, offsets[str(lv)])
        rows.append(pd.DataFrame(rec))
        wape = {c[5:]: float(sel[c]) for c in selection.columns if c.startswith("wape_")}
        entries[sid] = {"store": panel.stores[i], "category": panel.categories[i], "model": sel["model"],
                        "demand_class": sel["demand_class"], "backtest_wape": wape,
                        "params": models[sel["model"]].describe(i), "interval_offsets": offsets}
    fc = pd.concat(rows, ignore_index=True)
    fc.round(3).to_csv(out / "forecasts.csv", index=False)

    tot_rows = []
    offs_tot = summary["hierarchy"]["store_total_offsets"]
    for st, g in fc.groupby("store", sort=True):
        point = g.groupby("date", sort=True)["forecast"].sum().to_numpy()
        rec = {"store": st, "date": fdates.strftime("%Y-%m-%d"), "h": np.arange(1, H + 1), "forecast": point}
        for lv in levels:
            tag = int(lv * 100)
            rec[f"lo_{tag}"], rec[f"hi_{tag}"] = apply_offsets(point, offs_tot[st][str(lv)])
        tot_rows.append(pd.DataFrame(rec))
    pd.concat(tot_rows, ignore_index=True).round(3).to_csv(out / "store_totals.csv", index=False)

    bm = pd.read_csv(reports / "backtest_metrics.csv")
    bm.to_csv(out / "backtest_metrics.csv", index=False)
    if "gbm" in models:
        joblib.dump(models["gbm"].model_, out / "gbm.joblib")

    meta = {
        "version": version, "label": "SYNTHETIC data",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": panel.data_sha256, "config_sha256": _config_sha(cfg),
        "train_start": str(panel.dates[0].date()), "train_end": str(panel.dates[-1].date()),
        "forecast_start": str(fdates[0].date()), "forecast_end": str(fdates[-1].date()), "max_horizon": H,
        "interval_levels": levels, "n_series": len(panel.series),
        "model_counts": {k: int(v) for k, v in selection["model"].value_counts().items()},
        "backtest": {"cutoffs": summary["cutoffs"], "scored_folds": summary["scored_folds"],
                     "selected_wape": summary["overall_scored_folds"]["selected"]["wape"],
                     "interval_coverage": summary["interval_coverage"]["selected"]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "registry.json").write_text(json.dumps(entries, indent=2))
    (root / "models" / "registry" / "LATEST").write_text(version + "\n")
    log.info("registered %s (%d series)", version, len(entries))
    return out


class Registry:
    """Read side of the registry, used by the API and the `forecast` command."""

    def __init__(self, registry_root: Path | None = None, version: str | None = None):
        base = Path(registry_root or config.REGISTRY)
        version = version or (base / "LATEST").read_text().strip()
        self.dir = base / version
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.entries = json.loads((self.dir / "registry.json").read_text())
        self.forecasts = pd.read_csv(self.dir / "forecasts.csv")
        self.totals = pd.read_csv(self.dir / "store_totals.csv")
        self.backtests = pd.read_csv(self.dir / "backtest_metrics.csv")
        self.levels = [int(lv * 100) for lv in self.meta["interval_levels"]]
        self.stores = sorted({e["store"] for e in self.entries.values()})
        self.categories = sorted({e["category"] for e in self.entries.values()})

    def _rows(self, df: pd.DataFrame, horizon: int) -> list[dict]:
        out = []
        for r in df[df["h"] <= horizon].itertuples(index=False):
            d = {"date": r.date, "h": int(r.h), "forecast": round(float(r.forecast), 2)}
            for lv in self.levels:
                d[f"lo_{lv}"] = round(float(getattr(r, f"lo_{lv}")), 2)
                d[f"hi_{lv}"] = round(float(getattr(r, f"hi_{lv}")), 2)
            out.append(d)
        return out

    def forecast(self, store: str, category: str | None, horizon: int) -> dict:
        if category is None:
            df = self.totals[self.totals["store"] == store]
            return {"store": store, "category": None, "level": "store_total", "method": "bottom_up",
                    "unit": "units/day", "forecast": self._rows(df, horizon)}
        sid = f"{store}-{category}"
        e = self.entries[sid]
        df = self.forecasts[self.forecasts["series"] == sid]
        return {"series": sid, "store": store, "category": category, "level": "series", "model": e["model"],
                "unit": "units/day", "forecast": self._rows(df, horizon)}

    def backtest(self, sid: str) -> dict:
        e = self.entries[sid]
        bt = self.backtests[self.backtests["series"] == sid]
        folds = {}
        for m, g in bt.groupby("model"):
            folds[m] = [{"cutoff": r.cutoff, "wape": round(r.wape, 4), "mase": round(r.mase, 4),
                         "bias": round(r.bias, 4), "mae": round(r.mae, 3)} for r in g.itertuples(index=False)]
        return {"series": sid, "chosen_model": e["model"], "demand_class": e["demand_class"],
                "mean_wape_by_model": e["backtest_wape"], "interval_offsets": e["interval_offsets"], "folds": folds}
