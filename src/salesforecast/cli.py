"""``python -m salesforecast generate|backtest|train|forecast|report|serve``"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from . import config

log = logging.getLogger("salesforecast")


def cmd_generate(cfg, args):
    from .generate import generate

    s = generate(cfg)
    print(f"SYNTHETIC data: {s['n_series']} series × {s['n_days']} days, {s['rows_written']:,} rows written, "
          f"{s['rows_missing']} rows missing on purpose, {s['outliers_injected']} outliers injected, "
          f"{len(s['intermittent_series'])} intermittent series")


def cmd_backtest(cfg, args):
    from .backtest import run_backtest, write_backtest
    from .prep import load_panel

    t0 = time.perf_counter()
    panel = load_panel(cfg, write=True)
    print(f"prep: {panel.meta['cells_imputed']} gaps filled, {panel.meta['cells_replaced']} outliers replaced")
    res = run_backtest(cfg, panel)
    out = write_backtest(res, panel)
    s = res.summary
    print(f"backtest: {len(s['cutoffs'])} cutoffs × {s['horizon']}-day horizon × {s['n_series']} series "
          f"in {time.perf_counter() - t0:.0f}s → {out}")
    print(f"{'model':<16}{'WAPE':>8}{'bias':>8}{'MASE':>8}{'cov80':>8}{'cov95':>8}"
          f"   (scored folds {s['scored_folds'][0]} … {s['scored_folds'][-1]})")
    for m, r in s["overall_scored_folds"].items():
        cov = s["interval_coverage"][m]
        print(f"{m:<16}{r['wape']:>8.3f}{r['bias']:>+8.3f}{r['mase_mean']:>8.3f}{cov['0.8']:>8.3f}{cov['0.95']:>8.3f}")
    print("selection share:", json.dumps(s["selection_share"]))


def cmd_train(cfg, args):
    from .registry import train

    out = train(cfg)
    meta = json.loads((out / "metadata.json").read_text())
    print(f"registered {meta['version']}: {meta['n_series']} series, models {json.dumps(meta['model_counts'])}, "
          f"forecasts {meta['forecast_start']} … {meta['forecast_end']}")


def cmd_forecast(cfg, args):
    from .registry import Registry

    reg = Registry()
    res = reg.forecast(args.store, args.category, args.horizon)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    label = res.get("series") or f"{args.store} total (bottom-up)"
    print(f"{label} · model: {res.get('model', 'sum of categories')} · registry {reg.meta['version']} · SYNTHETIC")
    print(f"{'date':<12}{'forecast':>10}{'80% interval':>20}{'95% interval':>20}")
    for r in res["forecast"]:
        print(f"{r['date']:<12}{r['forecast']:>10.1f}{r['lo_80']:>10.1f}–{r['hi_80']:<9.1f}"
              f"{r['lo_95']:>10.1f}–{r['hi_95']:<9.1f}")


def cmd_report(cfg, args):
    from .report import render_all

    for p in render_all(Path(args.out), cfg=cfg):
        print(f"wrote {p}")


def cmd_serve(cfg, args):
    import uvicorn

    uvicorn.run("salesforecast.api:app", host=args.host, port=args.port, log_level="info")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="salesforecast", description="Sales forecasting with backtesting (SYNTHETIC data)")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="write the synthetic sales, price/promotion plan and truth files")
    sub.add_parser("backtest", help="clean the data, run the rolling-origin backtest, select a model per series")
    sub.add_parser("train", help="fit the chosen models on all data and publish a registry version")
    f = sub.add_parser("forecast", help="print a forecast from the latest registry version")
    f.add_argument("--store", required=True)
    f.add_argument("--category", default=None, help="omit for the store total")
    f.add_argument("--horizon", type=int, default=7)
    f.add_argument("--json", action="store_true")
    r = sub.add_parser("report", help="render the README charts")
    r.add_argument("--out", default=str(config.IMAGES))
    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    cfg = config.load_config(args.config)
    {"generate": cmd_generate, "backtest": cmd_backtest, "train": cmd_train, "forecast": cmd_forecast,
     "report": cmd_report, "serve": cmd_serve}[args.cmd](cfg, args)
