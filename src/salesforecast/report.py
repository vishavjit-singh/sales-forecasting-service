"""README images, rendered from reports/ and data/ after a backtest (and the architecture diagram)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                     "#008300", "#6250d6")
# one fixed colour per model, everywhere
COLORS = {"gbm": BLUE, "ets": ORANGE, "ensemble": AQUA, "croston_sba": YELLOW, "moving_average": MAGENTA,
          "seasonal_naive": GREEN, "selected": VIOLET}
LABELS = {"gbm": "global GBM", "ets": "Holt-Winters (ETS)", "ensemble": "ensemble (ETS + GBM)",
          "croston_sba": "Croston / SBA", "moving_average": "28-day moving average",
          "seasonal_naive": "seasonal naive", "selected": "per-series selection"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _save(fig, out: Path, name: str, rect=(0, 0, 1, 1)) -> Path:
    fig.tight_layout(rect=rect)
    p = out / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def _suptitle(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, weight="bold", color=INK)


def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.5, color=color))

    ax.text(0.15, 5.85, "sales-forecasting-service: clean history → rolling-origin backtest → per-series model → "
            "registry → API", fontsize=13, weight="bold", color=INK)
    ax.text(0.2, 5.35, "python -m salesforecast generate | backtest | train | report | serve     (config.yaml "
            "drives every step · SYNTHETIC data)", fontsize=8.6, color=VIOLET)
    top = [("generate", "10 stores × 8 categories\n3 years daily + truth files", INK),
           ("prep", "gap fill · outlier check\n(causal: past data only)", INK),
           ("backtest", "6 cutoffs × 28 days\nexpanding window", VIOLET),
           ("select + intervals", "lowest WAPE per series\nresidual quantiles 80/95%", VIOLET),
           ("train", "chosen model per series\nrefit on all history", VIOLET)]
    for i, (t, s, col) in enumerate(top):
        box(0.2 + i * 2.55, 3.55, 2.2, 1.45, t, s, col)
        if i:
            arrow(0.2 + i * 2.55 - 0.35, 4.27, 0.2 + i * 2.55, 4.27)
    box(0.2, 0.55, 3.4, 1.8, "Models (one interface)",
        "seasonal naive · moving average\nHolt-Winters ETS(A,Ad,A) in NumPy\nCroston / SBA · global GBM (Poisson)\n"
        "ensemble (ETS + GBM)", BLUE)
    box(4.1, 0.55, 2.6, 1.8, "reports/", "metrics per series × model × fold\nselection.csv\nbacktest_summary.json",
        INK)
    box(7.2, 0.55, 2.7, 1.8, "models/registry/<version>",
        "registry.json · forecasts.csv\nstore_totals.csv · gbm.joblib\nmetadata (data SHA-256, cutoff)", GREEN)
    box(10.4, 0.55, 2.45, 1.8, "Forecast API", "Starlette + pydantic\n/forecast · /series\n/backtest/{series}",
        ORANGE)
    arrow(5.9, 3.55, 2.6, 2.35, BLUE)
    ax.text(2.0, 3.05, "fit / predict at each cutoff", fontsize=7.6, color=BLUE)
    arrow(7.3, 3.55, 5.6, 2.35)
    arrow(10.5, 3.55, 8.8, 2.35, GREEN)
    arrow(9.9, 1.45, 10.4, 1.45, ORANGE)
    ax.text(7.2, 0.25, "store totals are the bottom-up sum of their categories, with intervals from their own "
            "residuals",
            fontsize=7.6, color=MUTED)
    return _save(fig, out, "architecture.png")


def data_overview(out: Path, cfg: dict, root: Path) -> Path:
    clean = pd.read_csv(root / "data" / "prepared" / "sales_clean.csv", parse_dates=["date"])
    gen = json.loads((root / "data" / "truth" / "generator.json").read_text())
    total = clean.groupby("date")["units_clean"].sum()
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    a1 = fig.add_subplot(gs[0, :])
    a1.plot(total.index, total.values, color=GRID, lw=0.8, label="daily")
    a1.plot(total.index, total.rolling(7, center=True).mean(), color=BLUE, lw=1.8, label="7-day mean")
    cal = cfg["calendar"]["festival"]
    for i, p in enumerate(cal["peaks"]):
        pk = pd.Timestamp(p)
        if pk > total.index[-1]:
            continue
        a1.axvspan(pk - pd.Timedelta(days=cal["days_before"]), pk + pd.Timedelta(days=cal["days_after"]),
                   color=YELLOW, alpha=0.18, lw=0, label="festival window" if i == 0 else None)
    a1.set_ylabel("units / day, all series")
    a1.grid(axis="y", color=GRID)
    a1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    a1.set_title("Chain total: weekly cycle, a festival peak each autumn and a year-end sale")

    br = cfg["data"]["structural_break"]
    st = clean[clean["series"].str.startswith(br["store"] + "-")].groupby("date")["units_clean"].sum()
    a2 = fig.add_subplot(gs[1, 0])
    a2.plot(st.index, st.rolling(28, center=True).mean(), color=INK, lw=1.8)
    a2.axvline(pd.Timestamp(br["date"]), color=ORANGE, lw=1.4, ls="--")
    a2.text(pd.Timestamp(br["date"]), a2.get_ylim()[1] * 0.98, f"  break: ×{br['factor']} (competitor opens)",
            color=INK, fontsize=8.5, va="top")
    a2.set_ylabel("units / day, 28-day mean")
    a2.grid(axis="y", color=GRID)
    a2.set_title(f"Structural break: store {br['store']} total")

    sid = (gen["intermittent_series"] or [sorted(clean["series"].unique())[0]])[0]
    s = clean[clean["series"] == sid].set_index("date")["units_clean"].iloc[-180:]
    a3 = fig.add_subplot(gs[1, 1])
    a3.bar(s.index, s.values, width=0.9, color=BLUE)
    zero = (s == 0).mean()
    a3.set_title(f"Intermittent: {sid}, last 180 days, {zero:.0%} zero days")
    a3.set_ylabel("units / day")
    a3.grid(axis="y", color=GRID)
    for a in (a2, a3):
        a.tick_params(axis="x", labelsize=8)
    _suptitle(fig, f"SYNTHETIC sales: {gen['n_series']} series ({len(gen['stores'])} stores × "
              f"{len(gen['categories'])} categories), {gen['start']} to {gen['end']}")
    return _save(fig, out, "data_overview.png")


def model_comparison(out: Path, s: dict, sel: pd.DataFrame) -> Path:
    ov = s["overall_scored_folds"]
    order = sorted(ov, key=lambda m: ov[m]["wape"], reverse=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    vals = [ov[m]["wape"] for m in order]
    a1.barh([LABELS[m] for m in order], vals, color=[COLORS[m] for m in order], height=0.6)
    for i, (m, v) in enumerate(zip(order, vals)):
        a1.text(v + 0.003, i, f"{v:.3f}  (MASE {ov[m]['mase_mean']:.2f}, bias {ov[m]['bias']:+.1%})", va="center",
                fontsize=8.5, color=INK)
    a1.set_xlim(0, max(vals) * 1.55)
    a1.set_xlabel("WAPE over all series, lower is better")
    a1.grid(axis="x", color=GRID)
    a1.set_title(f"Backtest WAPE on {len(s['scored_folds'])} scored folds × {s['horizon']} days")

    classes = ["smooth", "erratic", "intermittent", "lumpy"]
    ct = sel.groupby(["model", "demand_class"]).size().unstack(fill_value=0).reindex(columns=classes, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    left = np.zeros(len(ct))
    shades = {"smooth": BLUE, "erratic": "#7fa9e0", "intermittent": YELLOW, "lumpy": "#f3c766"}
    for c in classes:
        a2.barh([LABELS[m] for m in ct.index], ct[c], left=left, color=shades[c], height=0.6, label=c,
                edgecolor="white", lw=1.5)
        left += ct[c].to_numpy()
    for i, v in enumerate(left):
        a2.text(v + 0.6, i, f"{int(v)}", va="center", fontsize=9, color=INK)
    a2.set_xlabel("series where the model has the lowest mean WAPE (all folds)")
    a2.legend(frameon=False, fontsize=8.5, title="demand class", title_fontsize=8.5, loc="lower right")
    a2.grid(axis="x", color=GRID)
    a2.set_title("Model-selection share by demand class")
    best = s["best_single_model"]
    _suptitle(fig, f"The global GBM has the lowest WAPE ({ov[best]['wape']:.3f}); per-series selection "
              f"({ov['selected']['wape']:.3f}) does not beat it")
    return _save(fig, out, "model_comparison.png")


def _pick_examples(s: dict, sel: pd.DataFrame, gen: dict, preds: pd.DataFrame) -> list[tuple[str, int, str]]:
    folds = sorted(preds["fold"].unique())
    starts = {k: pd.Timestamp(s["cutoffs"][k]) for k in folds}
    peaks = [pd.Timestamp(p) for p in s["festival_peaks"]]
    fest = [k for k in folds if any(starts[k] <= p < starts[k] + pd.Timedelta(days=s["horizon"]) for p in peaks)]
    snacks = sel[sel["category"] == "snacks"].sort_values("wape_gbm")
    inter = [x for x in gen["intermittent_series"] if x in set(sel["series"])]
    inter = inter or list(sel.sort_values("adi", ascending=False)["series"])      # sparsest series as fallback
    return [(snacks.iloc[0]["series"], fest[0] if fest else folds[-1], "festival peak inside the window"),
            (f"{s['break_store']['store']}-grocery", folds[-1], "store after its structural break"),
            (inter[0], folds[-1], "intermittent series")]


def forecast_vs_actual(out: Path, s: dict, sel: pd.DataFrame, root: Path) -> Path:
    preds = pd.read_csv(root / "reports" / "backtest_predictions.csv", parse_dates=["date"])
    preds = preds[preds["model"] == "selected"]
    clean = pd.read_csv(root / "data" / "prepared" / "sales_clean.csv", parse_dates=["date"])
    gen = json.loads((root / "data" / "truth" / "generator.json").read_text())
    examples = _pick_examples(s, sel, gen, preds)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (sid, fold, why) in zip(axes, examples):
        p = preds[(preds["series"] == sid) & (preds["fold"] == fold)]
        start = p["date"].min()
        hist = clean[(clean["series"] == sid) & (clean["date"] < start) &
                     (clean["date"] >= start - pd.Timedelta(days=42))]
        ax.plot(hist["date"], hist["units_clean"], color=MUTED, lw=1.2, label="history")
        ax.fill_between(p["date"], p["lo_95"], p["hi_95"], color=VIOLET, alpha=0.12, lw=0, label="95% interval")
        ax.fill_between(p["date"], p["lo_80"], p["hi_80"], color=VIOLET, alpha=0.22, lw=0, label="80% interval")
        ax.plot(p["date"], p["forecast"], color=VIOLET, lw=2, label="forecast")
        ax.plot(p["date"], p["actual"], color=INK, lw=0, marker="o", ms=3.2, label="actual")
        inside = ((p["actual"] >= p["lo_80"]) & (p["actual"] <= p["hi_80"])).mean()
        w = np.abs(p["actual"] - p["forecast"]).sum() / p["actual"].sum()
        ax.axvline(start, color=GRID, lw=1)
        ax.set_title(f"{sid}\n{why}", fontsize=10.5)
        used = p["model_used"].iloc[0]
        ax.text(0.02, 0.97, f"WAPE {w:.2f} · {inside:.0%} of days inside 80%\nmodel used: {LABELS.get(used, used)}",
                transform=ax.transAxes, fontsize=8, va="top", color=INK)
        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("units / day")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    _suptitle(fig, "Forecast vs actual on held-out backtest windows (per-series selection, intervals from earlier "
              "folds only)")
    return _save(fig, out, "forecast_vs_actual.png", rect=(0, 0.05, 1, 1))


def horizon_and_coverage(out: Path, s: dict) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    hz = s["wape_by_horizon"]
    days = np.arange(1, s["horizon"] + 1)
    for m in ["seasonal_naive", "moving_average", "croston_sba", "ets", "ensemble", "gbm"]:
        a1.plot(days, hz[m], color=COLORS[m], lw=2, label=LABELS[m])
    a1.set_xlabel("horizon day")
    a1.set_ylabel("WAPE (all series, scored folds)")
    a1.set_xticks([1, 7, 14, 21, 28])
    a1.grid(color=GRID)
    a1.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    a1.set_ylim(0.2, max(max(v) for v in hz.values()) * 1.18)
    a1.set_title("Error by horizon day: flat, because daily noise dominates")

    cov = s["interval_coverage"]
    models = ["seasonal_naive", "moving_average", "croston_sba", "ets", "ensemble", "gbm", "selected"]
    y = np.arange(len(models))
    for lv in ("0.8", "0.95"):
        a2.axvline(float(lv), color=MUTED, lw=1, ls="--")
        a2.text(float(lv), -0.85, f"nominal {float(lv):.0%}", fontsize=8, color=INK, ha="center")
    for yi, m in zip(y, models):
        c80, c95 = cov[m]["0.8"], cov[m]["0.95"]
        a2.plot([c80, 0.8], [yi, yi], color=COLORS[m], lw=1, alpha=0.5)
        a2.plot([c95, 0.95], [yi, yi], color=COLORS[m], lw=1, alpha=0.5)
        a2.scatter([c80], [yi], s=60, facecolor="white", edgecolor=COLORS[m], lw=2, zorder=3)
        a2.scatter([c95], [yi], s=60, color=COLORS[m], zorder=3)
        a2.text(1.005, yi, f"{c80:.1%} · {c95:.1%}", va="center", fontsize=8.5, color=INK)
    a2.scatter([], [], s=50, facecolor="white", edgecolor=MUTED, lw=2, label="80% interval")
    a2.scatter([], [], s=50, color=MUTED, label="95% interval")
    a2.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    a2.set_yticks(y, [LABELS[m] for m in models], fontsize=8.5)
    a2.set_xlim(0.7, 1.08)
    a2.set_ylim(-1.2, len(models) + 0.3)
    a2.set_xticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    a2.set_xlabel("share of actual days inside the interval")
    a2.grid(axis="x", color=GRID)
    a2.set_title("Interval coverage vs nominal (out of sample)")
    _suptitle(fig, "How error and uncertainty behave on the scored backtest folds")
    return _save(fig, out, "horizon_and_coverage.png")


def render_all(out: Path, root: Path | None = None, cfg: dict | None = None) -> list[Path]:
    root = Path(root or config.ROOT)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or config.load_config()
    s = json.loads((root / "reports" / "backtest_summary.json").read_text())
    s["festival_peaks"] = cfg["calendar"]["festival"]["peaks"]
    sel = pd.read_csv(root / "reports" / "selection.csv")
    return [architecture(out), data_overview(out, cfg, root), model_comparison(out, s, sel),
            forecast_vs_actual(out, s, sel, root), horizon_and_coverage(out, s)]
