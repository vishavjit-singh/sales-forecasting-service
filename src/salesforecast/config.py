"""Paths and settings shared by every command."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REGISTRY = ROOT / "models" / "registry"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "docs" / "images"


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())


def small_config(cfg: dict | None = None, n_stores: int = 3, n_cutoffs: int = 4) -> dict:
    """A smaller copy of the config with the same code paths (tests and quick runs)."""
    c = copy.deepcopy(cfg or load_config())
    c["data"]["n_stores"] = n_stores
    c["data"]["structural_break"]["store"] = f"S{n_stores:02d}"
    c["data"]["start"] = "2024-01-01"
    c["backtest"]["n_cutoffs"] = n_cutoffs
    c["models"]["gbm"]["max_iter"] = 80
    c["models"]["ets"]["fit_window"] = 364
    return c
