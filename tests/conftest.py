"""Shared fixtures: one small end-to-end run (generate → backtest → train) in a temporary project root."""
import pytest

from salesforecast import config
from salesforecast.backtest import run_backtest, write_backtest
from salesforecast.generate import generate
from salesforecast.prep import load_panel
from salesforecast.registry import train


@pytest.fixture(scope="session")
def cfg():
    return config.small_config()            # 3 stores × 8 categories, 2 years, 4 cutoffs; same code paths


@pytest.fixture(scope="session")
def project(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("project")
    gen = generate(cfg, root)
    panel = load_panel(cfg, root, write=True)
    res = run_backtest(cfg, panel)
    write_backtest(res, panel, root)
    version_dir = train(cfg, root)
    return {"root": root, "gen": gen, "panel": panel, "res": res, "version_dir": version_dir}
