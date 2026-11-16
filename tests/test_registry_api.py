import json

import numpy as np
import pandas as pd
import pytest

from salesforecast import cli, config
from salesforecast.api import MAX_BATCH, MAX_HORIZON, create_app
from salesforecast.generate import file_sha256
from salesforecast.registry import train
from tests.asgi import call


@pytest.fixture(scope="module")
def app(project):
    return create_app(project["root"] / "models" / "registry")


def test_registry_version_and_metadata(project):
    root, vdir = project["root"], project["version_dir"]
    meta = json.loads((vdir / "metadata.json").read_text())
    assert (root / "models/registry/LATEST").read_text().strip() == meta["version"] == vdir.name
    assert meta["data_sha256"] == file_sha256(root / "data/raw/sales.csv")
    assert meta["train_end"] == str(project["panel"].dates[-1].date())
    assert meta["forecast_start"] == str((project["panel"].dates[-1] + pd.Timedelta(days=1)).date())
    for f in ("registry.json", "forecasts.csv", "store_totals.csv", "backtest_metrics.csv"):
        assert (vdir / f).exists()
    entries = json.loads((vdir / "registry.json").read_text())
    assert len(entries) == len(project["panel"].series)
    assert all(e["model"] == min(e["backtest_wape"], key=e["backtest_wape"].get) for e in entries.values())


def test_train_refuses_when_data_changed_since_the_backtest(project, cfg, tmp_path):
    import shutil

    shutil.copytree(project["root"] / "data", tmp_path / "data")
    shutil.copytree(project["root"] / "reports", tmp_path / "reports")
    with open(tmp_path / "data/raw/sales.csv", "a") as fh:
        fh.write("2099-01-01,S01,snacks,1\n")
    with pytest.raises(RuntimeError, match="changed since the backtest"):
        train(cfg, tmp_path)


def test_health_metadata_and_series(app, project):
    status, body = call(app, "GET", "/health")
    assert status == 200 and body["status"] == "ok" and body["series"] == len(project["panel"].series)
    status, meta = call(app, "GET", "/metadata")
    assert meta["label"] == "SYNTHETIC data" and meta["max_horizon"] == MAX_HORIZON
    status, s = call(app, "GET", "/series", {"store": "S01"})
    assert status == 200 and s["count"] == 8 and all(x["store"] == "S01" for x in s["series"])


def test_forecast_get_returns_ordered_intervals(app, project):
    status, body = call(app, "GET", "/forecast", {"store": "S02", "category": "snacks", "horizon": 10})
    assert status == 200 and body["series"] == "S02-snacks" and len(body["forecast"]) == 10
    days = pd.to_datetime([r["date"] for r in body["forecast"]])
    assert days[0] == project["panel"].dates[-1] + pd.Timedelta(days=1) and (np.diff(days).astype(int) > 0).all()
    for r in body["forecast"]:
        assert 0 <= r["lo_95"] <= r["lo_80"] <= r["forecast"] <= r["hi_80"] <= r["hi_95"]


def test_store_total_is_the_sum_of_its_categories(app):
    _, total = call(app, "GET", "/forecast", {"store": "S01", "horizon": 7})
    assert total["level"] == "store_total" and total["method"] == "bottom_up"
    parts = np.zeros(7)
    for cat in config.load_config()["categories"]:
        _, b = call(app, "GET", "/forecast", {"store": "S01", "category": cat, "horizon": 7})
        parts += [r["forecast"] for r in b["forecast"]]
    assert np.allclose([r["forecast"] for r in total["forecast"]], parts, atol=0.05)


def test_forecast_post_batch(app):
    body = {"requests": [{"store": "S01", "category": "dairy", "horizon": 3}, {"store": "S03", "horizon": 5}]}
    status, r = call(app, "POST", "/forecast", body=body)
    assert status == 200 and r["count"] == 2
    assert len(r["results"][0]["forecast"]) == 3 and r["results"][1]["level"] == "store_total"


def test_forecast_validation_errors(app):
    for query, field in [({"store": "S01", "horizon": 0}, "horizon"), ({"store": "S01", "horizon": MAX_HORIZON + 1},
                         "horizon"), ({"store": "S01", "colour": "red"}, "colour"), ({"category": "snacks"}, "store")]:
        status, body = call(app, "GET", "/forecast", query)
        assert status == 422 and body["error"]["code"] == "validation_error"
        assert any(e["loc"] == field for e in body["error"]["errors"]), query


def test_errors_unknown_series_bad_json_and_batch_limits(app):
    assert call(app, "GET", "/forecast", {"store": "S99"})[0] == 404
    assert call(app, "GET", "/forecast", {"store": "S01", "category": "toys"})[0] == 404
    assert call(app, "GET", "/backtest/S01-toys")[0] == 404
    status, body = call(app, "POST", "/forecast", raw=b"{nope")
    assert status == 400 and body["error"]["code"] == "malformed_json"
    assert call(app, "POST", "/forecast", body={"requests": []})[0] == 422
    big = {"requests": [{"store": "S01"}] * (MAX_BATCH + 1)}
    assert call(app, "POST", "/forecast", body=big)[0] == 422


def test_backtest_endpoint(app, project):
    status, body = call(app, "GET", "/backtest/S01-grocery")
    assert status == 200 and body["chosen_model"] in body["mean_wape_by_model"]
    assert len(body["folds"]["gbm"]) == len(project["res"].cutoffs)
    assert len(body["folds"]["selected"]) == len(project["res"].cutoffs) - 2


def test_with_starlette_testclient_when_httpx_is_installed(project):
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    client = TestClient(create_app(project["root"] / "models" / "registry"))
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/forecast", params={"store": "S01", "category": "snacks", "horizon": 3})
    assert r.status_code == 200 and len(r.json()["forecast"]) == 3


def test_cli_forecast_prints_from_the_registry(project, monkeypatch, capsys):
    monkeypatch.setattr(config, "REGISTRY", project["root"] / "models" / "registry")
    cli.main(["forecast", "--store", "S01", "--category", "beverages", "--horizon", "3"])
    out = capsys.readouterr().out
    assert "S01-beverages" in out and "SYNTHETIC" in out and len(out.strip().splitlines()) == 5
