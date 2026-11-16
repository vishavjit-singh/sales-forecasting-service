"""Forecast API (Starlette + pydantic), served from the latest registry version.

FastAPI could not be installed in the environment this was built in, so the service uses Starlette, the ASGI
toolkit FastAPI is built on, with pydantic models for validation. Swapping FastAPI in only touches this file.

GET  /health                                     status and registry version
GET  /metadata                                   registry metadata (data hash, cutoff, backtest headline)
GET  /series[?store=S01]                         series with their chosen model and backtest WAPE
GET  /forecast?store=S01&category=snacks&horizon=14   point forecast + 80% / 95% intervals
GET  /forecast?store=S01&horizon=7               store total (bottom-up sum of its categories)
POST /forecast  {"requests": [{"store": "S01", "category": "snacks", "horizon": 7}, ...]}
GET  /backtest/{series}                          per-model, per-fold backtest scores for one series
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import config
from .registry import Registry

CFG = config.load_config()
MAX_HORIZON = CFG["api"]["max_horizon"]
MAX_BATCH = CFG["api"]["max_batch"]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store: str = Field(min_length=1, max_length=10)
    category: str | None = Field(default=None, max_length=40)
    horizon: int = Field(default=MAX_HORIZON, ge=1, le=MAX_HORIZON)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[ForecastRequest] = Field(min_length=1, max_length=MAX_BATCH)


class NotFound(Exception):
    pass


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status)


def _validation(exc: ValidationError) -> JSONResponse:
    errors = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return _error(422, "validation_error", "request validation failed", errors=errors)


def _forecast(reg: Registry, req: ForecastRequest) -> dict:
    if req.store not in reg.stores:
        raise NotFound(f"unknown store {req.store!r}")
    if req.category is not None and req.category not in reg.categories:
        raise NotFound(f"unknown category {req.category!r}")
    return reg.forecast(req.store, req.category, req.horizon)


async def health(request: Request):
    reg = request.app.state.registry
    return JSONResponse({"status": "ok", "version": reg.meta["version"], "series": len(reg.entries)})


async def metadata(request: Request):
    return JSONResponse(request.app.state.registry.meta)


async def series(request: Request):
    reg = request.app.state.registry
    store = request.query_params.get("store")
    items = [{"series": sid, "store": e["store"], "category": e["category"], "model": e["model"],
              "demand_class": e["demand_class"], "backtest_wape": e["backtest_wape"][e["model"]]}
             for sid, e in reg.entries.items() if store is None or e["store"] == store]
    return JSONResponse({"version": reg.meta["version"], "count": len(items), "series": items})


async def forecast_get(request: Request):
    reg = request.app.state.registry
    try:
        req = ForecastRequest.model_validate(dict(request.query_params))
        return JSONResponse({"version": reg.meta["version"], **_forecast(reg, req)})
    except ValidationError as exc:
        return _validation(exc)
    except NotFound as exc:
        return _error(404, "not_found", str(exc))


async def forecast_post(request: Request):
    reg = request.app.state.registry
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "malformed_json", "request body is not valid JSON")
    try:
        batch = BatchRequest.model_validate(body)
    except ValidationError as exc:
        return _validation(exc)
    results = []
    for i, req in enumerate(batch.requests):
        try:
            results.append(_forecast(reg, req))
        except NotFound as exc:
            return _error(404, "not_found", f"requests.{i}: {exc}")
    return JSONResponse({"version": reg.meta["version"], "count": len(results), "results": results})


async def backtest(request: Request):
    reg = request.app.state.registry
    sid = request.path_params["series"]
    if sid not in reg.entries:
        return _error(404, "not_found", f"unknown series {sid!r} (format: S01-snacks)")
    return JSONResponse(reg.backtest(sid))


def create_app(registry_root: str | Path | None = None) -> Starlette:
    app = Starlette(routes=[
        Route("/health", health), Route("/metadata", metadata), Route("/series", series),
        Route("/forecast", forecast_get, methods=["GET"]), Route("/forecast", forecast_post, methods=["POST"]),
        Route("/backtest/{series}", backtest),
    ])
    app.state.registry = Registry(Path(registry_root or os.environ.get("REGISTRY_DIR", config.REGISTRY)))
    return app


def __getattr__(name):            # `uvicorn salesforecast.api:app` builds the app lazily (registry must exist)
    if name == "app":
        return create_app()
    raise AttributeError(name)
