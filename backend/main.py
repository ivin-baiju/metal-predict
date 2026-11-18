"""FastAPI application entry point serving both the Metal Predict ML REST API and editorial static assets."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.cache import get_suite, reload_suite_cache
from backend.routers import forecast, pulse, simulate
from scripts.nightly_refresh import get_refresh_status, run_nightly_market_refresh

STATIC_ROOT_DIR = Path(__file__).resolve().parent.parent / "kigo.studio"
PREDICTOR_HTML = STATIC_ROOT_DIR / "tools" / "metal-price-predictor.html"

app = FastAPI(
    title="Metal Predict — Multi-Horizon Commodity Price Forecasting API",
    version="2.2.0",
    description="Three-Tier Quantile ML Ensemble (XGBoost + LightGBM/Basis + VECM/Monte Carlo) with TreeSHAP, FinBERT News Pulse, and COMEX-MCX Parity.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(simulate.router)
app.include_router(pulse.router)


@app.get("/api/health")
def health_check() -> dict:
    suite = get_suite()
    return {
        "status": "ok",
        "engine": "Metal Predict Multi-Horizon Hybrid Ensemble",
        "metals_loaded": list(suite.latest_rows.keys()),
        "refresh_status": get_refresh_status(),
    }


@app.get("/api/refresh/status")
def refresh_status_endpoint() -> dict:
    """Return latest nightly market data & model refresh telemetry."""
    return get_refresh_status()


@app.post("/api/refresh/trigger")
def trigger_refresh_endpoint(retrain: bool = False) -> dict:
    """
    Trigger a safe, zero-downtime market data & model cache refresh.
    - When `retrain=True`, runs full DuckDB ingestion + 3-Tier Walk-Forward retraining in a staging directory before swapping.
    - When `retrain=False` (default fast reload), hot-reloads the latest `multi_horizon_suite.joblib` from disk into memory.
    """
    if retrain:
        status = run_nightly_market_refresh()
    else:
        status = get_refresh_status()
    reload_suite_cache()
    return {
        "reloaded_in_memory": True,
        "retrained_models": retrain,
        **status,
    }


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
@app.get("/about-us.html", include_in_schema=False)
@app.get("/blog.html", include_in_schema=False)
@app.get("/contact.html", include_in_schema=False)
def serve_root_predictor() -> FileResponse:
    return FileResponse(str(PREDICTOR_HTML), media_type="text/html")


if STATIC_ROOT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_ROOT_DIR), html=True), name="metal_predict_static")
