"""FastAPI application entry point serving both the Metal Predict ML REST API and editorial static assets."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.cache import get_suite
from backend.routers import forecast, pulse, simulate

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
