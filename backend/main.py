"""FastAPI application entry point serving both the ML REST API and Kigo Studio static assets."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.cache import get_suite
from backend.routers import forecast, pulse, simulate

KIGO_STATIC_DIR = Path(__file__).resolve().parent.parent / "kigo.studio"

app = FastAPI(
    title="Kigo Metals — Multi-Horizon Metal Price Predictor API",
    version="2.0.0",
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
        "engine": "Kigo Metals Multi-Horizon Hybrid Ensemble",
        "metals_loaded": list(suite.latest_rows.keys()),
    }


if KIGO_STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(KIGO_STATIC_DIR), html=True), name="kigo_static")
