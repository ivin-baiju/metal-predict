"""Forecast endpoints: /api/predict/horizons and /api/predict/date."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.cache import (
    get_all_metal_summaries,
    get_cached_horizons,
    get_suite,
    resolve_metal,
)
from data_pipeline.ingest_market_data import METAL_SPECS
from ml_engine.explainability import (
    build_shap_audit_narrative,
    compute_shap_drivers,
    predict_specific_date,
)
from ml_engine.parity_engine import compute_comex_mcx_parity, compute_gtt_trigger_zone

router = APIRouter(prefix="/api/predict", tags=["forecast"])


@router.get("/horizons")
def predict_horizons(
    metal: str = Query("gold", description="Metal name or symbol (gold/XAU, silver/XAG, platinum/XPT, copper/HG)"),
    usdinr: float = Query(86.48, ge=40.0, le=150.0),
    bcd_pct: float = Query(6.0, ge=0.0, le=40.0),
    aidc_pct: float = Query(1.0, ge=0.0, le=20.0),
    gst_pct: float = Query(3.0, ge=0.0, le=28.0),
) -> dict:
    m = resolve_metal(metal)
    suite = get_suite()
    spec = METAL_SPECS[m]
    horizons = get_cached_horizons(m)
    hist = suite.history_tail[m]

    spot = float(hist[-1]["close"])
    prev = float(hist[-2]["close"]) if len(hist) > 1 else spot
    chg_24h_pct = round(((spot - prev) / max(prev, 1e-6)) * 100.0, 2)

    parity = compute_comex_mcx_parity(
        m,
        spot_usd=spot,
        usdinr=usdinr,
        bcd_pct=bcd_pct,
        aidc_pct=aidc_pct,
        gst_pct=gst_pct,
    )

    shap_by_horizon: dict[str, list[dict]] = {}
    narratives_by_horizon: dict[str, str] = {}
    for h_key in ["1M", "3M", "6M", "1Y", "5Y"]:
        h_obj = horizons[h_key]
        term = h_obj["terminal"]
        drivers = compute_shap_drivers(suite, m, h_key)
        shap_by_horizon[h_key] = drivers
        narratives_by_horizon[h_key] = build_shap_audit_narrative(
            metal=m,
            tier_name=h_obj["tier"],
            delta_days=h_obj["days"],
            spot_usd=spot,
            q50_usd=float(term["q50"]),
            drivers=drivers,
            catalysts=term.get("catalysts", {"active_catalysts": ["Multi-Horizon Structural Equilibrium"]}),
        )

    term_1m = horizons["1M"]["terminal"]
    gtt = compute_gtt_trigger_zone(
        m,
        spot_usd=spot,
        q10_usd=float(term_1m["q10"]),
        q50_usd=float(term_1m["q50"]),
        q90_usd=float(term_1m["q90"]),
        usdinr=usdinr,
        bcd_pct=bcd_pct,
        aidc_pct=aidc_pct,
        gst_pct=gst_pct,
    )

    return {
        "metal": m,
        "spec": spec,
        "all_metals": get_all_metal_summaries(),
        "latest_date": str(hist[-1]["date"]),
        "spot_usd": round(spot, 2),
        "change_24h_pct": chg_24h_pct,
        "parity": parity,
        "gtt_zone": gtt,
        "horizons": horizons,
        "shap_drivers": shap_by_horizon["1M"],
        "shap_by_horizon": shap_by_horizon,
        "narratives_by_horizon": narratives_by_horizon,
        "history": hist,
    }


@router.get("/date")
def predict_date_endpoint(
    metal: str = Query("gold"),
    target_date: str = Query(..., description="Target future calendar date YYYY-MM-DD"),
    usdinr: float = Query(86.48, ge=40.0, le=150.0),
    bcd_pct: float = Query(6.0, ge=0.0, le=40.0),
    aidc_pct: float = Query(1.0, ge=0.0, le=20.0),
    gst_pct: float = Query(3.0, ge=0.0, le=28.0),
) -> dict:
    m = resolve_metal(metal)
    suite = get_suite()
    try:
        return predict_specific_date(
            suite=suite,
            metal=m,
            target_date_str=target_date,
            usdinr=usdinr,
            bcd_pct=bcd_pct,
            aidc_pct=aidc_pct,
            gst_pct=gst_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
