"""Interactive Macro Stress-Tester ('What-If' Sliders) & Duty/Tax Shock Simulator endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.cache import get_cached_horizons, get_suite, resolve_metal
from data_pipeline.ingest_market_data import METAL_SPECS
from ml_engine.models.short_term_xgb import _enforce_quantile_order
from ml_engine.parity_engine import compute_comex_mcx_parity, compute_gtt_trigger_zone

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


class MacroShockRequest(BaseModel):
    metal: str = Field(default="gold")
    horizon: str = Field(default="1Y")
    usdinr_rate: float = Field(default=86.48, ge=50.0, le=140.0)
    us_cpi_pct: float = Field(default=2.7, ge=-2.0, le=15.0)
    rate_cuts_bps: float = Field(default=50.0, ge=-300.0, le=400.0)
    real_yield_delta_bps: float = Field(default=0.0, ge=-300.0, le=300.0)
    customs_duty_pct: float = Field(default=6.0, ge=0.0, le=35.0)
    aidc_pct: float = Field(default=1.0, ge=0.0, le=15.0)
    gst_pct: float = Field(default=3.0, ge=0.0, le=28.0)


@router.post("/macro-shock")
def simulate_macro_shock(req: MacroShockRequest) -> dict:
    """
    Dynamically perturb the macro input vector (USD/INR, US CPI, Fed/RBI Rate Cuts, Real Yields,
    and Indian Customs Duty/GST) and recompute the forecast curve while guaranteeing quantile monotonicity.
    """
    m = resolve_metal(req.metal)
    spec = METAL_SPECS[m]
    suite = get_suite()
    horizons = get_cached_horizons(m)
    h_key = req.horizon.upper() if req.horizon.upper() in horizons else "1Y"
    base_h = horizons[h_key]
    base_series = base_h["series"]
    spot = float(suite.latest_rows[m]["close"])

    # Baseline assumptions: USD/INR = 86.48, US CPI = 2.7%, Rate Cuts = 50 bps, Real Yield delta = 0 bps
    fx_shock = (req.usdinr_rate - 86.48) / 86.48
    cpi_shock = (req.us_cpi_pct - 2.7) / 100.0
    cuts_shock = (req.rate_cuts_bps - 50.0) / 10000.0
    ry_shock = req.real_yield_delta_bps / 10000.0

    # Elasticities by metal class
    cpi_elast = 1.45 if m in ("gold", "silver") else 1.10
    cuts_elast = 1.65 if m == "silver" else (1.40 if m == "gold" else 1.15)
    ry_elast = spec["real_yield_beta"] * 4.2  # negative beta: lower real yields lift metal price
    fx_global_pass = 0.22 * fx_shock

    total_annual_impulse = (
        cpi_elast * cpi_shock
        + cuts_elast * cuts_shock
        + ry_elast * ry_shock
        + fx_global_pass
    )
    total_annual_impulse = float(np.clip(total_annual_impulse, -0.45, 0.65))

    # Volatility cone multiplier increases under large macro shocks
    shock_mag = abs(cpi_shock) * 4.0 + abs(cuts_shock) * 3.0 + abs(ry_shock) * 4.0
    vol_stretch = float(np.clip(1.0 + 0.35 * shock_mag, 1.0, 1.45))

    stressed_series: list[dict] = []
    for pt in base_series:
        d = int(pt["days_ahead"])
        time_weight = np.power(min(d, 1825) / 365.25, 0.72)
        mult = float(np.exp(total_annual_impulse * time_weight))

        base_q50 = float(pt["q50"])
        new_q50 = base_q50 * mult
        low80 = abs(base_q50 - float(pt["q10"])) * mult * vol_stretch
        high80 = abs(float(pt["q90"]) - base_q50) * mult * vol_stretch
        low95 = abs(base_q50 - float(pt["q025"])) * mult * vol_stretch
        high95 = abs(float(pt["q975"]) - base_q50) * mult * vol_stretch

        q025, q10, q50, q90, q975 = _enforce_quantile_order(
            new_q50 - low95,
            new_q50 - low80,
            new_q50,
            new_q50 + high80,
            new_q50 + high95,
        )
        stressed_series.append(
            {
                "date": pt["date"],
                "days_ahead": d,
                "baseline_q50": round(base_q50, 2),
                "q025": round(q025, 2),
                "q10": round(q10, 2),
                "q50": round(q50, 2),
                "q90": round(q90, 2),
                "q975": round(q975, 2),
            }
        )

    stressed_term = stressed_series[-1]
    base_term_q50 = float(base_series[-1]["q50"])
    delta_vs_baseline_pct = round(
        ((stressed_term["q50"] - base_term_q50) / max(base_term_q50, 1e-6)) * 100.0,
        2,
    )

    # Compute current & projected landed MCX parity under the user's customs duty, AIDC, GST, and USD/INR sliders
    spot_stressed_parity = compute_comex_mcx_parity(
        m,
        spot_usd=spot,
        usdinr=req.usdinr_rate,
        bcd_pct=req.customs_duty_pct,
        aidc_pct=req.aidc_pct,
        gst_pct=req.gst_pct,
    )
    target_stressed_parity = compute_comex_mcx_parity(
        m,
        spot_usd=stressed_term["q50"],
        usdinr=req.usdinr_rate,
        bcd_pct=req.customs_duty_pct,
        aidc_pct=req.aidc_pct,
        gst_pct=req.gst_pct,
    )
    gtt = compute_gtt_trigger_zone(
        m,
        spot_usd=spot,
        q10_usd=stressed_term["q10"],
        q50_usd=stressed_term["q50"],
        q90_usd=stressed_term["q90"],
        usdinr=req.usdinr_rate,
        bcd_pct=req.customs_duty_pct,
        aidc_pct=req.aidc_pct,
        gst_pct=req.gst_pct,
    )

    return {
        "metal": m,
        "horizon": h_key,
        "inputs": req.model_dump(),
        "annualized_macro_impulse_pct": round(total_annual_impulse * 100.0, 2),
        "baseline_terminal_q50": round(base_term_q50, 2),
        "stressed_terminal": stressed_term,
        "delta_vs_baseline_pct": delta_vs_baseline_pct,
        "stressed_series": stressed_series,
        "stressed_parity": spot_stressed_parity,
        "projected_target_parity": target_stressed_parity,
        "stressed_gtt_zone": gtt,
    }
