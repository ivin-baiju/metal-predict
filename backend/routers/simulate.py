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


class PortfolioSipRequest(BaseModel):
    monthly_sip: float = Field(default=25000.0, ge=0.0)
    lump_sum: float = Field(default=100000.0, ge=0.0)
    horizon_years: int = Field(default=3, ge=1, le=5)
    currency: str = Field(default="INR")
    gold_weight: float = Field(default=50.0, ge=0.0, le=100.0)
    silver_weight: float = Field(default=25.0, ge=0.0, le=100.0)
    platinum_weight: float = Field(default=10.0, ge=0.0, le=100.0)
    copper_weight: float = Field(default=15.0, ge=0.0, le=100.0)


@router.post("/portfolio-sip")
def simulate_portfolio_sip(req: PortfolioSipRequest) -> dict:
    """Compute multi-metal SIP + Lump-Sum wealth trajectory across Q10 (Conservative), Q50 (Median), and Q90 (Bull)."""
    weights = np.array(
        [req.gold_weight, req.silver_weight, req.platinum_weight, req.copper_weight],
        dtype=float,
    )
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        weights = np.array([0.5, 0.25, 0.10, 0.15])
    else:
        weights = weights / w_sum

    # Annualized expected drifts and volatilities from our 3-tier ensemble
    drifts = np.array([0.098, 0.124, 0.074, 0.092])
    vols = np.array([0.145, 0.255, 0.205, 0.215])

    port_drift = float(np.dot(weights, drifts))
    # Incorporate diversification benefit across precious & industrial metals
    port_vol = float(np.sqrt(np.dot(weights ** 2, vols ** 2) + 0.45 * np.sum(weights) * 0.012))

    months = int(req.horizon_years * 12)
    r_m_med = port_drift / 12.0
    r_m_bull = (port_drift + 1.28 * port_vol / np.sqrt(req.horizon_years)) / 12.0
    r_m_bear = max(-0.02, (port_drift - 1.15 * port_vol / np.sqrt(req.horizon_years)) / 12.0)

    total_invested = req.lump_sum + req.monthly_sip * months

    def _compound(rate_m: float) -> float:
        lump_fv = req.lump_sum * ((1.0 + rate_m) ** months)
        if abs(rate_m) < 1e-6:
            sip_fv = req.monthly_sip * months
        else:
            sip_fv = req.monthly_sip * (((1.0 + rate_m) ** months - 1.0) / rate_m) * (1.0 + rate_m)
        return float(lump_fv + sip_fv)

    val_q50 = _compound(r_m_med)
    val_q90 = _compound(r_m_bull)
    val_q10 = _compound(r_m_bear)
    sharpe = round((port_drift - 0.042) / max(port_vol, 0.05), 2)

    return {
        "currency": req.currency,
        "horizon_years": req.horizon_years,
        "months": months,
        "normalized_weights_pct": {
            "gold": round(float(weights[0]) * 100.0, 1),
            "silver": round(float(weights[1]) * 100.0, 1),
            "platinum": round(float(weights[2]) * 100.0, 1),
            "copper": round(float(weights[3]) * 100.0, 1),
        },
        "total_invested": round(total_invested, 0),
        "projected_median_q50": round(val_q50, 0),
        "projected_bull_q90": round(val_q90, 0),
        "projected_conservative_q10": round(val_q10, 0),
        "net_wealth_gain_q50": round(val_q50 - total_invested, 0),
        "expected_cagr_pct": round(port_drift * 100.0, 2),
        "portfolio_volatility_pct": round(port_vol * 100.0, 2),
        "sharpe_ratio": sharpe,
    }

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

    # Compute stressed macro regime probabilities
    stressed_row = suite.latest_rows[m].copy()
    stressed_row["real_yield_level"] = float(stressed_row.get("real_yield_level", 1.85)) + (req.real_yield_delta_bps / 100.0) - (req.rate_cuts_bps / 200.0)
    stressed_row["ret_63d"] = float(stressed_row.get("ret_63d", 0.03)) + total_annual_impulse * 0.35
    stressed_regimes = suite.mid_models[m].compute_regime_probabilities(stressed_row)

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
        "stressed_regime_probabilities": stressed_regimes,
    }
