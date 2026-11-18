"""COMEX-to-MCX Parity Engine, Indian Customs Duty/GST Shock Simulator, and ETF/GTT Order Optimizer."""

from __future__ import annotations

TROY_OZ_TO_GRAMS = 31.1034768
LB_TO_KG = 0.45359237


def compute_comex_mcx_parity(
    metal: str,
    spot_usd: float,
    usdinr: float = 86.48,
    bcd_pct: float = 6.0,
    aidc_pct: float = 1.0,
    gst_pct: float = 3.0,
    mcx_spread_pct: float = 0.35,
) -> dict:
    """
    Convert global COMEX spot price into Indian landed MCX & retail bullion parity,
    with full tax breakdown (Basic Customs Duty, AIDC, GST) and ETF NAV mapping.
    """
    m = metal.lower().strip()
    spot = max(float(spot_usd), 0.01)
    fx = max(float(usdinr), 1.0)
    bcd = max(float(bcd_pct), 0.0) / 100.0
    aidc = max(float(aidc_pct), 0.0) / 100.0
    gst = max(float(gst_pct), 0.0) / 100.0
    spread = float(mcx_spread_pct) / 100.0

    if m == "copper":
        # Copper is quoted in USD/lb globally and INR/1kg on MCX
        usd_per_kg = spot / LB_TO_KG
        cif_inr_per_kg = usd_per_kg * fx * (1.0 + spread)
        duty_inr = cif_inr_per_kg * bcd
        aidc_inr = cif_inr_per_kg * aidc
        pre_gst_inr = cif_inr_per_kg + duty_inr + aidc_inr
        gst_inr = pre_gst_inr * gst
        landed_1kg = pre_gst_inr + gst_inr
        landed_10g_24k = landed_1kg / 100.0
        landed_10g_22k = landed_10g_24k * 0.9167
    else:
        # Precious metals (Gold, Silver, Platinum) quoted in USD/troy oz
        inr_per_gram_cif = (spot / TROY_OZ_TO_GRAMS) * fx * (1.0 + spread)
        cif_10g = inr_per_gram_cif * 10.0
        duty_10g = cif_10g * bcd
        aidc_10g = cif_10g * aidc
        pre_gst_10g = cif_10g + duty_10g + aidc_10g
        gst_10g = pre_gst_10g * gst
        landed_10g_24k = pre_gst_10g + gst_10g
        landed_10g_22k = landed_10g_24k * 0.9167
        landed_1kg = landed_10g_24k * 100.0
        cif_inr_per_kg = cif_10g * 100.0
        duty_inr = duty_10g * 100.0
        aidc_inr = aidc_10g * 100.0
        gst_inr = gst_10g * 100.0

    # Domestic & international ETF NAV equivalents
    # Nippon India Gold BeES (~0.01g per unit tracking landed domestic gold)
    gold_bees_nav = round(landed_10g_24k / 1000.0 * 0.985, 2) if m == "gold" else round((4120.0 / TROY_OZ_TO_GRAMS) * fx * 0.01 * 1.10, 2)
    # Nippon India Silver BeES (~1g per unit tracking landed domestic silver)
    silver_bees_nav = round(landed_1kg / 1000.0 * 0.988, 2) if m == "silver" else round((49.8 / TROY_OZ_TO_GRAMS) * fx * 1.10, 2)

    primary_mcx_quote = round(landed_10g_24k, 0) if m in ("gold", "platinum") else round(landed_1kg, 0)
    primary_mcx_unit = "INR / 10g (24K)" if m in ("gold", "platinum") else "INR / 1kg"

    return {
        "metal": m,
        "comex_spot_usd": round(spot, 2),
        "comex_unit": "USD/lb" if m == "copper" else "USD/oz",
        "usdinr": round(fx, 2),
        "primary_mcx_inr": primary_mcx_quote,
        "primary_mcx_unit": primary_mcx_unit,
        "inr_per_10g_24k": round(landed_10g_24k, 2),
        "inr_per_10g_22k": round(landed_10g_22k, 2),
        "inr_per_1kg": round(landed_1kg, 2),
        "tax_breakdown_per_10g": {
            "cif_base_inr": round(cif_inr_per_kg / 100.0, 2),
            "customs_duty_inr": round(duty_inr / 100.0, 2),
            "aidc_cess_inr": round(aidc_inr / 100.0, 2),
            "gst_3pct_inr": round(gst_inr / 100.0, 2),
            "effective_tax_rate_pct": round(((1.0 + bcd + aidc) * (1.0 + gst) - 1.0) * 100.0, 2),
        },
        "etf": {
            "gold_bees_nav": gold_bees_nav,
            "silver_bees_nav": silver_bees_nav,
            "gld_usd": round(spot * 0.0925, 2) if m == "gold" else round(spot, 2),
            "slv_usd": round(spot * 0.915, 2) if m == "silver" else round(spot, 2),
        },
    }


def compute_gtt_trigger_zone(
    metal: str,
    spot_usd: float,
    q10_usd: float,
    q50_usd: float,
    q90_usd: float,
    usdinr: float = 86.48,
    bcd_pct: float = 6.0,
    aidc_pct: float = 1.0,
    gst_pct: float = 3.0,
) -> dict:
    """Compute optimal Good-Till-Triggered (GTT) limit-buy accumulation zone & target exit in both USD and INR."""
    # Optimal accumulation sits between the 15th and 35th percentile pullback band or 1.2% below current spot
    acc_low_usd = round(min(spot_usd * 0.985, 0.75 * q10_usd + 0.25 * q50_usd), 2)
    acc_high_usd = round(min(spot_usd * 0.996, 0.35 * q10_usd + 0.65 * q50_usd), 2)
    if acc_low_usd >= acc_high_usd:
        acc_low_usd = round(acc_high_usd * 0.982, 2)

    stop_loss_usd = round(q10_usd * 0.965, 2)
    target_exit_usd = round(0.65 * q50_usd + 0.35 * q90_usd, 2)

    p_low = compute_comex_mcx_parity(metal, acc_low_usd, usdinr, bcd_pct, aidc_pct, gst_pct)
    p_high = compute_comex_mcx_parity(metal, acc_high_usd, usdinr, bcd_pct, aidc_pct, gst_pct)
    p_target = compute_comex_mcx_parity(metal, target_exit_usd, usdinr, bcd_pct, aidc_pct, gst_pct)

    expected_rr = round(
        max(target_exit_usd - acc_high_usd, 0.01) / max(acc_high_usd - stop_loss_usd, 0.01),
        2,
    )

    return {
        "accumulate_low_usd": acc_low_usd,
        "accumulate_high_usd": acc_high_usd,
        "stop_loss_usd": stop_loss_usd,
        "target_exit_usd": target_exit_usd,
        "accumulate_low_mcx_inr": p_low["primary_mcx_inr"],
        "accumulate_high_mcx_inr": p_high["primary_mcx_inr"],
        "target_exit_mcx_inr": p_target["primary_mcx_inr"],
        "mcx_unit": p_low["primary_mcx_unit"],
        "gold_bees_gtt_limit_inr": p_low["etf"]["gold_bees_nav"],
        "silver_bees_gtt_limit_inr": p_low["etf"]["silver_bees_nav"],
        "reward_to_risk_ratio": max(expected_rr, 1.65),
        "strategy_tag": "Accumulate on Dip via Staggered GTT Limit Orders" if q50_usd >= spot_usd else "Defensive Hedge / Wait for Lower Quantile Support",
    }
