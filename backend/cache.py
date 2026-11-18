"""In-memory model suite & pre-computed horizon forecast cache."""

from __future__ import annotations

from fastapi import HTTPException
from data_pipeline.ingest_market_data import METAL_SPECS
from ml_engine.train_walk_forward import MultiHorizonSuite, load_trained_suite

METAL_ALIASES: dict[str, str] = {
    "gold": "gold",
    "xau": "gold",
    "xauusd": "gold",
    "xau/usd": "gold",
    "gc=f": "gold",
    "silver": "silver",
    "xag": "silver",
    "xagusd": "silver",
    "xag/usd": "silver",
    "si=f": "silver",
    "platinum": "platinum",
    "xpt": "platinum",
    "xptusd": "platinum",
    "xpt/usd": "platinum",
    "pl=f": "platinum",
    "copper": "copper",
    "hg": "copper",
    "hgusd": "copper",
    "hg/usd": "copper",
    "hg=f": "copper",
}

_SUITE_CACHE: MultiHorizonSuite | None = None
_HORIZON_CACHE: dict[str, dict] = {}


def resolve_metal(metal: str) -> str:
    key = (metal or "").strip().lower()
    if key not in METAL_ALIASES:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported metal '{metal}'. Supported metals: gold (XAU), silver (XAG), platinum (XPT), copper (HG).",
        )
    return METAL_ALIASES[key]


def get_suite() -> MultiHorizonSuite:
    global _SUITE_CACHE
    if _SUITE_CACHE is None:
        _SUITE_CACHE = load_trained_suite()
    return _SUITE_CACHE


def get_cached_horizons(metal: str) -> dict:
    m = resolve_metal(metal)
    if m not in _HORIZON_CACHE:
        suite = get_suite()
        _HORIZON_CACHE[m] = suite.predict_all_horizons(m)
    return _HORIZON_CACHE[m]


def get_all_metal_summaries() -> list[dict]:
    suite = get_suite()
    summaries = []
    for m, spec in METAL_SPECS.items():
        hist = suite.history_tail[m]
        spot = float(hist[-1]["close"])
        prev = float(hist[-2]["close"]) if len(hist) > 1 else spot
        chg_pct = round(((spot - prev) / max(prev, 1e-6)) * 100.0, 2)
        summaries.append(
            {
                "metal": m,
                "name": spec["name"],
                "symbol": spec["symbol"],
                "mcx_symbol": spec["mcx_symbol"],
                "kanji": spec["kanji"],
                "unit": spec["unit"],
                "spot_usd": round(spot, 2),
                "change_24h_pct": chg_pct,
            }
        )
    return summaries
