"""Real-World Pulse (FinBERT News, GSR Radar, Physical Squeeze) & Walk-Forward Backtest Time Machine endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.cache import get_suite, resolve_metal
from ml_engine.nlp_sentiment import get_live_news_pulse, get_physical_and_gsr_radar

router = APIRouter(prefix="/api", tags=["pulse & backtest"])


@router.get("/pulse/overview")
def get_pulse_overview(
    metal: str = Query("gold"),
) -> dict:
    m = resolve_metal(metal)
    suite = get_suite()
    row = suite.latest_rows[m]
    radars = get_physical_and_gsr_radar(row, m)
    news = get_live_news_pulse(m)
    return {
        "metal": m,
        "gsr_radar": radars["gsr_radar"],
        "physical_squeeze": radars["physical_squeeze"],
        "central_bank_tracker": radars["central_bank_tracker"],
        "news": news,
    }


@router.get("/backtest/replay")
def get_backtest_replay(
    metal: str = Query("gold"),
    fold_index: int = Query(0, ge=0, le=4),
) -> dict:
    m = resolve_metal(metal)
    suite = get_suite()
    folds = suite.backtest_folds[m]
    idx = min(max(int(fold_index), 0), len(folds) - 1)
    selected = folds[idx]
    return {
        "metal": m,
        "selected_fold_index": idx,
        "total_folds": len(folds),
        "available_folds": [
            {
                "fold_index": f["fold_index"],
                "label": f["label"],
                "anchor_date": f["anchor_date"],
                "anchor_price": f["anchor_price"],
                "fold_mape": f["fold_mape"],
                "fold_da": f["fold_da"],
            }
            for f in folds
        ],
        "walk_forward_metrics": suite.walk_forward_metrics[m],
        **selected,
    }
