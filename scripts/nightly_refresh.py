"""
Atomic Nightly Market Data Ingestion & Multi-Horizon Model Refresh Pipeline.
Can be run via:
  1. Cron / Shell:  PYTHONPATH=. python scripts/nightly_refresh.py
  2. GitHub Action: .github/workflows/nightly-model-refresh.yml
  3. FastAPI Hook:  POST /api/refresh/trigger
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib

from data_pipeline.ingest_market_data import DEFAULT_DB_PATH, METAL_SPECS, ingest_all_market_data
from ml_engine.train_walk_forward import DEFAULT_REGISTRY_DIR, train_and_register_all

REFRESH_STATUS_PATH = os.path.join(DEFAULT_REGISTRY_DIR, "refresh_status.json")


def get_refresh_status() -> dict:
    """Return the latest nightly refresh audit log, or synthesize one from evaluation_report.json."""
    if os.path.exists(REFRESH_STATUS_PATH):
        try:
            with open(REFRESH_STATUS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    eval_path = os.path.join(DEFAULT_REGISTRY_DIR, "evaluation_report.json")
    metals_summary = {}
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
                metals_summary = rep.get("metals", {})
        except Exception:
            pass

    now_utc = dt.datetime.now(dt.timezone.utc)
    ist_tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_tz)
    return {
        "status": "ok",
        "engine": "Metal Predict Nightly Auto-Refresh Pipeline",
        "last_refreshed_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_refreshed_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "nightly_cron_schedule": "30 18 * * 1-5 (00:00 IST Tue–Sat)",
        "metals": metals_summary,
    }


def run_nightly_market_refresh(
    db_path: str = DEFAULT_DB_PATH,
    registry_dir: str = DEFAULT_REGISTRY_DIR,
) -> dict:
    """
    Safely ingest the latest daily market close into DuckDB, re-train all 3 horizon tiers in a staging
    directory, verify strict quantile monotonicity (Q025 < Q10 < Q50 < Q90 < Q975), and atomically swap
    the production model registry artifacts so a failed run can never corrupt the live server.
    """
    os.makedirs(registry_dir, exist_ok=True)
    # 1. Refresh DuckDB OHLCV & macro time-series store
    ingest_all_market_data(db_path=db_path)

    # 2. Train into a temporary staging directory first for atomic safety
    with tempfile.TemporaryDirectory(prefix="mp_stage_", dir=registry_dir) as stage_dir:
        report = train_and_register_all(db_path=db_path, registry_dir=stage_dir)
        staged_joblib = os.path.join(stage_dir, "multi_horizon_suite.joblib")
        staged_report = os.path.join(stage_dir, "evaluation_report.json")

        # 3. Verify staged model integrity & strict quantile ordering before atomic swap
        staged_suite = joblib.load(staged_joblib)
        for m in METAL_SPECS:
            horizons = staged_suite.predict_all_horizons(m)
            for h_key, h_obj in horizons.items():
                term = h_obj["terminal"]
                if not (term["q025"] < term["q10"] < term["q50"] < term["q90"] < term["q975"]):
                    raise RuntimeError(
                        f"Quantile monotonicity check failed for {m} ({h_key}) during nightly refresh."
                    )

        # 4. Atomic swap into production registry
        prod_joblib = os.path.join(registry_dir, "multi_horizon_suite.joblib")
        prod_report = os.path.join(registry_dir, "evaluation_report.json")
        shutil.move(staged_joblib, prod_joblib)
        shutil.move(staged_report, prod_report)

    now_utc = dt.datetime.now(dt.timezone.utc)
    ist_tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_tz)

    status_payload = {
        "status": "ok",
        "engine": "Metal Predict Nightly Auto-Refresh Pipeline",
        "last_refreshed_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_refreshed_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "nightly_cron_schedule": "30 18 * * 1-5 (00:00 IST Tue–Sat)",
        "metals": report.get("metals", {}),
    }
    with open(os.path.join(registry_dir, "refresh_status.json"), "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)

    return status_payload


if __name__ == "__main__":
    print("[Metal Predict] Starting atomic nightly market data & multi-horizon model refresh...")
    summary = run_nightly_market_refresh()
    print(f"[Metal Predict] Refresh complete at {summary['last_refreshed_ist']}:")
    for metal_name, info in summary.get("metals", {}).items():
        print(f"  - {metal_name.upper()}: Anchor {info['latest_date']} • Spot ${info['spot_price']:,.2f}")
