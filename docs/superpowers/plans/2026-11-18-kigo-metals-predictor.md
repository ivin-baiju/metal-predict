# Kigo Metals Multi-Horizon Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack Metal Price Prediction & Commodity Desk web application for Gold, Silver, Platinum, and Copper across 1M, 3M, 6M, 1Y, 5Y, and any specific calendar date using a Three-Tier Quantile ML Ensemble, DuckDB, TreeSHAP, Financial NLP, COMEX-MCX/ETF parity, FastAPI, and Kigo Studio's editorial ukiyo-e design system.

**Architecture:** A DuckDB-backed multimodal data and leak-free feature store (`data_pipeline/`) feeds a Three-Tier Direct Multi-Horizon ML Engine (`ml_engine/`): Tier 1 (1–90d, 1M, 3M) uses direct multi-horizon XGBoost Quantile Regressors; Tier 2 (91–365d, 6M, 1Y) pairs polynomial/Fourier basis decomposition with LightGBM Quantile residual ensembles and a 4-state Macro Regime Classifier; Tier 3 (>365d, 5Y) runs a cointegrated VECM / Ornstein-Uhlenbeck drift model with a 1,000-path Jump-Diffusion Monte Carlo simulator. A FastAPI backend (`backend/`) exposes forecast, Any-Day routing, TreeSHAP, macro/duty stress-simulation, walk-forward backtest replay, and NLP pulse endpoints, while serving the Kigo Studio–styled interactive frontend at `kigo.studio/tools/metal-price-predictor.html`.

**Tech Stack:** Python 3.12 (`uv`), DuckDB, Pandas, NumPy, SciPy, Scikit-Learn, XGBoost, LightGBM, Statsmodels, SHAP, yfinance, FastAPI, Uvicorn, Pytest, HTML5 Canvas/SVG, Kigo Studio CSS/JS design tokens.

**Spec:** `/Users/ivinbaiju/Stark/kigo.studio/docs/superpowers/specs/2026-11-18-metal-predictor-design.md`

## Global Constraints

- **Python Environment:** Managed via `uv` using Python 3.12 at `/Users/ivinbaiju/Stark/kigo.studio/.venv` (`arm64` wheels for `xgboost`, `lightgbm`, `shap`, `duckdb`, `statsmodels`, `scikit-learn`, `yfinance`, `fastapi`, `uvicorn`, `httpx`, `pytest`).
- **Supported Metals:** `gold` (`XAU/USD`, MCX 10g), `silver` (`XAG/USD`, MCX 1kg), `platinum` (`XPT/USD`, MCX 10g), `copper` (`HG/USD`, MCX 1kg).
- **Horizons:** `1M` (30d), `3M` (90d), `6M` (180d), `1Y` (365d), `5Y` (1825d), plus `Any-Day` ($\Delta t \in [1, 2500]$ days).
- **Quantile Monotonicity:** Every forecast point must satisfy $Q_{0.025} \le Q_{0.10} \le Q_{0.50} \le Q_{0.90} \le Q_{0.975}$ strictly via post-hoc isotonic sorting + positive spread enforcement.
- **Zero Lookahead Leakage:** Rolling technical, cointegration, and macro features at date $t$ may only reference prices at $\le t$; Walk-Forward CV splits strictly chronologically.
- **Offline / Rate-Limit Resilience:** `ingest_market_data.py` attempts live `yfinance` / RSS fetch with a 6-second timeout and automatically falls back to a deterministic, regime-calibrated 2011–2026 multi-asset generator if offline or rate-limited.
- **Kigo Studio Visual Fidelity:** Frontend uses Kigo Studio's paper palette (`#fff`, `#efece8`), sumi ink (`#202020`), sakura italic accent (`#db5b7f`), hanko crimson (`#d62828`), double-bordered `#9d3835` ornamental frames, local fonts (`Poppins`, `Kanit`, `Playfair Display`, `YujiBoku-kigo`, `NotoSerifJP-haiku`), and ukiyo-e artwork from `kigo.studio/`.

## Review Focus

- **Past or Invalid Target Date in `/api/predict/date`:** Requesting `target_date <= today` or malformed dates must return HTTP 400 with a clear error message rather than a negative $\Delta t$ crash.
- **Beyond-5Y Target Date ($\Delta t > 1825\text{d}$):** Dates up to 7 years out ($\Delta t \le 2555\text{d}$) must extrapolate cleanly via Tier 3 VECM/Monte Carlo; dates beyond 2555 days return HTTP 400 explaining the 7-year horizon ceiling.
- **Quantile Crossing Under Extreme Macro Shock Sliders:** Perturbing USD/INR ($\pm 20\%$), CPI ($0\text{–}12\%$), and rate cuts ($\pm 300\text{bps}$) simultaneously must never invert the 80% or 95% confidence bands ($Q_{0.025} < Q_{0.10} < Q_{0.50} < Q_{0.90} < Q_{0.975}$).
- **Case-Insensitive / Alias Metal Names:** Passing `Gold`, `XAU`, `gold`, or `SILVER` to API endpoints must normalize cleanly to canonical keys (`gold`, `silver`, `platinum`, `copper`) or return HTTP 404 for unsupported tickers.
- **Zero/Negative Duty or Extreme USD/INR in Parity Engine:** Setting customs duty to `0%` or `25%` must accurately recompute landed INR/10g (24K & 22K), INR/1kg, and Gold/Silver BeES ETF NAV without NaN or division-by-zero errors.

---

### Task 1: Environment, DuckDB Schema & Leak-Free Multimodal Feature Pipeline

**Files:**
- Create: `data_pipeline/__init__.py`
- Create: `data_pipeline/schema.sql`
- Create: `data_pipeline/ingest_market_data.py`
- Create: `data_pipeline/feature_engineering.py`
- Test: `tests/test_feature_pipeline.py`

**Interfaces:**
- Consumes: `yfinance` (optional live network) or deterministic fallback seed `42`.
- Produces:
  - `data_pipeline.ingest_market_data.ingest_all_market_data(db_path: str = "data_pipeline/metal_alpha.duckdb", force_refresh: bool = False) -> dict[str, pd.DataFrame]`
  - `data_pipeline.feature_engineering.build_feature_frame(raw_df: pd.DataFrame, metal: str) -> pd.DataFrame`
  - `data_pipeline.feature_engineering.FEATURE_COLUMNS: list[str]` (36 numerical feature column names)
  - `data_pipeline.feature_engineering.populate_feature_store(db_path: str = "data_pipeline/metal_alpha.duckdb") -> dict[str, pd.DataFrame]`

- [ ] **Step 1: Create Python 3.12 virtualenv, install dependencies, and write the failing feature pipeline tests**

Run:
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python duckdb pandas numpy scipy scikit-learn xgboost lightgbm statsmodels shap yfinance fastapi uvicorn httpx pytest
```

Write `tests/test_feature_pipeline.py`:
```python
import numpy as np
import pandas as pd
from data_pipeline.ingest_market_data import ingest_all_market_data
from data_pipeline.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_frame,
    compute_calendar_catalysts,
)

def test_ingestion_and_feature_no_lookahead(tmp_path):
    db_file = str(tmp_path / "test_metal.duckdb")
    frames = ingest_all_market_data(db_path=db_file, force_refresh=True)
    assert set(frames.keys()) == {"gold", "silver", "platinum", "copper"}
    for metal, df in frames.items():
        assert len(df) >= 3500
        assert {"date", "open", "high", "low", "close", "volume", "dxy", "usdinr", "us10y", "real_yield", "crude", "sp500", "gsr"}.issubset(df.columns)

    # Verify zero lookahead leakage: mutating future prices after index 2000 must not alter features at index 2000
    gold_df = frames["gold"].copy().reset_index(drop=True)
    feat_full = build_feature_frame(gold_df, "gold")
    gold_mutated = gold_df.copy()
    gold_mutated.loc[2001:, "close"] *= 2.5
    feat_mutated = build_feature_frame(gold_mutated, "gold")

    row_orig = feat_full.loc[2000, FEATURE_COLUMNS].to_numpy(dtype=float)
    row_mut = feat_mutated.loc[2000, FEATURE_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_allclose(row_orig, row_mut, rtol=1e-7)
    assert len(FEATURE_COLUMNS) >= 35

def test_calendar_catalysts_detects_festive_and_fomc():
    cat = compute_calendar_catalysts(pd.Timestamp("2027-10-29"))  # Late Oct Dhanteras/Diwali window
    assert cat["is_indian_festive"] is True
    assert cat["wedding_season_multiplier"] > 1.0
    assert "volatility_cone_multiplier" in cat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_feature_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_pipeline'`

- [ ] **Step 3: Implement `data_pipeline/schema.sql`, `data_pipeline/ingest_market_data.py`, and `data_pipeline/feature_engineering.py`**

Implement `schema.sql` with tables `raw_market_daily`, `engineered_features`, `walk_forward_metrics`, and `backtest_trajectories`.
Implement `ingest_all_market_data` to attempt `yfinance` download (`GC=F`, `SI=F`, `PL=F`, `HG=F`, `DX-Y.NYB`, `INR=X`, `^TNX`, `CL=F`, `^GSPC`) from `2011-01-03` to `2026-11-18`, falling back to a realistic cointegrated regime-switching generator anchored to real 2011–2026 price levels (Gold reaching ~$3,920/oz, Silver ~$48.40/oz, Platinum ~$1,145/oz, Copper ~$4.68/lb, USD/INR ~86.45, DXY ~104.2, US10Y ~4.28%, Real Yield ~1.95%).
Implement `build_feature_frame(raw_df, metal)` and `compute_calendar_catalysts(target_date: pd.Timestamp) -> dict` with 36 backward-looking features (`ret_1d`..`ret_252d`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_width`, `bb_pct`, `atr_14`, `vol_21d`, `vol_63d`, `gsr`, `gsr_zscore_63d`, `coint_sma20_200`, `coint_gold_silver`, `dxy_ret_21d`, `usdinr_ret_21d`, `real_yield_level`, `real_yield_chg_21d`, `crude_ret_21d`, `sp500_ret_21d`, `etf_flow_zscore`, `cb_tonnage_pace`, `warehouse_squeeze_idx`, `sin_doy`, `cos_doy`, `sin_dow`, `cos_dow`, `is_wedding_season`, `is_festive_peak`, `is_macro_event_week`, `lag_ret_1`, `lag_ret_2`, `lag_ret_3`, `lag_ret_5`, `lag_ret_10`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_feature_pipeline.py -v`
Expected: PASS

---

### Task 2: Three-Tier Multi-Horizon ML Engine & Walk-Forward Cross-Validation

**Files:**
- Create: `ml_engine/__init__.py`
- Create: `ml_engine/models/__init__.py`
- Create: `ml_engine/models/short_term_xgb.py`
- Create: `ml_engine/models/mid_term_lgbm.py`
- Create: `ml_engine/models/long_term_macro.py`
- Create: `ml_engine/train_walk_forward.py`
- Test: `tests/test_ml_horizons.py`

**Interfaces:**
- Consumes: `data_pipeline.feature_engineering.populate_feature_store`, `FEATURE_COLUMNS`.
- Produces:
  - `ml_engine.models.short_term_xgb.ShortTermXGBEnsemble` (`fit(df)`, `predict_step(feature_row, current_price, days_ahead) -> dict`, `predict_trajectory(feature_row, current_price, max_days=90) -> list[dict]`)
  - `ml_engine.models.mid_term_lgbm.MidTermLGBMEnsemble` (`fit(df)`, `predict_trajectory(feature_row, current_price, max_days=365) -> dict`)
  - `ml_engine.models.long_term_macro.LongTermMacroVECM` (`fit(df)`, `predict_trajectory(feature_row, current_price, max_days=1825, n_sims=1000) -> dict`)
  - `ml_engine.train_walk_forward.train_and_register_all(db_path: str = "data_pipeline/metal_alpha.duckdb", registry_dir: str = "ml_engine/registry") -> dict`

- [ ] **Step 1: Write the failing multi-horizon and walk-forward tests**

Write `tests/test_ml_horizons.py`:
```python
from data_pipeline.feature_engineering import populate_feature_store
from ml_engine.train_walk_forward import train_and_register_all, load_trained_suite

def test_three_tier_training_and_quantile_monotonicity(tmp_path):
    db_file = str(tmp_path / "ml_test.duckdb")
    reg_dir = str(tmp_path / "registry")
    populate_feature_store(db_path=db_file)
    report = train_and_register_all(db_path=db_file, registry_dir=reg_dir)

    assert set(report["metals"].keys()) == {"gold", "silver", "platinum", "copper"}
    suite = load_trained_suite(registry_dir=reg_dir)

    for metal in ["gold", "silver", "platinum", "copper"]:
        m_metrics = report["metals"][metal]["walk_forward_metrics"]
        assert set(m_metrics.keys()) == {"1M", "3M", "6M", "1Y", "5Y"}
        for h, vals in m_metrics.items():
            assert vals["mape"] > 0 and vals["mape"] < 35.0
            assert 45.0 <= vals["directional_accuracy"] <= 95.0

        # Check strict quantile ordering across all 5 horizons
        horizons = suite.predict_all_horizons(metal)
        for h_key in ["1M", "3M", "6M", "1Y", "5Y"]:
            pts = horizons[h_key]["series"]
            assert len(pts) >= 12
            for pt in pts:
                assert pt["q025"] < pt["q10"] < pt["q50"] < pt["q90"] < pt["q975"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ml_horizons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_engine'`

- [ ] **Step 3: Implement Tier 1 (`short_term_xgb.py`), Tier 2 (`mid_term_lgbm.py`), Tier 3 (`long_term_macro.py`), and `train_walk_forward.py`**

- In `short_term_xgb.py`: Train direct horizon `XGBRegressor` quantile heads ($\alpha \in \{0.025, 0.10, 0.50, 0.90, 0.975\}$) at anchor steps $h \in \{1, 5, 10, 21, 30, 45, 60, 90\}$ days; interpolate daily trajectories with seasonal/cyclical modulation and enforce strict quantile ordering (`np.maximum.accumulate` + minimum spread guard).
- In `mid_term_lgbm.py`: Decompose price log-paths into polynomial trend + 3-harmonic Fourier seasonal basis plus direct `LGBMRegressor` quantile residual heads at $h \in \{90, 135, 180, 270, 365\}$ days, and compute 4-state soft regime probabilities (`Sovereign Accumulation Bull`, `Reflation & Industrial Breakout`, `Rangebound Equilibrium`, `Real-Yield Tightening Pullback`).
- In `long_term_macro.py`: Fit a cointegrated Vector Error Correction / Ornstein-Uhlenbeck mean-reverting real-drift model against real yields, DXY, and M2/debt proxy, and simulate 1,000 Jump-Diffusion Monte Carlo paths over 1,825 days (60 monthly steps + 8 representative sample paths for visualization).
- In `train_walk_forward.py`: Execute 5 expanding-window walk-forward folds (`2022-06-01`, `2023-06-01`, `2024-06-01`, `2025-05-01`, `2026-05-01`), compute out-of-sample RMSE, MAPE, Pinball Loss, and Directional Accuracy, store backtest replay curves in DuckDB & JSON registry, and save the trained `MultiHorizonSuite`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ml_horizons.py -v`
Expected: PASS

---

### Task 3: TreeSHAP Explainability, Event-Aware Any-Day Router, Financial NLP & Parity Engine

**Files:**
- Create: `ml_engine/explainability.py`
- Create: `ml_engine/nlp_sentiment.py`
- Create: `ml_engine/parity_engine.py`
- Modify: `tests/test_ml_horizons.py`

**Interfaces:**
- Consumes: `MultiHorizonSuite`, `compute_calendar_catalysts`.
- Produces:
  - `ml_engine.explainability.predict_specific_date(suite, metal: str, target_date_str: str) -> dict`
  - `ml_engine.explainability.compute_shap_drivers(suite, metal: str, horizon_or_days: int | str) -> list[dict]`
  - `ml_engine.nlp_sentiment.get_live_news_pulse(metal: str = "all") -> dict`
  - `ml_engine.parity_engine.compute_comex_mcx_parity(metal: str, spot_usd: float, usdinr: float = 86.45, bcd_pct: float = 6.0, aidc_pct: float = 1.0, gst_pct: float = 3.0, mcx_spread_pct: float = 0.35) -> dict`

- [ ] **Step 1: Add tests for Any-Day routing ($\Delta t \le 90$, $91\text{–}365$, $>365$), TreeSHAP weights, NLP sentiment, and COMEX-MCX/ETF parity**

Append to `tests/test_ml_horizons.py`:
```python
import pytest
from ml_engine.explainability import predict_specific_date, compute_shap_drivers
from ml_engine.nlp_sentiment import get_live_news_pulse
from ml_engine.parity_engine import compute_comex_mcx_parity

def test_any_day_routing_tiers_and_validation(tmp_path):
    db_file = str(tmp_path / "ml_test.duckdb")
    reg_dir = str(tmp_path / "registry")
    populate_feature_store(db_path=db_file)
    train_and_register_all(db_path=db_file, registry_dir=reg_dir)
    suite = load_trained_suite(registry_dir=reg_dir)

    d_short = predict_specific_date(suite, "gold", "2026-12-15")   # <= 90d -> Tier 1
    assert d_short["tier"] == "Tier 1: Short-Term TFT/XGBoost Matrix"
    assert d_short["q025"] < d_short["q10"] < d_short["q50"] < d_short["q90"] < d_short["q975"]
    assert len(d_short["shap_drivers"]) >= 6

    d_mid = predict_specific_date(suite, "silver", "2027-06-15")   # 91-365d -> Tier 2
    assert d_mid["tier"] == "Tier 2: Medium-Term N-BEATS/LightGBM Basis"

    d_long = predict_specific_date(suite, "gold", "2029-11-18")    # > 365d -> Tier 3
    assert d_long["tier"] == "Tier 3: Long-Term VECM + 1,000-Path Monte Carlo"

    with pytest.raises(ValueError, match="future date"):
        predict_specific_date(suite, "gold", "2025-01-01")

    with pytest.raises(ValueError, match="7-year"):
        predict_specific_date(suite, "gold", "2036-01-01")

def test_parity_and_nlp_pulse():
    parity = compute_comex_mcx_parity("gold", spot_usd=4100.0, usdinr=86.5, bcd_pct=6.0, aidc_pct=1.0, gst_pct=3.0)
    assert parity["inr_per_10g_24k"] > 100000
    assert parity["inr_per_10g_22k"] < parity["inr_per_10g_24k"]
    assert parity["etf"]["gold_bees_nav"] > 50

    parity_zero_duty = compute_comex_mcx_parity("gold", spot_usd=4100.0, usdinr=86.5, bcd_pct=0.0, aidc_pct=0.0, gst_pct=0.0)
    assert parity_zero_duty["inr_per_10g_24k"] < parity["inr_per_10g_24k"]

    pulse = get_live_news_pulse("gold")
    assert len(pulse["headlines"]) >= 6
    assert -1.0 <= pulse["composite_sentiment"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ml_horizons.py -k "test_any_day_routing or test_parity" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_engine.explainability'`

- [ ] **Step 3: Implement `ml_engine/explainability.py`, `ml_engine/nlp_sentiment.py`, and `ml_engine/parity_engine.py`**

- In `explainability.py`: Use `shap.TreeExplainer` on the active tier's median booster to extract top positive/negative feature attributions, normalize weights to percentage contributions, fuse NLP sentiment weight, apply FOMC/CPI/Budget cone widening and Indian wedding/festive multipliers, and compute recommended GTT entry zones (`gtt_accumulate_low`, `gtt_accumulate_high`, `stop_loss`, `target_exit`).
- In `nlp_sentiment.py`: Attempt live RSS headline fetch with a 3-second timeout and score headlines with a domain-specific financial commodity sentiment analyzer (FinBERT domain lexicon + entity/catalyst classifier tagging `Central Bank`, `Fed & Yields`, `Tariffs & Geopolitics`, `Physical Supply`, `Indian Retail`), outputting polarity (`+1`, `0`, `-1`), confidence, and estimated basis-point impact.
- In `parity_engine.py`: Convert `USD/oz` (or `USD/lb` for Copper) to landed MCX & retail prices (`INR/10g 24K`, `INR/10g 22K`, `INR/1kg`), compute tax breakdown (`base_inr`, `customs_duty_inr`, `aidc_inr`, `gst_inr`), and map to ETF NAVs (`Gold BeES`, `Silver BeES`, `GLD`, `SLV`, `CPER`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ml_horizons.py -v`
Expected: PASS

---

### Task 4: FastAPI Backend Service, Macro Shock Stress-Tester & API Endpoints

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/cache.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/routers/forecast.py`
- Create: `backend/routers/simulate.py`
- Create: `backend/routers/pulse.py`
- Create: `backend/main.py`
- Test: `tests/test_api_endpoints.py`

**Interfaces:**
- Consumes: `ml_engine` modules, `data_pipeline` DuckDB store.
- Produces REST Endpoints in `backend.main:app`:
  - `GET /api/health`
  - `GET /api/predict/horizons?metal=gold`
  - `GET /api/predict/date?metal=gold&target_date=YYYY-MM-DD`
  - `POST /api/simulate/macro-shock`
  - `POST /api/simulate/parity-duty`
  - `GET /api/pulse/overview?metal=gold`
  - `GET /api/backtest/replay?metal=gold&anchor_index=0`

- [ ] **Step 1: Write the failing FastAPI endpoint tests (including all 5 Review Focus edge cases)**

Write `tests/test_api_endpoints.py`:
```python
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_horizons_and_case_insensitive_metal():
    r = client.get("/api/predict/horizons?metal=XAU")
    assert r.status_code == 200
    data = r.json()
    assert data["metal"] == "gold"
    assert set(data["horizons"].keys()) == {"1M", "3M", "6M", "1Y", "5Y"}

    r_bad = client.get("/api/predict/horizons?metal=unobtanium")
    assert r_bad.status_code == 404

def test_any_day_endpoint_and_date_validation():
    r = client.get("/api/predict/date?metal=gold&target_date=2027-10-29")
    assert r.status_code == 200
    body = r.json()
    assert body["delta_days"] > 0
    assert body["q025"] < body["q10"] < body["q50"] < body["q90"] < body["q975"]
    assert "gtt_zone" in body

    r_past = client.get("/api/predict/date?metal=gold&target_date=2020-01-01")
    assert r_past.status_code == 400

    r_far = client.get("/api/predict/date?metal=gold&target_date=2038-01-01")
    assert r_far.status_code == 400

def test_macro_shock_extreme_sliders_preserve_quantile_order():
    r = client.post("/api/simulate/macro-shock", json={
        "metal": "silver",
        "horizon": "1Y",
        "usdinr_rate": 98.0,
        "us_cpi_pct": 9.5,
        "rate_cuts_bps": 250,
        "real_yield_delta_bps": -150,
        "customs_duty_pct": 15.0,
        "aidc_pct": 2.5,
        "gst_pct": 3.0
    })
    assert r.status_code == 200
    sim = r.json()
    for pt in sim["stressed_series"]:
        assert pt["q025"] < pt["q10"] < pt["q50"] < pt["q90"] < pt["q975"]

def test_backtest_replay_and_pulse_overview():
    r_bt = client.get("/api/backtest/replay?metal=gold&fold_index=2")
    assert r_bt.status_code == 200
    assert len(r_bt.json()["predicted_series"]) > 10

    r_pulse = client.get("/api/pulse/overview?metal=gold")
    assert r_pulse.status_code == 200
    p = r_pulse.json()
    assert "gsr_radar" in p and "physical_squeeze" in p and "news" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_endpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend'`

- [ ] **Step 3: Implement `backend/cache.py`, `backend/routers/forecast.py`, `backend/routers/simulate.py`, `backend/routers/pulse.py`, and `backend/main.py`**

Mount `/api/*` routers and mount `/Users/ivinbaiju/Stark/kigo.studio/kigo.studio` as static files at `/` so visiting `http://localhost:8000/tools/metal-price-predictor.html` or `http://localhost:8000/` serves the entire Kigo Studio site and API seamlessly.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_endpoints.py -v`
Expected: PASS

---

### Task 5: Kigo Studio Editorial Frontend & Interactive Commodity Desk

**Files:**
- Create: `kigo.studio/tools/metal-price-predictor.html`
- Create: `kigo.studio/_nuxt/metal-predictor.css`
- Create: `kigo.studio/_nuxt/metal-predictor.js`
- Modify: `kigo.studio/tools.html`
- Create: `run_app.sh`
- Test: `tests/test_api_endpoints.py`

**Interfaces:**
- Consumes: All `/api/*` endpoints from Task 4, local Kigo Studio fonts (`kigo.studio/fonts/*`), CSS (`kigo.studio/_nuxt/entry.pKyUpuCW.css`, `PaperPage.C8l7_zFm.css`), and ukiyo-e illustrations (`kigo.studio/.netlify/*`).
- Produces:
  - `GET /tools/metal-price-predictor.html` (complete Kigo Studio Commodity Desk UI)
  - Updated `GET /tools.html` featuring the Metal Price Predictor card with red kanji hanko stamp (`金`).

- [ ] **Step 1: Add frontend integration test to `tests/test_api_endpoints.py`**

Append to `tests/test_api_endpoints.py`:
```python
def test_kigo_frontend_pages_served():
    r_tool = client.get("/tools/metal-price-predictor.html")
    assert r_tool.status_code == 200
    html = r_tool.text
    assert "Metal" in html and "metal-predictor.js" in html
    assert "contact-form-card" in html  # Kigo Studio double-bordered ornamental frame

    r_hub = client.get("/tools.html")
    assert r_hub.status_code == 200
    assert "metal-price-predictor.html" in r_hub.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_endpoints.py::test_kigo_frontend_pages_served -v`
Expected: FAIL with `404 != 200`

- [ ] **Step 3: Build `kigo.studio/tools/metal-price-predictor.html`, `kigo.studio/_nuxt/metal-predictor.css`, `kigo.studio/_nuxt/metal-predictor.js`, update `kigo.studio/tools.html`, and create `run_app.sh`**

Implement the Kigo Studio Commodity Desk with:
1. **Kigo Header, Falling Sakura Petals, Ukiyo-e Hero Backdrop & Vertical Haiku (`金銀の 波間を照らす 先見かな`).**
2. **Metal Selector & Parity Toolbar:** Switch between **Gold (`XAU / 金`)**, **Silver (`XAG / 銀`)**, **Platinum (`XPT / 白`)**, and **Copper (`HG / 銅`)**; toggle units (`USD/oz`, `INR/10g 24K`, `INR/10g 22K`, `INR/1kg`, `ETF NAV`).
3. **Interactive Multi-Horizon Probability Fan Chart (`1M`, `3M`, `6M`, `1Y`, `5Y`):** Custom high-DPI Canvas/SVG chart rendering historical candlesticks/line, glowing sakura/crimson median projection, shaded 80% and 95% quantile cones, 5Y Monte Carlo paths, and crosshair tooltip.
4. **Walk-Forward Backtest Time Machine Scrubber:** Rewind present-day anchor across 5 walk-forward folds to compare out-of-sample predictions against actual realized prices, displaying RMSE, MAPE, Pinball Coverage, and Directional Accuracy.
5. **"Predict Any Specific Day" Precision Inspector Card:** Double-framed `#9d3835` Kigo ornamental card with calendar date picker, $\Delta t$ tier router badge, predicted median + 80%/95% range, festive/macro catalyst badges, and ETF/GTT limit-buy trigger zone.
6. **TreeSHAP Feature Attribution Bar ("Why This Price?")** + Audit Narrative.
7. **Interactive Macro & Customs Duty Stress-Tester ("What-If" Sliders):** Live sliders for `USD/INR`, `US CPI (%)`, `Rate Cuts (bps)`, `Real Yield Delta`, and `India Customs Duty + AIDC (%)` updating the chart curve and COMEX-MCX landed cost table in real time.
8. **Real-World Pulse Desk:** Gold-to-Silver Ratio (GSR) Radar, Physical vs. Paper Warehouse Squeeze Gauge, Central Bank Accumulation Tracker, and FinBERT Live News Sentiment Feed.

- [ ] **Step 4: Run full test suite to verify everything passes**

Run: `.venv/bin/pytest -v`
Expected: All tests PASS.
