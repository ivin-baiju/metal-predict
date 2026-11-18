# Kigo Metals — Multi-Horizon Metal Price Prediction & Commodity Desk Design Spec

**Date:** 2026-11-18  
**Project Workspace:** `/Users/ivinbaiju/Stark/kigo.studio`  
**Selected Architecture:** Approach 1 — Direct Multi-Horizon Quantile GBDT (XGBoost + LightGBM) + Basis Decomposition + Cointegrated VECM / Monte Carlo Engine with DuckDB, TreeSHAP, Financial NLP Sentiment, and FastAPI serving a Kigo Studio–styled Web App.

---

## 1. Intent and Why

Predicting metal prices across both short-term windows (any specific calendar day, 1 month, 3 months) and multi-year horizons (6 months, 1 year, 5 years) fails when built on a single autoregressive model because feeding $\hat{y}_{t+1}$ back into $\hat{y}_{t+2}$ compounds noise until the forecast flatlines. Moreover, purely technical OHLCV models get blindsided by macroeconomic shifts (real yields, DXY, USD/INR depreciation), physical supply deficits, Indian customs duty changes, seasonal wedding/festive buying, and breaking financial news.

This project builds **Kigo Metals (Metal Alpha Predictor)** for commodity traders, swing/positional investors, and bullion/ETF buyers:
1. **A production-grade Multi-Horizon Hybrid ML Engine** in Python that routes predictions by horizon tier (`1M`, `3M`, `6M`, `1Y`, `5Y`, and `Any Specific Calendar Day`), trained with expanding-window Walk-Forward Cross-Validation and Pinball (Quantile) Loss to output 10th/50th/90th (80% band) and 2.5th/97.5th (95% band) probability cones alongside exact **TreeSHAP** feature attributions.
2. **A Full-Stack Python FastAPI Backend + Kigo Studio Frontend** that pairs quantitative depth with **Kigo Studio's** signature Japanese ukiyo-e editorial design system from `kigo.studio/`—warm washi paper surfaces (`#fff` and `#efece8`), sumi ink typography (`#202020`), sakura italic accents (`#db5b7f`), crimson hanko kanji stamps (`#d62828`), double-bordered `#9d3835` ornamental frames, and woodblock floral/character artwork.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **G1. Automated Multimodal Data & Feature Store (`data_pipeline/`):**
  - Fetch 15+ years of daily historical data via `yfinance` (with an automatic deterministic high-fidelity fallback generator if Yahoo Finance/FRED is offline or rate-limited) for **Gold (`GC=F`)**, **Silver (`SI=F`)**, **Platinum (`PL=F`)**, and **Copper (`HG=F`)**, plus cross-asset & macro series: **US Dollar Index (`DX-Y.NYB`)**, **USD/INR (`INR=X`)**, **US 10Y Treasury Yield (`^TNX`)**, **TIPS / Real Yield proxy (`TIP`)**, **Crude Oil (`CL=F`)**, **S&P 500 (`^GSPC`)**, **Nifty 50 (`^NSEI`)**, and **Gold/Silver ETFs (`GLD`, `SLV`, `GOLDBEES.NS`, `SILVERBEES.NS`)**.
  - Persist cleaned daily and weekly feature tables, model evaluation metrics, and walk-forward backtest snapshots in **DuckDB** (`data_pipeline/metal_alpha.duckdb`).
  - Engineer 35+ strictly leak-free features (computed using only data available at $\le t$): multi-window log returns (`1d, 5d, 21d, 63d, 126d, 252d`), `RSI(14)`, `MACD(12,26,9)`, Bollinger Band width, `ATR(14)`, rolling 30d/90d volatility, rolling 30d Gold-to-Silver Ratio (`GSR`) & z-score, Engle-Granger cointegration residuals (`SMA20 - SMA200` and Gold-Silver spread), cyclical calendar encodings ($\sin/\cos$ of day-of-year, day-of-week, month), Indian festive/wedding demand windows (Dhanteras, Akshaya Tritiya, Nov–Feb & Apr–May marriage seasons), and scheduled macro event flags (FOMC, US CPI, India Union Budget).
- **G2. Three-Tier Direct Multi-Horizon ML Engine (`ml_engine/`):**
  - **Tier 1 — Short-Term & Exact Day ($\Delta t \le 90\text{d}$, `1M` = 30d, `3M` = 90d):** Direct multi-horizon **XGBoost Quantile Regressors** (`objective="reg:quantileerror"` for $\alpha \in \{0.025, 0.10, 0.50, 0.90, 0.975\}$) over lagged sequence embeddings and technical/currency momentum.
  - **Tier 2 — Medium-Term ($91\text{d} \le \Delta t \le 365\text{d}$, `6M` = 180d, `1Y` = 365d):** Direct **LightGBM Quantile Ensemble + Trend/Seasonality Basis Decomposition** (N-BEATS–inspired polynomial trend + Fourier seasonal basis functions paired with LightGBM quantile residual heads) driven by real interest rates, CPI inflation, ETF flow momentum, and USD/INR, plus a 4-state **Macro Regime Classifier** (`Sovereign Accumulation Bull`, `Reflation / Industrial Breakout`, `Rangebound Consolidation`, `Real-Yield Tightening Pullback`).
  - **Tier 3 — Long-Term Macro ($\Delta t > 365\text{d}$, `5Y` = 1,825d):** **Vector Error Correction (VECM) / Cointegrated Ornstein-Uhlenbeck Structural Trend Model + 1,000-Path Jump-Diffusion Monte Carlo Simulator** anchored to M2/sovereign debt debasement drift, central bank reserve accumulation, green-energy/EV industrial demand, and mine supply constraints.
- **G3. Walk-Forward Cross-Validation & Backtest Time Machine:**
  - Train and evaluate models using **Expanding-Window Walk-Forward Cross-Validation** across 5 historical cutoff dates (e.g., 2022, 2023, 2024, 2025, 2026).
  - Compute and log **RMSE**, **MAPE (%)**, **Pinball / Quantile Coverage (80% & 95%)**, and **Directional Accuracy (DA %)** per metal and horizon.
  - Store historical out-of-sample forecast trajectories vs. actual realized prices so the frontend **Walk-Forward Backtest Time Machine** scrubber can replay how the model performed from any historical anchor date.
- **G4. Event-Aware "Predict Any Specific Day" Engine & TreeSHAP Explainability:**
  - Route any user-selected future date (`YYYY-MM-DD`) by $\Delta t = \text{Target Date} - \text{Today}$ to Tier 1 ($\le 90\text{d}$), Tier 2 ($91\text{–}365\text{d}$), or Tier 3 ($> 365\text{d}$).
  - Apply **Macro Catalyst Volatility Cones** (widening 80%/95% uncertainty bands on FOMC, US CPI, and India Union Budget dates) and **Festive/Wedding Season Multipliers** (empirical physical demand premium during peak Indian buying windows).
  - Extract exact **TreeSHAP** values (`shap.TreeExplainer`) fused with live news sentiment weights to generate human-readable audit narratives and horizontal attribution bars ("Why This Price?").
- **G5. Real-World Trading & Physical Market Modules:**
  - **COMEX-to-MCX Parity & Duty/Tax Shock Simulator:** Convert global spot prices (`USD/oz` or `USD/lb`) into live Indian landed prices (`INR/10g` 24K & 22K for Gold, `INR/1kg` for Silver & Copper, `INR/10g` for Platinum) factoring in live `USD/INR`, Basic Customs Duty (BCD), Agriculture Infrastructure & Development Cess (AIDC), local MCX premium, and 3% GST.
  - **ETF & GTT Order Optimizer:** Map metal forecasts to domestic ETF NAV equivalents (**Nippon India ETF Gold BeES**, **Silver BeES**, **GLD**, **SLV**) and calculate optimal Good-Till-Triggered (GTT) limit-buy accumulation zones based on the 10th/25th percentile support bands and ATR.
  - **Live News & Multimodal NLP Pulse:** Ingest financial headlines (live RSS feeds from Yahoo Finance / CNBC / MarketWatch / Mint with curated fallback headlines) scored by a domain-specific financial sentiment engine (FinBERT financial vocabulary + optional transformer pipeline) into $+1$ (Bullish), $0$ (Neutral), $-1$ (Bearish) scores, rolling sentiment dispersion, and estimated basis-point ($\text{bps}$) curve impact.
  - **Gold-to-Silver Ratio (GSR) Radar & Physical vs. Paper Squeeze Gauge:** Track live GSR vs. 15-year historical percentiles for mean-reversion swap signals, and compare COMEX/SHFE registered warehouse stocks against paper futures open interest to quantify physical delivery tightness.
- **G6. Kigo Studio Editorial Frontend (`frontend/` & `kigo.studio/`):**
  - Build the frontend with **Kigo Studio's exact visual language**: local fonts (`Poppins`, `Kanit`, `Playfair Display Italic`, `YujiBoku-kigo` brush kanji, `NotoSerifJP-haiku`), warm paper (`#fff`, `#efece8`), sumi ink (`#202020`), sakura (`#db5b7f`), crimson hanko stamps (`#d62828`), double-bordered `#9d3835` ornamental frames (`contact-form-card`), ambient falling sakura petals, vertical Japanese haiku motifs, and ukiyo-e illustrations.
  - Link the application directly from `kigo.studio/tools.html` and `kigo.studio/index.html` and serve it seamlessly from the FastAPI server.

### Non-Goals
- Live brokerage order execution (no real-money API keys or trade execution).
- Sub-minute high-frequency order-book tick scalping.

### Anti-Goals
- **No recursive autoregressive rollouts** that flatline after 30 days.
- **No random `train_test_split`** on time-series data (zero lookahead leakage).
- **No generic dark-mode SaaS dashboard**—the UI must feel unmistakably crafted by Kigo Studio.

---

## 3. Constraints

- **Local Environment:** macOS (Apple Silicon `arm64`), managed via `uv` with Python 3.12 (`.venv`) so `xgboost`, `lightgbm`, `scikit-learn`, `shap`, `statsmodels`, `duckdb`, `yfinance`, `fastapi`, and `uvicorn` install cleanly from prebuilt wheels and train in under 60 seconds.
- **Offline / Rate-Limit Resilience:** Financial APIs (`yfinance`, RSS feeds) can rate-limit or drop connections. The data pipeline must attempt live `yfinance` / RSS ingestion first and seamlessly fall back to a calibrated, realistic multi-asset historical generator anchored to real 2011–2026 market regimes so training and inference never crash.
- **Existing Kigo Studio Assets:** Reuse fonts in `/Users/ivinbaiju/Stark/kigo.studio/kigo.studio/fonts/` and ukiyo-e illustrations in `/Users/ivinbaiju/Stark/kigo.studio/kigo.studio/.netlify/` without breaking existing static pages in `kigo.studio/`.

---

## 4. Approaches Considered

1. **Approach 1 (Selected): Direct Multi-Horizon Quantile GBDT (XGBoost + LightGBM) + Trend/Seasonality Basis Decomposition + Cointegrated VECM / Monte Carlo with DuckDB, TreeSHAP & FastAPI**
   - *Pros:* Trains all 4 metals across 5 horizons and 5 walk-forward folds in <60s on Apple Silicon; immune to autoregressive error compounding; provides exact fast `TreeSHAP` explanations; allows sub-20ms live recalculation when users drag the Macro "What-If" and Customs Duty sliders.
   - *Cons:* Uses temporal lag embeddings + polynomial/Fourier basis decomposition + LightGBM/XGBoost instead of heavy PyTorch neural weights.
2. **Approach 2: Heavy Deep-Learning Sequence Stack (PyTorch TFT + N-BEATS + HuggingFace FinBERT)**
   - *Pros:* Uses literal PyTorch neural network classes.
   - *Cons:* Takes 15–30 minutes to train across 4 metals and walk-forward splits; overfits small daily financial tables (~3,800 rows); lacks native fast `TreeSHAP` and lags on interactive slider stress-tests.
3. **Approach 3: Classical Econometric Stack (SARIMAX + VECM Only)**
   - *Pros:* Lightweight and parametric.
   - *Cons:* Misses non-linear regime interactions across DXY, yields, and technical momentum; cannot provide TreeSHAP attributions.

**Why Approach 1 was chosen:** The user prioritized model quality, multi-horizon precision, and a responsive Python FastAPI full-stack application. Approach 1 delivers true walk-forward quantile ensembles, exact TreeSHAP telemetry, cointegrated 5-year Monte Carlo cones, and instantaneous interactive stress-testing.

---

## 5. Technical Architecture & Component Design

### 5.1 Project Directory Structure
```text
/Users/ivinbaiju/Stark/kigo.studio/
├── data_pipeline/
│   ├── __init__.py
│   ├── ingest_market_data.py     # Live yfinance + RSS news fetcher with deterministic multi-asset fallback
│   ├── feature_engineering.py    # 35+ leak-free technical, macro, cointegration, GSR & calendar features
│   ├── schema.sql                # DuckDB time-series & model registry schema
│   └── metal_alpha.duckdb        # DuckDB feature store, predictions, & walk-forward backtest database
├── ml_engine/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── short_term_xgb.py     # Tier 1 (1D–90D, 1M, 3M): Direct multi-horizon XGBoost Quantile Regressors
│   │   ├── mid_term_lgbm.py      # Tier 2 (6M, 1Y): LightGBM Quantile Ensemble + Fourier/Polynomial Basis + Regime Classifier
│   │   └── long_term_macro.py    # Tier 3 (5Y): Cointegrated VECM / Ornstein-Uhlenbeck + 1,000-Path Monte Carlo Simulator
│   ├── train_walk_forward.py     # Expanding-window Walk-Forward CV, metric logging (RMSE, MAPE, DA, Pinball), model serialization
│   ├── explainability.py         # TreeSHAP extractor + Event/Festive catalyst adjuster + audit narrative generator
│   ├── nlp_sentiment.py          # Financial headline NLP scorer (FinBERT lexicon + sentiment dispersion + bps impact)
│   ├── parity_engine.py          # COMEX-to-MCX landed cost, Customs Duty/GST shock simulator, ETF NAV & GTT optimizer
│   └── registry/                 # Trained model artifacts (.joblib / .json) & walk-forward evaluation reports
├── backend/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point + static asset & CORS configuration
│   ├── cache.py                  # In-memory / DuckDB pre-computed horizon & parity cache
│   └── routers/
│       ├── __init__.py
│       ├── forecast.py           # GET /api/predict/horizons, GET /api/predict/date, GET /api/models/metrics
│       ├── simulate.py           # POST /api/simulate/macro-shock, POST /api/simulate/parity-duty
│       └── pulse.py              # GET /api/pulse/news, GET /api/pulse/physical-radar, GET /api/backtest/replay
├── kigo.studio/                  # Existing Kigo Studio static site + integrated Metal Predictor app
│   ├── tools/
│   │   └── metal-price-predictor.html  # Full interactive Kigo Studio Metal Predictor Desk
│   ├── _nuxt/
│   │   ├── metal-predictor.css         # Kigo Studio editorial design system styles for the commodity desk
│   │   └── metal-predictor.js          # Interactive chart, fan cones, Any-Day picker, SHAP, sliders & backtest time machine
│   └── tools.html                # Updated to feature the Metal Price Predictor tool card with Kigo hanko stamp
└── tests/
    ├── test_feature_pipeline.py  # Verifies zero lookahead leakage, stationarity transforms, and GSR math
    ├── test_ml_horizons.py       # Verifies quantile ordering (q025 < q10 < q50 < q90 < q975), Any-Day routing & SHAP
    └── test_api_endpoints.py     # FastAPI TestClient verification of all forecast, simulation, parity & backtest routes
```

### 5.2 Multi-Horizon ML Pipeline Details

1. **Direct Horizon Targets (No Recursive Rollout):**
   - For any anchor date $t$ and target horizon $h \in \{1, 5, 10, 21, 30, 45, 60, 90, 180, 270, 365, 730, 1095, 1460, 1825\}$ calendar days, the model predicts the log return $\Delta \ln P_{t \to t+h} = \ln(P_{t+h} / P_t)$ directly from the feature vector $\mathbf{x}_t$, then reconstructs price trajectories $P_{t+h}^{(\alpha)} = P_t \cdot \exp(\hat{y}_{t+h}^{(\alpha)})$.
   - Intermediate daily points along `1M` (30d), `3M` (90d), `6M` (180d), `1Y` (365d), and `5Y` (1,825d) curves are synthesized by blending the direct horizon anchor heads with the tier's structural basis curve and seasonal/calendar modulation so trajectories are smooth, realistic, and respect strict quantile monotonicity ($Q_{0.025} \le Q_{0.10} \le Q_{0.50} \le Q_{0.90} \le Q_{0.975}$).
2. **"Predict Any Specific Day" Routing ($\Delta t = \text{Target Date} - \text{Today}$):**
   - **$\Delta t \le 90$ days:** Queries Tier 1 (`short_term_xgb.py`) daily quantile matrix, applying day-of-week, MCX/COMEX holiday liquidity adjustments, Indian festive/wedding demand multipliers, and FOMC/CPI event volatility cone expansion.
   - **$91 \le \Delta t \le 365$ days:** Queries Tier 2 (`mid_term_lgbm.py`) interpolating the LightGBM + Fourier/Polynomial basis quantile distribution, returning the predicted median, 80%/95% range, regime classification probabilities, and top TreeSHAP drivers.
   - **$\Delta t > 365$ days (up to 5+ years):** Queries Tier 3 (`long_term_macro.py`) VECM / cointegrated structural trend model and runs 1,000 Jump-Diffusion Monte Carlo paths in real time, returning the expected median, 80%/95% fan bounds, sample Monte Carlo paths, and structural macro drivers.

### 5.3 Kigo Studio Frontend Experience
- **Aesthetic & Typography:**
  - Built on Kigo Studio's `paper-page` system (`--paper: #fff`, `--paper-warm: #efece8`, `--ink: #202020`, `--ink-soft: #4a4144`, `--sakura: #db5b7f`, `--red: #d62828`, `--frame-ink: #9d3835`).
  - Uses Kigo Studio's header, mobile drawer, custom cursor, ambient falling sakura petals, ornamental double-framed cards (`contact-form-card` with SVG corner brackets), chamfered cards, and red `押` / `金` / `銀` / `白` / `銅` kanji hanko stamps.
- **Interactive Modules on the Page:**
  1. **Editorial Hero & Metal Selector Bar:** Switch between **Gold (`XAU / 金`)**, **Silver (`XAG / 銀`)**, **Platinum (`XPT / 白`)**, and **Copper (`HG / 銅`)** with live spot price, 24h change, MCX landed parity, and global unit/currency/purity toggles (`USD/oz`, `INR/10g 24K`, `INR/10g 22K`, `INR/1kg`, `Gold/Silver BeES ETF`).
  2. **Multi-Horizon Probability Fan Chart (`1M`, `3M`, `6M`, `1Y`, `5Y`):** Interactive HTML5 Canvas / SVG financial terminal chart in Kigo ink-and-sakura styling showing historical candlesticks/price line, glowing median projection trajectory, shaded 80% and 95% probability cones, Monte Carlo sample paths (on 5Y), and hover crosshair telemetry.
  3. **Walk-Forward Backtest Time Machine Scrubber:** Interactive timeline slider under the chart allowing users to rewind the "Present Day" anchor to historical dates (2022–2026) and visually overlay the model's historical out-of-sample prediction cone against actual realized prices, displaying fold RMSE, MAPE, and Directional Accuracy.
  4. **"Predict Any Specific Day" Precision Inspector Card:** Framed in Kigo's `#9d3835` double-corner ornamental card; lets the user pick any future calendar date, displays $\Delta t$ days ahead, active routing tier (`Tier 1 TFT/XGBoost`, `Tier 2 Basis/LightGBM`, or `Tier 3 VECM/Monte Carlo`), predicted median price, 80% & 95% confidence bounds, active calendar/macro catalysts (e.g., Dhanteras wedding peak, FOMC meeting), and **ETF / GTT Limit-Buy Trigger Zone** recommendations.
  5. **TreeSHAP Attribution Bar ("Why This Price?"):** Horizontal diverging bar chart showing positive and negative feature attributions (in `%` weight and estimated currency impact) alongside an auto-generated TreeSHAP + NLP audit narrative.
  6. **Interactive Macro & Indian Duty Stress-Tester ("What-If" Sliders):** Live sliders for `USD/INR Exchange Rate`, `US CPI Inflation (%)`, `Fed / RBI Rate Cuts (bps)`, `US 10Y Real Yield (%)`, `India Basic Customs Duty + AIDC (%)`, and `GST (%)` that hit `/api/simulate/macro-shock` and dynamically shift the forecast curve and landed MCX prices on screen.
  7. **Real-World Pulse Desk (Physical Squeeze, GSR Radar & FinBERT News Feed):**
     - **Gold-to-Silver Ratio (GSR) Radar:** Visual gauge showing current GSR, 15-year z-score, and regime swap signal.
     - **Physical vs. Paper Squeeze Gauge:** COMEX/SHFE registered inventory drawdown vs. paper open interest coverage ratio.
     - **Central Bank & Institutional Reserve Tracker:** Monthly sovereign accumulation & ETF tonnages.
     - **Live Financial News Impact Feed:** Real-time headlines tagged with NLP sentiment badges (`+1 Bullish`, `0 Neutral`, `-1 Bearish`), confidence scores, and estimated basis-point impact on the short-term curve.

---

## 6. What Is Left to the Builder

- Tuning hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, Fourier order $K$) so Walk-Forward validation achieves strong out-of-sample calibration and monotonic quantile bands.
- Crafting a unified start script (`run_app.sh` / `uvicorn backend.main:app`) so the FastAPI backend serves both the REST API (`/api/...`) and the Kigo Studio static site + Metal Predictor tool at a single local URL.
