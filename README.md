---
title: Metal Predict
emoji: ⛩️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 8000
pinned: true
short_description: Multi-Horizon Commodity Price Forecasting & Analytics Desk
---

# Metal Predict — Multi-Horizon Commodity Price Forecasting & Analytics Desk

Three-Tier Hybrid Machine Learning Engine (**XGBoost Quantile Ensemble** + **LightGBM Seasonal Basis** + **1,000-Path VECM Jump-Diffusion Monte Carlo**) for **Gold (`XAU/USD`), Silver (`XAG/USD`), Platinum (`XPT/USD`), and Copper (`HG/USD`)** across **1M, 3M, 6M, 1Y, 5Y, and Any Specific Calendar Day**.

## Core Capabilities
- **Multi-Horizon Probability Fan Charts:** 80% (`Q10–Q90`) and 95% (`Q025–Q975`) calibrated quantile cones with interactive historical OHLC & future trajectory crosshair inspection.
- **Calendar-Aware "Predict Any Specific Day" Router:** Automatically routes $\Delta t \le 90\text{d}$ to Tier 1, $91\text{–}365\text{d}$ to Tier 2, and $>365\text{d}$ to Tier 3 with FOMC, Union Budget, Dhanteras, and Akshaya Tritiya catalyst adjustments.
- **TreeSHAP Explainability & Macro/Duty Stress Lab:** Exact Shapley feature attributions (`shap.TreeExplainer`) + live counterfactual sliders (`USD/INR`, `US CPI`, `Fed/RBI Rate Cuts`, `10Y Real Yields`, `India Basic Customs Duty`, `GST`).
- **COMEX-to-MCX Landed Parity & ETF Optimizer:** Real-time `INR / 10g (24K)`, `INR / 10g (22K)`, `INR / 1kg`, Gold/Silver BeES ETF NAVs, and GTT limit-buy zones.
- **Multi-Metal SIP & Wealth Simulator, Cross-Asset Correlation Matrix & 5-Hub Global Bullion Arbitrage.**
- **Multilingual UI & Grounded AI Analyst Chatbot:** Instant switching between **English (`EN`)**, **Japanese (`日本語`)**, and **Hindi (`हिन्दी`)**.
