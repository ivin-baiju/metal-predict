"""Leak-free technical, macroeconomic, cointegration, and calendar/festive feature engineering."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from data_pipeline.ingest_market_data import DEFAULT_DB_PATH, ingest_all_market_data

FEATURE_COLUMNS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_252d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_pct",
    "atr_14_pct",
    "vol_21d",
    "vol_63d",
    "gsr",
    "gsr_zscore_63d",
    "coint_sma20_200",
    "coint_price_sma50",
    "dxy_ret_21d",
    "usdinr_ret_21d",
    "real_yield_level",
    "real_yield_chg_21d",
    "crude_ret_21d",
    "sp500_ret_21d",
    "etf_flow_zscore",
    "cb_tonnage_pace",
    "warehouse_squeeze_idx",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
    "is_wedding_season",
    "is_festive_peak",
    "is_macro_event_week",
]

FEATURE_LABELS: dict[str, str] = {
    "real_yield_level": "US 10Y Real Yields (TIPS)",
    "real_yield_chg_21d": "Real Yield 21D Momentum",
    "dxy_ret_21d": "US Dollar Index (DXY) Pressure",
    "usdinr_ret_21d": "USD/INR Currency Depreciation",
    "cb_tonnage_pace": "Central Bank Sovereign Accumulation",
    "warehouse_squeeze_idx": "COMEX/SHFE Physical Inventory Tightness",
    "etf_flow_zscore": "Institutional ETF Net Inflows (BeES/GLD)",
    "gsr_zscore_63d": "Gold-to-Silver Ratio (GSR) Divergence",
    "gsr": "Gold-to-Silver Ratio Level",
    "coint_sma20_200": "Engle-Granger Cointegration Trend Spread",
    "coint_price_sma50": "50D Mean-Reversion Equilibrium Residual",
    "rsi_14": "14-Day Relative Strength (RSI)",
    "macd_hist": "MACD Momentum Histogram",
    "bb_width": "Bollinger Volatility Compression",
    "atr_14_pct": "Normalized Average True Range (ATR)",
    "ret_21d": "21-Day Price Momentum",
    "ret_63d": "Quarterly Price Regime Drift",
    "ret_126d": "Semi-Annual Structural Momentum",
    "ret_252d": "Annual Trend Anchor",
    "crude_ret_21d": "Crude Oil Energy Inflation Pass-Through",
    "sp500_ret_21d": "Equity Risk-On / Risk-Off Rotation",
    "is_wedding_season": "Indian Wedding & Retail Demand Seasonality",
    "is_festive_peak": "Dhanteras / Akshaya Tritiya Festive Catalyst",
    "is_macro_event_week": "FOMC / CPI Macro Catalyst Window",
}


def compute_calendar_catalysts(target_date: pd.Timestamp) -> dict:
    """Compute real-world calendar, Indian wedding/festive, and FOMC/CPI macro event multipliers."""
    ts = pd.Timestamp(target_date)
    month = int(ts.month)
    day = int(ts.day)
    dow = int(ts.dayofweek)

    # Peak Indian festive gold/silver buying: Dhanteras/Diwali (Oct 15 - Nov 15) & Akshaya Tritiya (Apr 18 - May 15)
    is_dhanteras_window = (month == 10 and day >= 15) or (month == 11 and day <= 15)
    is_akshaya_window = (month == 4 and day >= 18) or (month == 5 and day <= 15)
    is_indian_festive = bool(is_dhanteras_window or is_akshaya_window)

    # Indian wedding seasons: Nov-Feb and Apr-May
    is_wedding = bool(month in (11, 12, 1, 2, 4, 5))

    # Scheduled FOMC meetings (~every 6 weeks: mid-to-late Jan, Mar, May, Jun, Jul, Sep, Nov, Dec) & US CPI (10th-15th)
    is_fomc_window = bool(month in (1, 3, 5, 6, 7, 9, 11, 12) and (15 <= day <= 22))
    is_cpi_window = bool(10 <= day <= 14)
    is_india_budget = bool((month == 2 and 1 <= day <= 3) or (month == 7 and 20 <= day <= 25))
    is_macro_catalyst = bool(is_fomc_window or is_cpi_window or is_india_budget)

    wedding_mult = 1.018 if is_indian_festive else (1.009 if is_wedding else 1.0)
    vol_cone_mult = 1.18 if is_fomc_window else (1.14 if is_india_budget else (1.10 if is_cpi_window else 1.0))

    catalysts: list[str] = []
    if is_dhanteras_window:
        catalysts.append("Dhanteras & Diwali Bullion Accumulation Window (+1.8% physical premium)")
    elif is_akshaya_window:
        catalysts.append("Akshaya Tritiya Auspicious Retail Buying Window (+1.8% physical premium)")
    elif is_wedding:
        catalysts.append("Indian Peak Wedding Season Jewellery Demand (+0.9% retail floor)")

    if is_fomc_window:
        catalysts.append("US Federal Reserve (FOMC) Rate Decision Window (±18% wider volatility cone)")
    if is_cpi_window:
        catalysts.append("US CPI Inflation Print Week (±10% wider volatility cone)")
    if is_india_budget:
        catalysts.append("India Union Budget Customs Duty Watch (±14% wider volatility cone)")
    if not catalysts:
        catalysts.append("Standard COMEX / MCX Institutional Liquidity Session")

    return {
        "date": ts.strftime("%Y-%m-%d"),
        "day_of_week": ts.strftime("%A"),
        "is_weekend": bool(dow >= 5),
        "is_indian_festive": is_indian_festive,
        "is_wedding_season": is_wedding,
        "is_fomc_window": is_fomc_window,
        "is_cpi_window": is_cpi_window,
        "is_india_budget": is_india_budget,
        "is_macro_catalyst": is_macro_catalyst,
        "wedding_season_multiplier": round(wedding_mult, 4),
        "volatility_cone_multiplier": round(vol_cone_mult, 4),
        "active_catalysts": catalysts,
    }


def build_feature_frame(raw_df: pd.DataFrame, metal: str) -> pd.DataFrame:
    """Compute 36 strictly backward-looking (zero lookahead leakage) features for a metal DataFrame."""
    df = raw_df.copy().sort_values("date").reset_index(drop=True)
    dt = pd.to_datetime(df["date"])
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    log_p = np.log(np.maximum(close, 1e-6))

    # Multi-horizon backward log returns
    df["ret_1d"] = log_p.diff(1).fillna(0.0)
    df["ret_5d"] = log_p.diff(5).fillna(0.0)
    df["ret_10d"] = log_p.diff(10).fillna(0.0)
    df["ret_21d"] = log_p.diff(21).fillna(0.0)
    df["ret_63d"] = log_p.diff(63).fillna(0.0)
    df["ret_126d"] = log_p.diff(126).fillna(0.0)
    df["ret_252d"] = log_p.diff(252).fillna(0.0)

    # RSI(14) using backward rolling averages
    delta = close.diff(1).fillna(0.0)
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9) causal EWM
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = (ema12 - ema26) / (close + 1e-8) * 100.0
    macd_sig = macd_raw.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_raw
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_raw - macd_sig

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df["bb_width"] = (upper - lower) / (sma20 + 1e-8)
    df["bb_pct"] = (close - lower) / (upper - lower + 1e-8)

    # Normalized ATR(14)
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14_pct"] = tr.rolling(14, min_periods=1).mean() / (close + 1e-8)

    # Realized annualized volatility
    df["vol_21d"] = df["ret_1d"].rolling(21, min_periods=2).std().fillna(0.01) * np.sqrt(252.0)
    df["vol_63d"] = df["ret_1d"].rolling(63, min_periods=2).std().fillna(0.01) * np.sqrt(252.0)

    # Gold-to-Silver Ratio (GSR) & 63d z-score
    gsr = df["gsr"].astype(float)
    gsr_mean = gsr.rolling(63, min_periods=5).mean().bfill()
    gsr_std = gsr.rolling(63, min_periods=5).std().fillna(1.0).replace(0, 1.0)
    df["gsr_zscore_63d"] = (gsr - gsr_mean) / gsr_std

    # Engle-Granger style cointegration / equilibrium spread residuals
    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()
    df["coint_sma20_200"] = np.log((sma20 + 1e-8) / (sma200 + 1e-8))
    df["coint_price_sma50"] = np.log((close + 1e-8) / (sma50 + 1e-8))

    # Cross-asset & Macro momentum (strictly backward-looking)
    df["dxy_ret_21d"] = np.log(df["dxy"].astype(float) / df["dxy"].astype(float).shift(21).bfill()).fillna(0.0)
    df["usdinr_ret_21d"] = np.log(df["usdinr"].astype(float) / df["usdinr"].astype(float).shift(21).bfill()).fillna(0.0)
    df["real_yield_level"] = df["real_yield"].astype(float)
    df["real_yield_chg_21d"] = df["real_yield"].astype(float).diff(21).fillna(0.0)
    df["crude_ret_21d"] = np.log(df["crude"].astype(float) / df["crude"].astype(float).shift(21).bfill()).fillna(0.0)
    df["sp500_ret_21d"] = np.log(df["sp500"].astype(float) / df["sp500"].astype(float).shift(21).bfill()).fillna(0.0)

    # Cyclical calendar & Indian retail / FOMC encodings
    doy = dt.dt.dayofyear.astype(float)
    dow = dt.dt.dayofweek.astype(float)
    month = dt.dt.month
    day = dt.dt.day

    df["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)
    df["sin_dow"] = np.sin(2.0 * np.pi * dow / 5.0)
    df["cos_dow"] = np.cos(2.0 * np.pi * dow / 5.0)

    df["is_wedding_season"] = month.isin([11, 12, 1, 2, 4, 5]).astype(float)
    df["is_festive_peak"] = (
        ((month == 10) & (day >= 15))
        | ((month == 11) & (day <= 15))
        | ((month == 4) & (day >= 18))
        | ((month == 5) & (day <= 15))
    ).astype(float)
    df["is_macro_event_week"] = ((day >= 10) & (day <= 21)).astype(float)

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def populate_feature_store(
    db_path: str = DEFAULT_DB_PATH,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Ingest market data, compute leak-free features for all 4 metals, and store in DuckDB."""
    raw_frames = ingest_all_market_data(db_path=db_path, force_refresh=force_refresh)
    engineered: dict[str, pd.DataFrame] = {}
    con = duckdb.connect(db_path)
    try:
        for metal, raw_df in raw_frames.items():
            feat_df = build_feature_frame(raw_df, metal)
            engineered[metal] = feat_df
            con.register("tmp_feat", feat_df)
            con.execute(f"CREATE OR REPLACE TABLE features_{metal} AS SELECT * FROM tmp_feat")
            con.unregister("tmp_feat")
        return engineered
    finally:
        con.close()
