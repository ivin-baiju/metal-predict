CREATE TABLE IF NOT EXISTS raw_market_daily (
    metal VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    dxy DOUBLE,
    usdinr DOUBLE,
    us10y DOUBLE,
    real_yield DOUBLE,
    crude DOUBLE,
    sp500 DOUBLE,
    gsr DOUBLE,
    etf_flow_zscore DOUBLE,
    cb_tonnage_pace DOUBLE,
    warehouse_squeeze_idx DOUBLE,
    PRIMARY KEY (metal, date)
);

CREATE TABLE IF NOT EXISTS walk_forward_metrics (
    metal VARCHAR,
    horizon VARCHAR,
    rmse DOUBLE,
    mape DOUBLE,
    pinball_loss DOUBLE,
    directional_accuracy DOUBLE,
    coverage_80 DOUBLE,
    coverage_95 DOUBLE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (metal, horizon)
);
