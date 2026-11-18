import numpy as np
import pandas as pd
from data_pipeline.ingest_market_data import ingest_all_market_data
from data_pipeline.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_frame,
    compute_calendar_catalysts,
    populate_feature_store,
)


def test_ingestion_and_feature_no_lookahead(tmp_path):
    db_file = str(tmp_path / "test_metal.duckdb")
    frames = ingest_all_market_data(db_path=db_file, force_refresh=True)
    assert set(frames.keys()) == {"gold", "silver", "platinum", "copper"}
    for metal, df in frames.items():
        assert len(df) >= 3500
        assert {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dxy",
            "usdinr",
            "us10y",
            "real_yield",
            "crude",
            "sp500",
            "gsr",
        }.issubset(df.columns)

    # Verify zero lookahead leakage: mutating future prices after index 2000 must not alter features at index 2000
    gold_df = frames["gold"].copy().reset_index(drop=True)
    feat_full = build_feature_frame(gold_df, "gold")
    gold_mutated = gold_df.copy()
    gold_mutated.loc[2001:, "close"] *= 2.5
    gold_mutated.loc[2001:, "high"] *= 2.5
    gold_mutated.loc[2001:, "low"] *= 2.5
    feat_mutated = build_feature_frame(gold_mutated, "gold")

    row_orig = feat_full.loc[2000, FEATURE_COLUMNS].to_numpy(dtype=float)
    row_mut = feat_mutated.loc[2000, FEATURE_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_allclose(row_orig, row_mut, rtol=1e-7)
    assert len(FEATURE_COLUMNS) >= 35

    stored = populate_feature_store(db_path=db_file)
    assert set(stored.keys()) == {"gold", "silver", "platinum", "copper"}


def test_calendar_catalysts_detects_festive_and_fomc():
    cat = compute_calendar_catalysts(pd.Timestamp("2027-10-29"))  # Late Oct Dhanteras/Diwali window
    assert cat["is_indian_festive"] is True
    assert cat["wedding_season_multiplier"] > 1.0
    assert "volatility_cone_multiplier" in cat
