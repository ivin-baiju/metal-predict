import pytest
from data_pipeline.feature_engineering import populate_feature_store
from ml_engine.train_walk_forward import train_and_register_all, load_trained_suite
from ml_engine.explainability import predict_specific_date, compute_shap_drivers
from ml_engine.nlp_sentiment import get_live_news_pulse
from ml_engine.parity_engine import compute_comex_mcx_parity


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


def test_any_day_routing_tiers_and_validation(tmp_path):
    db_file = str(tmp_path / "ml_test_anyday.duckdb")
    reg_dir = str(tmp_path / "registry_anyday")
    populate_feature_store(db_path=db_file)
    train_and_register_all(db_path=db_file, registry_dir=reg_dir)
    suite = load_trained_suite(registry_dir=reg_dir)

    d_short = predict_specific_date(suite, "gold", "2026-12-15")   # <= 90d -> Tier 1
    assert d_short["tier"] == "Tier 1: Short-Term TFT/XGBoost Matrix"
    assert d_short["q025"] < d_short["q10"] < d_short["q50"] < d_short["q90"] < d_short["q975"]
    assert len(d_short["shap_drivers"]) >= 6
    assert "gtt_zone" in d_short

    d_mid = predict_specific_date(suite, "silver", "2027-06-15")   # 91-365d -> Tier 2
    assert d_mid["tier"] == "Tier 2: Medium-Term N-BEATS/LightGBM Basis"

    d_long = predict_specific_date(suite, "gold", "2029-11-18")    # > 365d -> Tier 3
    assert d_long["tier"] == "Tier 3: Long-Term VECM + 1,000-Path Monte Carlo"

    shap_bars = compute_shap_drivers(suite, "gold", 30)
    assert len(shap_bars) >= 6

    with pytest.raises(ValueError, match="future date"):
        predict_specific_date(suite, "gold", "2025-01-01")

    with pytest.raises(ValueError, match="7-year"):
        predict_specific_date(suite, "gold", "2036-01-01")


def test_parity_and_nlp_pulse():
    parity = compute_comex_mcx_parity(
        "gold",
        spot_usd=4100.0,
        usdinr=86.5,
        bcd_pct=6.0,
        aidc_pct=1.0,
        gst_pct=3.0,
    )
    assert parity["inr_per_10g_24k"] > 100000
    assert parity["inr_per_10g_22k"] < parity["inr_per_10g_24k"]
    assert parity["etf"]["gold_bees_nav"] > 50

    parity_zero_duty = compute_comex_mcx_parity(
        "gold",
        spot_usd=4100.0,
        usdinr=86.5,
        bcd_pct=0.0,
        aidc_pct=0.0,
        gst_pct=0.0,
    )
    assert parity_zero_duty["inr_per_10g_24k"] < parity["inr_per_10g_24k"]

    pulse = get_live_news_pulse("gold")
    assert len(pulse["headlines"]) >= 6
    assert -1.0 <= pulse["composite_sentiment"] <= 1.0
