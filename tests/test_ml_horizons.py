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
