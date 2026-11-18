from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_horizons_and_case_insensitive_metal():
    r = client.get("/api/predict/horizons?metal=XAU")
    assert r.status_code == 200
    data = r.json()
    assert data["metal"] == "gold"
    assert set(data["horizons"].keys()) == {"1M", "3M", "6M", "1Y", "5Y"}
    assert len(data["history"]) >= 60
    assert len(data["shap_drivers"]) >= 6

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
    r = client.post(
        "/api/simulate/macro-shock",
        json={
            "metal": "silver",
            "horizon": "1Y",
            "usdinr_rate": 98.0,
            "us_cpi_pct": 9.5,
            "rate_cuts_bps": 250,
            "real_yield_delta_bps": -150,
            "customs_duty_pct": 15.0,
            "aidc_pct": 2.5,
            "gst_pct": 3.0,
        },
    )
    assert r.status_code == 200
    sim = r.json()
    assert len(sim["stressed_series"]) > 5
    for pt in sim["stressed_series"]:
        assert pt["q025"] < pt["q10"] < pt["q50"] < pt["q90"] < pt["q975"]
    assert sim["stressed_parity"]["inr_per_1kg"] > 0


def test_backtest_replay_and_pulse_overview():
    r_bt = client.get("/api/backtest/replay?metal=gold&fold_index=2")
    assert r_bt.status_code == 200
    assert len(r_bt.json()["predicted_series"]) > 10

    r_pulse = client.get("/api/pulse/overview?metal=gold")
    assert r_pulse.status_code == 200
    p = r_pulse.json()
    assert "gsr_radar" in p and "physical_squeeze" in p and "news" in p
