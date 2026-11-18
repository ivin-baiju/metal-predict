import re
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
    assert "regime_probabilities" in data["horizons"]["1M"]
    assert "regime_probabilities" in data["horizons"]["5Y"]

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
    assert "stressed_regime_probabilities" in sim


def test_backtest_replay_and_pulse_overview():
    r_bt = client.get("/api/backtest/replay?metal=gold&fold_index=2")
    assert r_bt.status_code == 200
    assert len(r_bt.json()["predicted_series"]) > 10

    r_pulse = client.get("/api/pulse/overview?metal=gold")
    assert r_pulse.status_code == 200
    p = r_pulse.json()
    assert "gsr_radar" in p and "physical_squeeze" in p and "news" in p
    assert len(p["correlations"]) >= 6
    assert len(p["arbitrage_hubs"]) == 5

    r_sip = client.post(
        "/api/simulate/portfolio-sip",
        json={
            "monthly_sip": 25000,
            "lump_sum": 100000,
            "horizon_years": 3,
            "currency": "INR",
            "gold_weight": 50,
            "silver_weight": 25,
            "platinum_weight": 10,
            "copper_weight": 15,
        },
    )
    assert r_sip.status_code == 200
    sip_body = r_sip.json()
    assert sip_body["projected_conservative_q10"] < sip_body["projected_median_q50"] < sip_body["projected_bull_q90"]


def test_metal_predict_branding_skeleton_and_no_kigo():
    # Health check branding
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert "Metal Predict" in r_health.json()["engine"]
    assert "kigo" not in r_health.text.lower()

    # Main Predictor page at both / and /tools/metal-price-predictor.html
    for path in ["/", "/tools/metal-price-predictor.html"]:
        r_page = client.get(path)
        assert r_page.status_code == 200
        html = r_page.text
        assert "Metal Predict" in html
        assert "mpPreloader" in html
        assert "mp-skeleton" in html
        assert "contact-form-card" in html
        assert re.search(r"kigo", html, re.IGNORECASE) is None

    # Tools Hub page
    r_hub = client.get("/tools.html")
    assert r_hub.status_code == 200
    assert "Metal Predict" in r_hub.text
    assert "metal-price-predictor.html" in r_hub.text
    assert re.search(r"kigo", r_hub.text, re.IGNORECASE) is None


def test_ai_chatbot_multilingual_and_live_clock():
    # Verify EN, JA, and HI chatbot responses across intents
    for lang, msg in [
        ("en", "What is the 1Y forecast for Gold?"),
        ("ja", "金 (ゴールド) のMCXパリティ価格を教えて"),
        ("hi", "चांदी (Silver) की कीमत क्यों बढ़ रही है? SHAP बताएं"),
        ("en", "Predict Gold price on 2027-10-29"),
    ]:
        r_chat = client.post(
            "/api/chat/ask",
            json={"message": msg, "metal": "gold", "horizon": "1M", "lang": lang, "usdinr": 86.48},
        )
        assert r_chat.status_code == 200
        body = r_chat.json()
        assert body["lang"] == lang
        assert len(body["answer"]) > 30
        assert "telemetry" in body

    # Verify Live Clock, Language Switcher, Bespoke Logo & AI Chatbot exist in HTML
    r_page = client.get("/")
    html = r_page.text
    assert 'id="mpLiveClock"' in html
    assert 'data-lang="ja"' in html
    assert 'data-lang="hi"' in html
    assert 'id="mpChatbotWidget"' in html
    assert "QUANTITATIVE BULLION AI" in html
