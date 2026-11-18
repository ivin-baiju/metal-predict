/**
 * Metal Predict — Multi-Horizon Commodity Price Forecasting & Analytics Desk
 * Self-contained frontend controller:
 * - Slow, choreographed Washi-Paper Startup Preloader & Shimmer Skeletons
 * - High-DPI Canvas Probability Fan & Walk-Forward Backtest Time Machine
 * - Any-Day Precision Router, TreeSHAP Attribution, Macro/Duty Stress Lab
 * - Multi-Metal SIP & Portfolio Wealth Simulator, Cross-Asset Correlation Matrix, 5-Hub Global Arbitrage
 */

(function () {
  const state = {
    metal: "gold",
    horizon: "1M",
    lang: "en", // en | ja | hi
    unitMode: "USD_OZ", // USD_OZ | INR_10G_24K | INR_10G_22K | INR_1KG | ETF_NAV
    usdinr: 86.48,
    cpi: 2.7,
    rateCutsBps: 50,
    realYieldDeltaBps: 0,
    bcdPct: 6.0,
    aidcPct: 1.0,
    gstPct: 3.0,
    horizonPayload: null,
    stressedPayload: null,
    anyDayPayload: null,
    pulsePayload: null,
    backtestPayload: null,
    foldIndex: 0,
    fanHover: null,
    backtestHover: null,
    fanAnimProgress: 1.0,
    fanAnimFrame: null,
    btAnimProgress: 1.0,
    btAnimFrame: null,
    stressDebounceTimer: null,
    stressSeq: 0,
    initialLoadDone: false,
  };

  const I18N_DICT = {
    en: {
      rotatingWords: ["forecasts", "quantiles", "regimes", "parity", "conviction"],
      metalNames: { gold: "Gold", silver: "Silver", platinum: "Platinum", copper: "Copper" },
      preloader_skip: "Enter Desk Immediately →",
      live_session_badge: "COMEX • LBMA • MCX LIVE",
      lang_label: "Language:",
      nav_desk: "Forecast Desk",
      nav_anyday: "Any-Day Router",
      nav_stress: "Stress Lab",
      nav_sip: "SIP & Wealth",
      nav_backtest: "Backtest Replay",
      nav_faq: "Methodology",
      nav_pulse_cta: "Real-World Pulse",
      hero_eyebrow: "Metal Predict • Three-Tier Hybrid ML Engine (XGBoost + LightGBM Basis + VECM Monte Carlo)",
      hero_title_1: "Metal Predict",
      hero_title_2: "Multi-Horizon",
      hero_lead:
        "Direct multi-horizon probability fan projections across 1M, 3M, 6M, 1Y, 5Y and Any Specific Calendar Day—fused with exact TreeSHAP attributions, FinBERT breaking-news sentiment, COMEX-to-MCX landed customs parity, and Walk-Forward backtest replay.",
      quote_unit_label: "Quote Unit:",
      unit_usd_oz: "USD / oz (COMEX)",
      unit_inr_24k: "INR / 10g (24K MCX)",
      unit_inr_22k: "INR / 10g (22K Retail)",
      unit_inr_1kg: "INR / 1kg",
      unit_etf: "Gold / Silver BeES ETF",
      kpi_spot_label: "Live Global Spot Benchmark",
      kpi_mcx_label: "Indian Landed MCX Parity",
      kpi_gtt_label: "Optimal GTT Limit-Buy Zone",
      legend_ohlc: "Historical OHLC Candles",
      legend_q50: "Median Forecast (Q50)",
      legend_80: "80% Confidence Cone (Q10–Q90)",
      legend_95: "95% Confidence Cone (Q025–Q975)",
      legend_mc: "5Y Monte Carlo Paths",
      anyday_eyebrow: "Calendar-Aware Δt Router",
      anyday_title: '"Predict Any Specific Day" Inspector',
      anyday_sub: "Routes Δt ≤ 90d to TFT/XGBoost, 91–365d to LightGBM Basis, and >365d to 1,000-Path VECM Monte Carlo.",
      anyday_date_label: "Target Calendar Date",
      anyday_btn: "Inspect Date",
      preset_fomc: "Next FOMC (Dec 18, 2026)",
      preset_budget: "India Budget (Feb 1, 2027)",
      preset_akshaya: "Akshaya Tritiya (May 2027)",
      preset_dhanteras: "Dhanteras Peak (Oct 2027)",
      preset_3y: "3Y Macro Cycle (Nov 2029)",
      anyday_80: "80% Quantile Cone (Q10–Q90)",
      anyday_95: "95% Volatility Cone (Q025–Q975)",
      anyday_mcx: "Projected MCX Landed Parity",
      anyday_etf: "Projected ETF NAV Equivalent",
      gtt_box_title: "ETF & GTT Limit-Buy Order Optimizer",
      gtt_acc: "Accumulate Band",
      gtt_target: "Target Profit Exit",
      gtt_stop: "Invalidation Stop",
      stress_eyebrow: "Model Transparency & Counterfactual Perturbation",
      stress_title: "Why This <em>price</em> & What-If Stress Lab",
      shap_card_title: 'TreeSHAP + FinBERT Feature Attribution ("Why This Price?")',
      shap_card_sub: "Exact Shapley additive weights extracted from the active horizon booster fused with live news sentiment.",
      regime_title: "Active Horizon Macro Regime Probabilities",
      stress_card_title: "Interactive Macro & Duty Stress-Tester",
      stress_card_sub: "Perturb currency, US inflation, central bank rate cuts, real yields, and Indian import duties to shift the forecast cone live.",
      reset_baseline: "Reset Baseline",
      slider_usdinr: "USD / INR Exchange Rate",
      slider_cpi: "US CPI Inflation (%)",
      slider_cuts: "Fed / RBI Rate Cuts (bps)",
      slider_realyield: "10Y TIPS Real Yield Shock",
      slider_duty: "India Basic Customs Duty (%)",
      slider_gst: "Bullion GST Rate (%)",
      stress_base_label: "Baseline Horizon Median",
      stress_new_label: "Stressed Horizon Median",
      stress_delta_label: "Net Shock Impulse",
      parity_eyebrow: "Cross-Exchange Arbitrage & Retail Bullion Conversion",
      parity_title: "COMEX-to-MCX <em>parity</em> & ETF Ledger",
      sip_eyebrow: "Systematic Accumulation & Basket Optimization",
      sip_title: "Multi-Metal SIP & <em>wealth</em> Simulator",
      sip_preset_1: "Sovereign Safe-Haven (70/20/5/5)",
      sip_preset_2: "Green Grid & Solar (25/40/10/25)",
      sip_preset_3: "All-Weather 4-Metal (40/30/15/15)",
      sip_left_title: "Capital & Allocation Weights",
      sip_monthly_label: "Monthly SIP (₹ INR)",
      sip_lump_label: "Initial Lump-Sum (₹ INR)",
      sip_years_label: "Investment Horizon (Years)",
      sip_w_gold: "Gold (XAU) Weight %",
      sip_w_silver: "Silver (XAG) Weight %",
      sip_w_ptcu: "Platinum / Copper %",
      sip_right_title: "Projected Multi-Horizon Portfolio Terminal Value",
      sip_invested_label: "Total Principal Invested",
      sip_q50_label: "Expected Median Corpus (Q50)",
      sip_q90_label: "90th Percentile Bullish Supercycle Case (Q90)",
      quant_eyebrow: "Cross-Market Cointegration & Physical Exchange Spreads",
      quant_title: "Correlation <em>matrix</em> & Global Arbitrage",
      corr_card_title: "Rolling Cross-Asset Correlation Matrix (30D vs 1Y)",
      arb_card_title: "5-Hub Global Bullion Arbitrage Scanner (COMEX • LBMA • MCX • SHFE • TOCOM)",
      arb_th_hub: "Exchange Hub",
      arb_th_quote: "Live Benchmark",
      arb_th_spread: "Arbitrage Spread",
      arb_th_regime: "Physical Flow Regime",
      bt_eyebrow: "Chronological Expanding-Window Validation (Zero Lookahead Leakage)",
      bt_title: "Walk-Forward <em>backtest</em> Time Machine",
      bt_leg_train: "Pre-Cutoff Training Window",
      bt_leg_actual: "Actual Realized Future Price",
      bt_leg_pred: "Out-of-Sample Model Forecast (Q50)",
      bt_leg_cone: "Out-of-Sample 80% Cone",
      pulse_eyebrow: "Multimodal NLP & Physical Supply-Chain Telemetry",
      pulse_title: "Real-World <em>pulse</em> & FinBERT News Feed",
      pulse_gsr_label: "Gold-to-Silver Ratio (GSR) Radar",
      pulse_sqz_label: "Physical vs. Paper Squeeze Gauge",
      pulse_cb_label: "Sovereign & Institutional Reserve Tracker",
      faq_eyebrow: "Architecture & Execution Playbook",
      faq_title: "Model <em>methodology</em> & FAQ",
      faq_q1: "Why does Metal Predict use a 3-Tier Direct Multi-Horizon Ensemble instead of recursive rollout?",
      faq_a1:
        "Standard autoregressive models predict day t+1 and feed it back into t+2, which compounds noise until the curve flatlines after 30–60 days. Metal Predict trains direct horizon heads: Tier 1 (1–90d) uses direct XGBoost Quantile Regressors, Tier 2 (91–365d) pairs polynomial/Fourier seasonal basis expansion with LightGBM quantile residuals, and Tier 3 (1Y–5Y) runs a 1,000-path Jump-Diffusion + Ornstein-Uhlenbeck cointegrated Monte Carlo simulator.",
      faq_q2: 'How does the "Predict Any Specific Day" calendar router adjust for FOMC meetings and Dhanteras?',
      faq_a2:
        "When you pick any future calendar date, the backend calculates Δt = Target Date − Today, selects the active model tier, and inspects the calendar catalyst matrix. Dates coinciding with FOMC rate decisions, US CPI prints, or India's Union Budget automatically widen the 80% and 95% volatility cones (+10% to +18%), while peak Indian retail buying windows (Dhanteras, Akshaya Tritiya, and Nov–Feb / Apr–May wedding seasons) apply empirical physical demand multipliers.",
      faq_q3: "How are COMEX-to-MCX landed parity prices and Gold/Silver BeES ETF trigger zones calculated?",
      faq_a3:
        "Global COMEX spot prices (USD/oz for precious metals, USD/lb for copper) are converted into Indian landed costs using live USD/INR, Basic Customs Duty (BCD), Agriculture Infrastructure & Development Cess (AIDC), local MCX liquidity spread, and 3% GST. GTT (Good-Till-Triggered) limit-buy zones are derived from the model's 10th-to-35th percentile pullback support band.",
      faq_q4: "How is lookahead data leakage prevented during Walk-Forward Cross-Validation?",
      faq_a4:
        "All 36 technical, macroeconomic, cointegration, and seasonal features at date t are computed strictly from historical observations at ≤ t. Expanding-window Walk-Forward folds train strictly on data prior to each historical cutoff date before generating out-of-sample projections.",
      footer_headline: "Metal Predict multi-horizon AI, built for <em>conviction</em>",
      footer_meta_1: "<strong>Engine:</strong> Three-Tier Quantile Ensemble (XGBoost + LightGBM + VECM Monte Carlo)",
      footer_meta_2: "<strong>Feature Store:</strong> DuckDB 36-Signal Multimodal Pipeline",
      footer_meta_3: "<strong>Explainability:</strong> TreeSHAP + FinBERT News Sentiment",
      chat_header_title: "Metal Predict AI Analyst",
      chat_chip_1: "1Y Forecast & Buy Zone",
      chat_chip_2: "MCX 24K/22K Parity",
      chat_chip_3: "Why This Price? (SHAP)",
      chat_chip_4: "Predict 2027-10-29",
      chat_send_btn: "Ask 押",
      chat_fab_title: "AI Analyst • Ask Metal Predict",
      chat_fab_sub: "Live Quantile, MCX & SHAP Q&A",
      chat_placeholder: "Ask about Gold, Silver, MCX, SHAP, or any date...",
      chat_welcome:
        "<strong>Metal Predict AI • Quantitative Bullion Desk</strong>\nAsk me about multi-horizon forecasts (1M–5Y), exact future calendar dates (YYYY-MM-DD), Indian MCX 24K/22K parity, TreeSHAP price drivers, or GTT buy zones!",
      fan_title_suffix: "Probability Fan",
      kpi_horizon_suffix: "Median Projection",
    },
    ja: {
      rotatingWords: ["未来予測", "分位推計", "市場均衡", "確信", "価格解析"],
      metalNames: { gold: "金 (ゴールド)", silver: "銀 (シルバー)", platinum: "白金 (プラチナ)", copper: "銅 (カッパー)" },
      preloader_skip: "すぐに分析デスクを開く →",
      live_session_badge: "COMEX • LBMA • MCX リアルタイム稼働中",
      lang_label: "表示言語:",
      nav_desk: "予測デスク",
      nav_anyday: "指定日予測ルーター",
      nav_stress: "ストレステスト",
      nav_sip: "積立・資産試算",
      nav_backtest: "バックテスト検証",
      nav_faq: "モデル方法論",
      nav_pulse_cta: "市場パルス・ニュース",
      hero_eyebrow: "Metal Predict • 3層ハイブリッド機械学習エンジン (XGBoost + LightGBM Basis + VECM モンテカルロ)",
      hero_title_1: "Metal Predict",
      hero_title_2: "マルチホライズン",
      hero_lead:
        "1ヶ月・3ヶ月・6ヶ月・1年・5年、および任意の将来カレンダー日を直接予測するマルチホライズン確率ファンモデル。TreeSHAP寄与度分析、FinBERTニュース感情解析、インドMCX輸入パリティ、Walk-Forwardバックテストを統合。",
      quote_unit_label: "表示単位:",
      unit_usd_oz: "USD / oz (COMEX)",
      unit_inr_24k: "INR / 10g (24K MCX)",
      unit_inr_22k: "INR / 10g (22K 宝飾)",
      unit_inr_1kg: "INR / 1kg",
      unit_etf: "Gold / Silver BeES ETF",
      kpi_spot_label: "グローバル・スポット基準価格",
      kpi_mcx_label: "インドMCX輸入パリティ価格",
      kpi_gtt_label: "推奨GTT指値買いゾーン",
      legend_ohlc: "過去OHLCローソク足",
      legend_q50: "予測中央値 (Q50)",
      legend_80: "80% 信頼区間コーン (Q10–Q90)",
      legend_95: "95% 信頼区間コーン (Q025–Q975)",
      legend_mc: "5年モンテカルロ軌道",
      anyday_eyebrow: "カレンダー触媒対応 Δt ルーター",
      anyday_title: "「任意の日付」ピンポイント価格予測",
      anyday_sub: "Δt ≤ 90日はTFT/XGBoost、91〜365日はLightGBM基底展開、365日超は1,000経路VECMモンテカルロへ自動ルーティング。",
      anyday_date_label: "予測対象カレンダー日",
      anyday_btn: "日付を予測",
      preset_fomc: "次回FOMC (2026年12月18日)",
      preset_budget: "インド国家予算 (2027年2月1日)",
      preset_akshaya: "アクシャヤ・トリティヤ (2027年5月)",
      preset_dhanteras: "ダンテラス繁忙期 (2027年10月)",
      preset_3y: "3年マクロサイクル (2029年11月)",
      anyday_80: "80% 分位予測コーン (Q10–Q90)",
      anyday_95: "95% ボラティリティコーン (Q025–Q975)",
      anyday_mcx: "予測MCX輸入パリティ (INR)",
      anyday_etf: "予測ETF基準価額 (NAV)",
      gtt_box_title: "ETF・GTT指値注文オプティマイザー",
      gtt_acc: "推奨押し目買い帯",
      gtt_target: "目標利確ライン",
      gtt_stop: "損切り撤退ライン",
      stress_eyebrow: "モデル透明性と反実仮想マクロ・シミュレーション",
      stress_title: "価格変動の<em>根拠 (SHAP)</em> & マクロストレステスト",
      shap_card_title: "TreeSHAP + FinBERT 特徴量寄与度（「なぜこの価格か？」）",
      shap_card_sub: "各ホライズンの勾配ブースティング木から抽出した厳密なShapley寄与度とニュース感情スコア。",
      regime_title: "現在ホライズンのマクロレジーム確率",
      stress_card_title: "インタラクティブ・マクロ＆関税ストレステスター",
      stress_card_sub: "為替レート、米国インフレ率、利下げ幅、実質金利、インド輸入関税を動かして予測コーンを即時再計算。",
      reset_baseline: "初期値に戻す",
      slider_usdinr: "USD / INR 為替レート",
      slider_cpi: "米国 CPI インフレ率 (%)",
      slider_cuts: "FRB / RBI 利下げ幅 (bps)",
      slider_realyield: "米10年実質金利ショック",
      slider_duty: "インド基本輸入関税 (%)",
      slider_gst: "貴金属 GST 税率 (%)",
      stress_base_label: "ベースライン予測中央値",
      stress_new_label: "ストレス適用後中央値",
      stress_delta_label: "純ショック変動率",
      parity_eyebrow: "国際取引所裁定スプレッド＆リテール換算",
      parity_title: "COMEX–MCX <em>価格パリティ</em> & ETF台帳",
      sip_eyebrow: "定額積立（SIP）＆マルチメタル・バスケット最適化",
      sip_title: "マルチメタル積立 (SIP) & <em>資産形成</em> シミュレーター",
      sip_preset_1: "ソブリン安全資産型 (70/20/5/5)",
      sip_preset_2: "グリーン電力・太陽光型 (25/40/10/25)",
      sip_preset_3: "全天候4金属バランス型 (40/30/15/15)",
      sip_left_title: "積立資金＆ポートフォリオ配分比率",
      sip_monthly_label: "毎月積立額 (₹ INR)",
      sip_lump_label: "初期一括投資額 (₹ INR)",
      sip_years_label: "投資期間 (年)",
      sip_w_gold: "金 (XAU) 配分比率 %",
      sip_w_silver: "銀 (XAG) 配分比率 %",
      sip_w_ptcu: "プラチナ / 銅 比率 %",
      sip_right_title: "マルチホライズン予測ポートフォリオ最終評価額",
      sip_invested_label: "投資元本合計",
      sip_q50_label: "期待中央値資産額 (Q50)",
      sip_q90_label: "上位90%強気スーパーサイクル (Q90)",
      quant_eyebrow: "市場間共和分＆世界の現物取引所スプレッド",
      quant_title: "クロスアセット<em>相関行列</em> & 国際裁定スキャナー",
      corr_card_title: "ローリング資産間相関マトリックス (30日 vs 1年)",
      arb_card_title: "世界5大取引所ブリオン裁定スキャナー (COMEX • LBMA • MCX • SHFE • TOCOM)",
      arb_th_hub: "取引所ハブ",
      arb_th_quote: "リアルタイム指標",
      arb_th_spread: "裁定スプレッド",
      arb_th_regime: "現物フロー状況",
      bt_eyebrow: "時系列拡大ウィンドウ交差検証（未来データ漏洩ゼロ）",
      bt_title: "Walk-Forward <em>バックテスト</em> タイムマシン",
      bt_leg_train: "カットオフ前の学習期間",
      bt_leg_actual: "実際の実現価格",
      bt_leg_pred: "アウトオブサンプル予測 (Q50)",
      bt_leg_cone: "アウトオブサンプル 80% コーン",
      pulse_eyebrow: "マルチモーダルNLP＆現物サプライチェーン監視",
      pulse_title: "リアルワールド・<em>パルス</em> & FinBERTニュース",
      pulse_gsr_label: "金銀比価 (GSR) レーダー",
      pulse_sqz_label: "現物 vs ペーパー在庫ひっ迫指数",
      pulse_cb_label: "各国中央銀行・機関投資家準備追跡",
      faq_eyebrow: "モデル構造＆クオンツ運用プレイブック",
      faq_title: "モデル<em>方法論</em> & よくある質問 (FAQ)",
      faq_q1: "なぜMetal Predictは再帰的ロールアウトではなく「3層ダイレクト・アンサンブル」を採用しているのですか？",
      faq_a1:
        "一般的な自己回帰モデルはt+1日の予測値をt+2日の入力に戻すため、30〜60日を超えると誤差が累積して予測線が平坦化します。Metal Predictは期間別に直接予測ヘッドを学習：第1層（1〜90日）はXGBoost分位回帰、第2層（91〜365日）はフーリエ季節基底＋LightGBM、第3層（1〜5年）は1,000経路のジャンプ拡散VECMモンテカルロを実行します。",
      faq_q2: "「任意の日付予測」ルーターはFOMC会合やインドの祭事（ダンテラス）をどのように反映しますか？",
      faq_a2:
        "指定された将来日付から Δt = 目標日 − 今日 を算出し、最適ティアを選択すると同時にカレンダー触媒行列を照会します。FOMC金利決定や米CPI発表、インド国家予算の日は80%・95%ボラティリティコーンを自動拡大し、ダンテラスや婚礼シーズンには現物需要プレミアムを適用します。",
      faq_q3: "COMEX–MCX輸入パリティ価格とGold/Silver BeES ETF指値ゾーンはどう計算されますか？",
      faq_a3:
        "COMEXスポット価格（貴金属はUSD/oz、銅はUSD/lb）にUSD/INR為替、基本関税(BCD)、農業インフラ目的税(AIDC)、MCX流動性スプレッド、3% GSTを適用してインド国内価格を算出します。GTT指値買いゾーンは予測分布の10〜35パーセンタイル押し目帯から導出されます。",
      faq_q4: "Walk-Forward交差検証において未来データのリーク（先読みバイアス）はどう防止されていますか？",
      faq_a4:
        "全36個のテクニカル・マクロ・共和分・季節性特徴量は時点 t 以前の観測値のみから計算されます。各バックテストFoldは過去のカットオフ日以前のデータのみで学習し、未知の将来期間に対してアウトオブサンプル予測を生成します。",
      footer_headline: "確信ある投資判断のための、Metal Predict <em>マルチホライズンAI</em>",
      footer_meta_1: "<strong>エンジン:</strong> 3層分位アンサンブル (XGBoost + LightGBM + VECM モンテカルロ)",
      footer_meta_2: "<strong>特徴量ストア:</strong> DuckDB 36シグナル・マルチモーダルパイプライン",
      footer_meta_3: "<strong>説明可能性:</strong> TreeSHAP + FinBERT ニュース感情解析",
      chat_header_title: "Metal Predict AI アナリスト",
      chat_chip_1: "1年予測と推奨買いゾーン",
      chat_chip_2: "MCX 24K/22K パリティ価格",
      chat_chip_3: "なぜ動く？ TreeSHAP要因",
      chat_chip_4: "2027-10-29 の価格を予測",
      chat_send_btn: "質問 押",
      chat_fab_title: "AI アナリスト • Metal Predictに質問",
      chat_fab_sub: "リアルタイム予測・MCX・SHAP応答",
      chat_placeholder: "金・銀・プラチナ・銅の予測や日付を入力...",
      chat_welcome:
        "<strong>Metal Predict AI • クオンツ貴金属デスク</strong>\n1ヶ月〜5年の価格予測、特定の日付（YYYY-MM-DD）、インドMCXパリティ、TreeSHAP変動要因、GTT推奨買いゾーンについて何でも質問してください！",
      fan_title_suffix: "確率ファン予測",
      kpi_horizon_suffix: "予測中央値",
    },
    hi: {
      rotatingWords: ["पूर्वानुमान", "प्रायिकता", "विश्वास", "मूल्य-समता", "विश्लेषण"],
      metalNames: { gold: "सोना (Gold)", silver: "चांदी (Silver)", platinum: "प्लैटिनम (Platinum)", copper: "तांबा (Copper)" },
      preloader_skip: "तुरंत एनालिटिक्स डेस्क खोलें →",
      live_session_badge: "COMEX • LBMA • MCX लाइव टेलीमेट्री",
      lang_label: "भाषा:",
      nav_desk: "पूर्वानुमान डेस्क",
      nav_anyday: "किसी भी दिन का मूल्य",
      nav_stress: "स्ट्रेस लैब",
      nav_sip: "SIP और संपत्ति",
      nav_backtest: "बैकटेस्ट रीप्ले",
      nav_faq: "कार्यप्रणाली",
      nav_pulse_cta: "मार्केट पल्स",
      hero_eyebrow: "Metal Predict • थ्री-टियर हाइब्रिड ML इंजन (XGBoost + LightGBM Basis + VECM मोंटे कार्लो)",
      hero_title_1: "Metal Predict",
      hero_title_2: "मल्टी-होराइजन",
      hero_lead:
        "1M, 3M, 6M, 1Y, 5Y और किसी भी विशिष्ट कैलेंडर दिन के लिए सटीक प्रायिकता फैन पूर्वानुमान—सटीक TreeSHAP व्याख्या, FinBERT समाचार भावना, COMEX-से-MCX भारतीय पैरिटी और Walk-Forward बैकटेस्ट के साथ।",
      quote_unit_label: "मूल्य इकाई:",
      unit_usd_oz: "USD / oz (COMEX)",
      unit_inr_24k: "INR / 10g (24K MCX)",
      unit_inr_22k: "INR / 10g (22K रिटेल)",
      unit_inr_1kg: "INR / 1kg",
      unit_etf: "Gold / Silver BeES ETF",
      kpi_spot_label: "लाइव ग्लोबल स्पॉट बेंचमार्क",
      kpi_mcx_label: "भारतीय लैंडेड MCX पैरिटी",
      kpi_gtt_label: "आदर्श GTT लिमिट-बाय ज़ोन",
      legend_ohlc: "ऐतिहासिक OHLC कैंडल्स",
      legend_q50: "मध्यमान पूर्वानुमान (Q50)",
      legend_80: "80% विश्वास कोन (Q10–Q90)",
      legend_95: "95% विश्वास कोन (Q025–Q975)",
      legend_mc: "5Y मोंटे कार्लो पथ",
      anyday_eyebrow: "कैलेंडर-जागरूक Δt राउटर",
      anyday_title: '"किसी भी विशिष्ट दिन की भविष्यवाणी" इंस्पेक्टर',
      anyday_sub: "Δt ≤ 90d को TFT/XGBoost, 91–365d को LightGBM Basis, और >365d को 1,000-पाथ VECM मोंटे कार्लो पर रूट करता है।",
      anyday_date_label: "लक्षित कैलेंडर तिथि चुनें",
      anyday_btn: "तिथि जांचें",
      preset_fomc: "अगली FOMC (18 दिसंबर 2026)",
      preset_budget: "भारत बजट (1 फरवरी 2027)",
      preset_akshaya: "अक्षय तृतीया (मई 2027)",
      preset_dhanteras: "धनतेरस पीक (अक्टूबर 2027)",
      preset_3y: "3-वर्षीय मैक्रो चक्र (नवंबर 2029)",
      anyday_80: "80% क्वांटाइल कोन (Q10–Q90)",
      anyday_95: "95% वोलैटिलिटी कोन (Q025–Q975)",
      anyday_mcx: "अनुमानित MCX लैंडेड पैरिटी",
      anyday_etf: "अनुमानित ETF NAV समतुल्य",
      gtt_box_title: "ETF और GTT लिमिट-बाय ऑर्डर ऑप्टिमाइज़र",
      gtt_acc: "संचय (Buy) बैंड",
      gtt_target: "लक्षित मुनाफा निकास",
      gtt_stop: "स्टॉप-लॉस स्तर",
      stress_eyebrow: "मॉडल पारदर्शिता और मैक्रो शॉक सिमुलेशन",
      stress_title: "यह <em>कीमत</em> क्यों? और मैक्रो स्ट्रेस लैब",
      shap_card_title: 'TreeSHAP + FinBERT फीचर योगदान ("यह कीमत क्यों?")',
      shap_card_sub: "सक्रिय मॉडल से निकाले गए सटीक Shapley भार और लाइव समाचार भावना का संयोजन।",
      regime_title: "सक्रिय अवधि मैक्रो रेजीम संभावनाएं",
      stress_card_title: "इंटरैक्टिव मैक्रो और कस्टम ड्यूटी स्ट्रेस-टेस्टर",
      stress_card_sub: "पूर्वानुमान कोन को लाइव बदलने के लिए USD/INR, मुद्रास्फीति, ब्याज दर कटौती और भारतीय आयात शुल्क बदलें।",
      reset_baseline: "रीसेट करें",
      slider_usdinr: "USD / INR विनिमय दर",
      slider_cpi: "US CPI मुद्रास्फीति (%)",
      slider_cuts: "Fed / RBI दर कटौती (bps)",
      slider_realyield: "10Y TIPS रियल यील्ड शॉक",
      slider_duty: "भारत बेसिक कस्टम ड्यूटी (%)",
      slider_gst: "बुलियन GST दर (%)",
      stress_base_label: "बेसलाइन मध्यमान",
      stress_new_label: "स्ट्रेस के बाद मध्यमान",
      stress_delta_label: "नेट शॉक प्रभाव",
      parity_eyebrow: "क्रॉस-एक्सचेंज आर्बिट्रेज और रिटेल बुलियन रूपांतरण",
      parity_title: "COMEX-से-MCX <em>पैरिटी</em> और ETF खाता",
      sip_eyebrow: "सिस्टमैटिक निवेश (SIP) और बास्केट ऑप्टिमाइज़ेशन",
      sip_title: "मल्टी-मेटल SIP और <em>संपत्ति</em> सिम्युलेटर",
      sip_preset_1: "सॉवरेन सेफ-हेवन (70/20/5/5)",
      sip_preset_2: "ग्रीन ग्रिड और सोलर (25/40/10/25)",
      sip_preset_3: "ऑल-वेदर 4-मेटल (40/30/15/15)",
      sip_left_title: "पूंजी और आवंटन भार (Weights)",
      sip_monthly_label: "मासिक SIP (₹ INR)",
      sip_lump_label: "प्रारंभिक एकमुश्त निवेश (₹ INR)",
      sip_years_label: "निवेश अवधि (वर्ष)",
      sip_w_gold: "सोना (XAU) भार %",
      sip_w_silver: "चांदी (XAG) भार %",
      sip_w_ptcu: "प्लैटिनम / तांबा %",
      sip_right_title: "अनुमानित मल्टी-होराइजन पोर्टफोलियो कुल मूल्य",
      sip_invested_label: "कुल निवेशित मूलधन",
      sip_q50_label: "अपेक्षित मध्यमान कोष (Q50)",
      sip_q90_label: "90वां परसेंटाइल बुलिश सुपरसाइकिल (Q90)",
      quant_eyebrow: "क्रॉस-मार्केट सह-एकीकरण और एक्सचेंज स्प्रेड",
      quant_title: "सहसंबंध <em>मैट्रिक्स</em> और ग्लोबल आर्बिट्रेज",
      corr_card_title: "रोलिंग क्रॉस-एसेट सहसंबंध मैट्रिक्स (30D बनाम 1Y)",
      arb_card_title: "5-हब ग्लोबल बुलियन आर्बिट्रेज स्कैनर (COMEX • LBMA • MCX • SHFE • TOCOM)",
      arb_th_hub: "एक्सचेंज हब",
      arb_th_quote: "लाइव बेंचमार्क",
      arb_th_spread: "आर्बिट्रेज स्प्रेड",
      arb_th_regime: "फिजिकल फ्लो स्थिति",
      bt_eyebrow: "कालानुक्रमिक एक्सपैंडिंग-विंडो सत्यापन (ज़ीरो डेटा लीकेज)",
      bt_title: "Walk-Forward <em>बैकटेस्ट</em> टाइम मशीन",
      bt_leg_train: "कट-ऑफ से पहले प्रशिक्षण डेटा",
      bt_leg_actual: "वास्तविक भविष्य मूल्य",
      bt_leg_pred: "मॉडल आउट-ऑफ-सैंपल पूर्वानुमान (Q50)",
      bt_leg_cone: "आउट-ऑफ-सैंपल 80% कोन",
      pulse_eyebrow: "मल्टीमॉडल NLP और फिजिकल सप्लाई-चेन टेलीमेट्री",
      pulse_title: "रियल-वर्ल्ड <em>पल्स</em> और FinBERT समाचार फीड",
      pulse_gsr_label: "गोल्ड-सिल्वर अनुपात (GSR) रडार",
      pulse_sqz_label: "फिजिकल बनाम पेपर स्क्वीज़ गेज",
      pulse_cb_label: "सेंट्रल बैंक और संस्थागत रिजर्व ट्रैकर",
      faq_eyebrow: "मॉडल आर्किटेक्चर और निष्पादन प्लेबुक",
      faq_title: "मॉडल <em>कार्यप्रणाली</em> और सामान्य प्रश्न (FAQ)",
      faq_q1: "Metal Predict रिकर्सिव रोलआउट के बजाय 3-टियर डायरेक्ट मल्टी-होराइजन एन्सेम्बल का उपयोग क्यों करता है?",
      faq_a1:
        "सामान्य ऑटोरिग्रेसिव मॉडल दिन t+1 की भविष्यवाणी करके उसे t+2 में वापस फीड करते हैं, जिससे 30–60 दिनों के बाद शोर (noise) बढ़ जाता है। Metal Predict प्रत्यक्ष होराइजन मॉडल प्रशिक्षित करता है: टियर 1 (1–90d) डायरेक्ट XGBoost क्वांटाइल रिग्रेसर, टियर 2 (91–365d) फूरियर मौसमी बेसिस + LightGBM, और टियर 3 (1Y–5Y) 1,000-पाथ मोंटे कार्लो सिमुलेशन चलाता है।",
      faq_q2: '"किसी भी विशिष्ट दिन की भविष्यवाणी" कैलेंडर राउटर FOMC बैठकों और धनतेरस को कैसे समायोजित करता है?',
      faq_a2:
        "जब आप कोई भविष्य की तारीख चुनते हैं, तो बैकएंड Δt = लक्षित तिथि − आज की गणना करता है और कैलेंडर उत्प्रेरक मैट्रिक्स की जांच करता है। FOMC दर निर्णयों, US CPI या भारत के बजट पर यह स्वचालित रूप से 80%/95% कोन को चौड़ा करता है, और धनतेरस व शादी के सीजन में भौतिक मांग गुणक लागू करता है।",
      faq_q3: "COMEX-से-MCX लैंडेड पैरिटी कीमतें और Gold/Silver BeES ETF ट्रिगर ज़ोन कैसेCalculated होते हैं?",
      faq_a3:
        "ग्लोबल COMEX स्पॉट कीमतों को लाइव USD/INR, बेसिक कस्टम ड्यूटी (BCD), AIDC सेस, स्थानीय MCX स्प्रेड और 3% GST का उपयोग करके भारतीय कीमतों में बदला जाता है। GTT लिमिट-बाय ज़ोन मॉडल के 10वें से 35वें परसेंटाइल सपोर्ट बैंड से निकाले जाते हैं।",
      faq_q4: "Walk-Forward क्रॉस-वैलिडेशन के दौरान लुकअहेड डेटा लीकेज को कैसे रोका जाता है?",
      faq_a4:
        "तारीख t पर सभी 36 तकनीकी, मैक्रोइकॉनॉमिक, सह-एकीकरण और मौसमी फीचर्स केवल ≤ t के ऐतिहासिक डेटा से गणना किए जाते हैं। प्रत्येक बैकटेस्ट फोल्ड कटऑफ तिथि से पहले के डेटा पर ही प्रशिक्षित होता है।",
      footer_headline: "Metal Predict मल्टी-होराइजन AI, <em>विश्वास</em> के साथ निर्मित",
      footer_meta_1: "<strong>इंजन:</strong> थ्री-टियर क्वांटाइल एन्सेम्बल (XGBoost + LightGBM + VECM मोंटे कार्लो)",
      footer_meta_2: "<strong>फीचर स्टोर:</strong> DuckDB 36-सिग्नल मल्टीमॉडल पाइपलाइन",
      footer_meta_3: "<strong>व्याख्या:</strong> TreeSHAP + FinBERT समाचार भावना",
      chat_header_title: "Metal Predict AI विश्लेषक",
      chat_chip_1: "1Y पूर्वानुमान और बाय ज़ोन",
      chat_chip_2: "MCX 24K/22K पैरिटी भाव",
      chat_chip_3: "कीमत क्यों बढ़ रही है? (SHAP)",
      chat_chip_4: "2027-10-29 की भविष्यवाणी",
      chat_send_btn: "पूछें 押",
      chat_fab_title: "AI विश्लेषक • Metal Predict से पूछें",
      chat_fab_sub: "लाइव पूर्वानुमान, MCX और SHAP प्रश्नोत्तर",
      chat_placeholder: "सोना, चांदी, MCX, SHAP या किसी भी तारीख के बारे में पूछें...",
      chat_welcome:
        "<strong>Metal Predict AI • क्वांटिटेटिव बुलियन डेस्क</strong>\nमुझसे मल्टी-होराइजन पूर्वानुमान (1M–5Y), किसी भी भविष्य की तारीख (YYYY-MM-DD), भारतीय MCX 24K/22K भाव, TreeSHAP कारण या GTT बाय ज़ोन के बारे में पूछें!",
      fan_title_suffix: "प्रायिकता फैन",
      kpi_horizon_suffix: "मध्यमान अनुमान",
    },
  };

  const TROY_OZ_TO_G = 31.1034768;
  const LB_TO_KG = 0.45359237;

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function convertPrice(usdVal, metal = state.metal, mode = state.unitMode) {
    const v = Number(usdVal) || 0;
    const taxMult = (1 + (state.bcdPct + state.aidcPct) / 100) * (1 + state.gstPct / 100) * 1.0035;
    const inrPerGram =
      metal === "copper"
        ? ((v / LB_TO_KG) * state.usdinr * taxMult) / 1000.0
        : (v / TROY_OZ_TO_G) * state.usdinr * taxMult;

    switch (mode) {
      case "INR_10G_24K":
        return inrPerGram * 10.0;
      case "INR_10G_22K":
        return inrPerGram * 10.0 * 0.9167;
      case "INR_1KG":
        return inrPerGram * 1000.0;
      case "ETF_NAV":
        if (metal === "gold") return inrPerGram * 0.01 * 0.985;
        if (metal === "silver") return inrPerGram * 1.0 * 0.988;
        return v * 0.1;
      case "USD_OZ":
      default:
        return v;
    }
  }

  function unitLabel(metal = state.metal, mode = state.unitMode) {
    switch (mode) {
      case "INR_10G_24K":
        return "INR / 10g (24K)";
      case "INR_10G_22K":
        return "INR / 10g (22K)";
      case "INR_1KG":
        return "INR / 1kg";
      case "ETF_NAV":
        return metal === "gold"
          ? "Gold BeES NAV (₹)"
          : metal === "silver"
          ? "Silver BeES NAV (₹)"
          : "ETF Unit ($)";
      case "USD_OZ":
      default:
        return metal === "copper" ? "USD / lb" : "USD / oz";
    }
  }

  function formatCurrency(usdVal, metal = state.metal, mode = state.unitMode) {
    const converted = convertPrice(usdVal, metal, mode);
    const isInr = mode.startsWith("INR_") || (mode === "ETF_NAV" && (metal === "gold" || metal === "silver"));
    if (isInr) {
      return (
        "₹" +
        converted.toLocaleString("en-IN", {
          minimumFractionDigits: converted < 1000 ? 2 : 0,
          maximumFractionDigits: converted < 1000 ? 2 : 0,
        })
      );
    }
    return (
      "$" +
      converted.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
    );
  }

  function setPreloaderStep(stepIdx, pct, label) {
    const fill = document.getElementById("mpPreloaderFill");
    const status = document.getElementById("mpPreloaderStatus");
    if (fill) fill.style.width = `${pct}%`;
    if (status) status.textContent = label;
    document.querySelectorAll(".mp-preloader__step").forEach((el, idx) => {
      el.classList.toggle("is-active", idx <= stepIdx);
    });
  }

  function toggleSkeletonState(isLoading) {
    const loader = document.getElementById("mpCanvasLoader");
    if (loader) loader.classList.toggle("is-loading", isLoading);
    document.querySelectorAll("[data-mp-skeleton-target]").forEach((el) => {
      el.classList.toggle("mp-skeleton", isLoading);
    });
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function runChoreographedPreloader() {
    setPreloaderStep(0, 24, "01 / 04 • Ingesting 15-Year DuckDB OHLCV & Macro Store...");
    await sleep(1150);
    setPreloaderStep(1, 54, "02 / 04 • Calibrating Tier 1–2 XGBoost & LightGBM Quantiles...");
    await sleep(1150);
    setPreloaderStep(2, 82, "03 / 04 • Simulating 1,000 VECM Jump-Diffusion Monte Carlo Paths...");
    await sleep(1150);
    setPreloaderStep(3, 100, "04 / 04 • Extracting TreeSHAP & FinBERT Sentiment Pulse...");
    await sleep(950);
    const pre = document.getElementById("mpPreloader");
    if (pre) pre.classList.add("is-dismissed");
  }

  async function loadAllForMetal(metal) {
    state.metal = metal;
    state.stressedPayload = null;
    state.fanHover = null;
    toggleSkeletonState(true);
    updateChatContextBadge();

    try {
      const q = `metal=${encodeURIComponent(metal)}&usdinr=${state.usdinr}&bcd_pct=${state.bcdPct}&aidc_pct=${state.aidcPct}&gst_pct=${state.gstPct}`;
      const fetchPromise = Promise.all([
        fetchJson(`/api/predict/horizons?${q}`),
        fetchJson(`/api/pulse/overview?metal=${encodeURIComponent(metal)}&usdinr=${state.usdinr}`),
        fetchJson(`/api/backtest/replay?metal=${encodeURIComponent(metal)}&fold_index=${state.foldIndex}`),
      ]);

      if (!state.initialLoadDone) {
        const [[horizonsData, pulseData, backtestData]] = await Promise.all([
          fetchPromise,
          runChoreographedPreloader(),
        ]);
        state.horizonPayload = horizonsData;
        state.pulsePayload = pulseData;
        state.backtestPayload = backtestData;
        // Let the slow Washi-Paper Shimmer Skeleton remain visible on the page for 2.6s after the curtain lifts
        await sleep(2600);
        state.initialLoadDone = true;
      } else {
        const [[horizonsData, pulseData, backtestData]] = await Promise.all([
          fetchPromise,
          sleep(1450), // Deliberate slow skeleton shimmer transition on metal switch
        ]);
        state.horizonPayload = horizonsData;
        state.pulsePayload = pulseData;
        state.backtestPayload = backtestData;
      }

      const dateInput = document.getElementById("kmTargetDateInput");
      if (dateInput && dateInput.value) {
        await runAnyDayPrediction(dateInput.value, true);
      }

      toggleSkeletonState(false);
      renderMetalTabs();
      renderKpiStrip();
      renderMainFanChart(true);
      renderShapAndRegimes();
      renderParityLedger();
      renderBacktestSection(true);
      renderPulseSection();
      runPortfolioSipSimulation();
    } catch (err) {
      toggleSkeletonState(false);
      const pre = document.getElementById("mpPreloader");
      if (pre) pre.classList.add("is-dismissed");
      console.error("Failed loading Metal Predict telemetry:", err);
    }
  }

  function renderMetalTabs() {
    const container = document.getElementById("kmMetalTabs");
    if (!container || !state.horizonPayload) return;
    const dict = I18N_DICT[state.lang] || I18N_DICT.en;
    const summaries = state.horizonPayload.all_metals || [];
    container.innerHTML = summaries
      .map((item) => {
        const active = item.metal === state.metal ? "is-active" : "";
        const chgSign = item.change_24h_pct >= 0 ? "+" : "";
        const locName = (dict.metalNames && dict.metalNames[item.metal]) || item.name;
        return `
          <button type="button" class="km-metal-tab ${active}" data-metal="${item.metal}">
            <span class="km-metal-tab__kanji" lang="ja">${item.kanji}</span>
            <span class="km-metal-tab__meta">
              <span class="km-metal-tab__name">${locName} • ${item.symbol}</span>
              <span class="km-metal-tab__price">${formatCurrency(item.spot_usd, item.metal, "USD_OZ")} (${chgSign}${item.change_24h_pct}%)</span>
            </span>
          </button>
        `;
      })
      .join("");

    container.querySelectorAll(".km-metal-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const m = btn.getAttribute("data-metal");
        if (m && m !== state.metal) loadAllForMetal(m);
      });
    });
  }

  function renderKpiStrip() {
    if (!state.horizonPayload) return;
    const dict = I18N_DICT[state.lang] || I18N_DICT.en;
    const p = state.horizonPayload;
    const hObj = p.horizons[state.horizon];
    const term = state.stressedPayload ? state.stressedPayload.stressed_terminal : hObj.terminal;
    const spot = p.spot_usd;
    const retPct = (((term.q50 - spot) / spot) * 100).toFixed(2);
    const sign = retPct >= 0 ? "+" : "";

    document.getElementById("kmKpiSpot").textContent = formatCurrency(spot);
    document.getElementById("kmKpiSpotSub").innerHTML = `COMEX Spot (${unitLabel()}) • <span class="${
      p.change_24h_pct >= 0 ? "km-badge-pos" : "km-badge-neg"
    }">${p.change_24h_pct >= 0 ? "+" : ""}${p.change_24h_pct}% 24h</span>`;

    document.getElementById("kmKpiHorizonLabel").textContent = `${state.horizon} ${dict.kpi_horizon_suffix || "Median Projection"} (${hObj.days}D)`;
    document.getElementById("kmKpiHorizonVal").textContent = formatCurrency(term.q50);
    document.getElementById("kmKpiHorizonSub").innerHTML = `<span class="${
      retPct >= 0 ? "km-badge-pos" : "km-badge-neg"
    }">${sign}${retPct}%</span> • 80% Cone: ${formatCurrency(term.q10)} – ${formatCurrency(term.q90)}`;

    const parity = state.stressedPayload ? state.stressedPayload.stressed_parity : p.parity;
    document.getElementById("kmKpiMcxVal").textContent =
      "₹" + Number(parity.primary_mcx_inr).toLocaleString("en-IN");
    document.getElementById("kmKpiMcxSub").textContent = `${parity.primary_mcx_unit} • USD/INR ${parity.usdinr} • Tax ${parity.tax_breakdown_per_10g.effective_tax_rate_pct}%`;

    const gtt = state.stressedPayload ? state.stressedPayload.stressed_gtt_zone : p.gtt_zone;
    document.getElementById("kmKpiGttVal").textContent = `${formatCurrency(gtt.accumulate_low_usd)} – ${formatCurrency(
      gtt.accumulate_high_usd
    )}`;
    document.getElementById("kmKpiGttSub").textContent = `Target Exit: ${formatCurrency(
      gtt.target_exit_usd
    )} • R:R ${gtt.reward_to_risk_ratio}x`;

    const baseTerm = hObj.terminal;
    if (!state.stressedPayload) {
      document.getElementById("kmStressBaseVal").textContent = formatCurrency(baseTerm.q50);
      document.getElementById("kmStressNewVal").textContent = formatCurrency(baseTerm.q50);
      const deltaEl = document.getElementById("kmStressDeltaVal");
      deltaEl.className = "km-badge-pos";
      deltaEl.textContent = "+0.00% (Baseline)";
    }
  }

  function animateFanChartSweep() {
    if (state.fanAnimFrame) cancelAnimationFrame(state.fanAnimFrame);
    const start = performance.now();
    const duration = 680;
    function step(now) {
      const t = Math.min(1.0, (now - start) / duration);
      state.fanAnimProgress = 1 - Math.pow(1 - t, 3);
      drawFanCanvas();
      if (t < 1.0) {
        state.fanAnimFrame = requestAnimationFrame(step);
      }
    }
    state.fanAnimFrame = requestAnimationFrame(step);
  }

  function renderMainFanChart(animate = false) {
    if (!state.horizonPayload) return;
    const dict = I18N_DICT[state.lang] || I18N_DICT.en;
    const p = state.horizonPayload;
    const hObj = p.horizons[state.horizon];
    const locName = (dict.metalNames && dict.metalNames[state.metal]) || p.spec.name;
    document.getElementById("kmActiveChartTitle").textContent = `${locName} (${p.spec.symbol}) — ${state.horizon} ${dict.fan_title_suffix || "Probability Fan"}`;
    document.getElementById("kmActiveTierBadge").textContent = hObj.tier;

    const m = hObj.metrics;
    const strip = document.getElementById("kmMetricsStrip");
    if (strip && m) {
      strip.innerHTML = `
        <div class="km-metric-box"><div class="km-metric-box__k">Walk-Forward MAPE</div><div class="km-metric-box__v">${m.mape}%</div></div>
        <div class="km-metric-box"><div class="km-metric-box__k">Directional Accuracy</div><div class="km-metric-box__v">${m.directional_accuracy}%</div></div>
        <div class="km-metric-box"><div class="km-metric-box__k">Pinball Quantile Loss</div><div class="km-metric-box__v">${m.pinball_loss}%</div></div>
        <div class="km-metric-box"><div class="km-metric-box__k">80% Band Coverage</div><div class="km-metric-box__v">${m.coverage_80}%</div></div>
        <div class="km-metric-box"><div class="km-metric-box__k">95% Band Coverage</div><div class="km-metric-box__v">${m.coverage_95}%</div></div>
      `;
    }

    if (animate) {
      animateFanChartSweep();
    } else {
      state.fanAnimProgress = 1.0;
      drawFanCanvas();
    }
  }

  function getFanChartGeometry(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    const pad = { top: 28, right: 82, bottom: 36, left: 18 };
    const plotW = Math.max(10, W - pad.left - pad.right);
    const plotH = Math.max(10, H - pad.top - pad.bottom);

    const histCount = state.horizon === "1M" ? 45 : state.horizon === "3M" ? 70 : 100;
    const histSlice = state.horizonPayload.history.slice(-histCount);
    const activeSeries = state.stressedPayload
      ? state.stressedPayload.stressed_series
      : state.horizonPayload.horizons[state.horizon].series;
    const baseSeries = state.horizonPayload.horizons[state.horizon].series;
    const totalPoints = histSlice.length + activeSeries.length;
    const splitIdx = histSlice.length - 1;

    return { rect, W, H, pad, plotW, plotH, histSlice, activeSeries, baseSeries, totalPoints, splitIdx };
  }

  function drawFanCanvas() {
    const canvas = document.getElementById("kmFanChart");
    if (!canvas || !state.horizonPayload) return;
    const geom = getFanChartGeometry(canvas);
    const { W, H, pad, plotW, plotH, histSlice, activeSeries, baseSeries, totalPoints, splitIdx } = geom;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const allPrices = [];
    histSlice.forEach((h) => {
      allPrices.push(convertPrice(h.high), convertPrice(h.low));
    });
    activeSeries.forEach((pt) => {
      allPrices.push(convertPrice(pt.q025), convertPrice(pt.q975));
      if (pt.baseline_q50) allPrices.push(convertPrice(pt.baseline_q50));
    });

    const minP = Math.min(...allPrices) * 0.992;
    const maxP = Math.max(...allPrices) * 1.008;

    const xOf = (idx) => pad.left + (idx / Math.max(totalPoints - 1, 1)) * plotW;
    const yOf = (val) => pad.top + (1 - (val - minP) / Math.max(maxP - minP, 1e-6)) * plotH;

    // Grid lines
    ctx.strokeStyle = "rgba(39, 39, 39, 0.11)";
    ctx.lineWidth = 1;
    ctx.font = "600 11px Kanit, sans-serif";
    ctx.fillStyle = "#4a4144";
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      const val = maxP - (i / 4) * (maxP - minP);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.fillText(
        val >= 1000 ? Math.round(val).toLocaleString() : val.toFixed(2),
        W - pad.right + 8,
        y + 4
      );
    }

    const splitX = xOf(splitIdx);
    const visibleForecastCount = Math.max(1, Math.round(activeSeries.length * state.fanAnimProgress));
    const visibleForecast = activeSeries.slice(0, visibleForecastCount);

    // 95% Confidence Cone
    ctx.beginPath();
    ctx.moveTo(splitX, yOf(convertPrice(histSlice[splitIdx].close)));
    visibleForecast.forEach((pt, i) => {
      ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(pt.q975)));
    });
    for (let i = visibleForecast.length - 1; i >= 0; i--) {
      ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(visibleForecast[i].q025)));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(33, 70, 75, 0.12)";
    ctx.fill();

    // 80% Confidence Cone
    ctx.beginPath();
    ctx.moveTo(splitX, yOf(convertPrice(histSlice[splitIdx].close)));
    visibleForecast.forEach((pt, i) => {
      ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(pt.q90)));
    });
    for (let i = visibleForecast.length - 1; i >= 0; i--) {
      ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(visibleForecast[i].q10)));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(219, 91, 127, 0.24)";
    ctx.fill();

    // 5Y Monte Carlo Sample Paths
    const samplePaths = state.horizonPayload.horizons[state.horizon].sample_paths;
    if (samplePaths && !state.stressedPayload) {
      ctx.lineWidth = 0.85;
      ctx.strokeStyle = "rgba(157, 56, 53, 0.24)";
      samplePaths.forEach((pathArr) => {
        ctx.beginPath();
        ctx.moveTo(splitX, yOf(convertPrice(histSlice[splitIdx].close)));
        pathArr.slice(0, visibleForecastCount).forEach((pVal, idx) => {
          ctx.lineTo(xOf(splitIdx + 1 + idx), yOf(convertPrice(pVal)));
        });
        ctx.stroke();
      });
    }

    // Historical Candlesticks
    const candleWidth = Math.max(2, Math.min(7, (plotW / totalPoints) * 0.65));
    histSlice.forEach((bar, i) => {
      const cx = xOf(i);
      const o = convertPrice(bar.open);
      const c = convertPrice(bar.close);
      const hi = convertPrice(bar.high);
      const lo = convertPrice(bar.low);
      const isUp = c >= o;
      ctx.strokeStyle = isUp ? "#21464b" : "#9d3835";
      ctx.fillStyle = isUp ? "#21464b" : "#9d3835";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, yOf(hi));
      ctx.lineTo(cx, yOf(lo));
      ctx.stroke();
      const topY = yOf(Math.max(o, c));
      const botY = yOf(Math.min(o, c));
      ctx.fillRect(cx - candleWidth / 2, topY, candleWidth, Math.max(1.5, botY - topY));
    });

    // TODAY Anchor Divider
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#9d3835";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(splitX, pad.top);
    ctx.lineTo(splitX, H - pad.bottom);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "#9d3835";
    ctx.font = "600 10px Kanit, sans-serif";
    ctx.fillText("TODAY • ANCHOR", Math.max(pad.left, splitX - 38), pad.top - 8);

    // Baseline Median Dashed if Stressed
    if (state.stressedPayload) {
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#4a4144";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(splitX, yOf(convertPrice(histSlice[splitIdx].close)));
      baseSeries.slice(0, visibleForecastCount).forEach((pt, i) => {
        ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(pt.q50)));
      });
      ctx.stroke();
      ctx.restore();
    }

    // Active Median Trajectory
    ctx.strokeStyle = "#d62828";
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(splitX, yOf(convertPrice(histSlice[splitIdx].close)));
    visibleForecast.forEach((pt, i) => {
      ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(pt.q50)));
    });
    ctx.stroke();

    // Terminal Forecast Beacon
    const lastVisPt = visibleForecast[visibleForecast.length - 1];
    const termX = xOf(splitIdx + visibleForecast.length);
    const termY = yOf(convertPrice(lastVisPt.q50));
    ctx.fillStyle = "rgba(214, 40, 40, 0.2)";
    ctx.beginPath();
    ctx.arc(termX, termY, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#d62828";
    ctx.beginPath();
    ctx.arc(termX, termY, 4.2, 0, Math.PI * 2);
    ctx.fill();

    // Interactive Crosshair
    const tooltip = document.getElementById("kmChartTooltip");
    if (state.fanHover) {
      const h = state.fanHover;
      const hx = xOf(h.globalIdx);
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#202020";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, pad.top);
      ctx.lineTo(hx, H - pad.bottom);
      ctx.stroke();
      ctx.restore();

      if (h.type === "hist") {
        const bar = histSlice[h.localIdx];
        const cy = yOf(convertPrice(bar.close));
        ctx.fillStyle = "#21464b";
        ctx.beginPath();
        ctx.arc(hx, cy, 4.5, 0, Math.PI * 2);
        ctx.fill();
        if (tooltip) {
          tooltip.innerHTML = `
            <strong>Historical Session: ${bar.date}</strong><br/>
            Close: <strong>${formatCurrency(bar.close)}</strong> • Open: ${formatCurrency(bar.open)}<br/>
            High: ${formatCurrency(bar.high)} • Low: ${formatCurrency(bar.low)}
          `;
        }
      } else {
        const pt = activeSeries[h.localIdx];
        const y50 = yOf(convertPrice(pt.q50));
        const y90 = yOf(convertPrice(pt.q90));
        const y10 = yOf(convertPrice(pt.q10));
        [
          [y90, "#db5b7f"],
          [y50, "#d62828"],
          [y10, "#db5b7f"],
        ].forEach(([py, col]) => {
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(hx, py, 4.2, 0, Math.PI * 2);
          ctx.fill();
        });
        if (tooltip) {
          tooltip.innerHTML = `
            <strong>Forecast Target: ${pt.date} (+${pt.days_ahead}D)</strong><br/>
            Median (Q50): <strong>${formatCurrency(pt.q50)}</strong><br/>
            80% Cone: ${formatCurrency(pt.q10)} – ${formatCurrency(pt.q90)}<br/>
            95% Cone: ${formatCurrency(pt.q025)} – ${formatCurrency(pt.q975)}
          `;
        }
      }
    } else if (tooltip) {
      const lastPt = activeSeries[activeSeries.length - 1];
      tooltip.innerHTML = `
        <strong>Horizon Terminal: ${lastPt.date} (+${lastPt.days_ahead}D)</strong><br/>
        Median (Q50): <strong>${formatCurrency(lastPt.q50)}</strong><br/>
        80% Cone: ${formatCurrency(lastPt.q10)} – ${formatCurrency(lastPt.q90)}<br/>
        95% Cone: ${formatCurrency(lastPt.q025)} – ${formatCurrency(lastPt.q975)}
      `;
    }

    ctx.fillStyle = "#4a4144";
    ctx.font = "400 11px Poppins, sans-serif";
    ctx.fillText(histSlice[0].date, pad.left, H - 10);
    ctx.fillText(histSlice[splitIdx].date, Math.max(pad.left + 80, splitX - 30), H - 10);
    const lastPt = activeSeries[activeSeries.length - 1];
    ctx.fillText(lastPt.date, W - pad.right - 72, H - 10);
  }

  async function runAnyDayPrediction(targetDate, silent = false) {
    const errEl = document.getElementById("kmAnyDayError");
    const box = document.getElementById("kmAnyDayTelemetryBox");
    if (errEl) errEl.textContent = "";
    if (!silent && box) box.classList.add("mp-skeleton");
    try {
      const q = `metal=${encodeURIComponent(state.metal)}&target_date=${encodeURIComponent(
        targetDate
      )}&usdinr=${state.usdinr}&bcd_pct=${state.bcdPct}&aidc_pct=${state.aidcPct}&gst_pct=${state.gstPct}`;
      const [data] = await Promise.all([
        fetchJson(`/api/predict/date?${q}`),
        silent ? Promise.resolve() : sleep(650),
      ]);
      state.anyDayPayload = data;
      if (box) box.classList.remove("mp-skeleton");
      renderAnyDayCard(data);
    } catch (err) {
      if (box) box.classList.remove("mp-skeleton");
      if (errEl) errEl.textContent = err.message;
    }
  }

  function renderAnyDayCard(d) {
    document.getElementById("kmAnyDayTier").textContent = `${d.tier} • Δt = +${d.delta_days} Days`;
    document.getElementById("kmAnyDayMedian").textContent = formatCurrency(d.q50);
    const chgEl = document.getElementById("kmAnyDayReturn");
    const sign = d.expected_return_pct >= 0 ? "+" : "";
    chgEl.className = d.expected_return_pct >= 0 ? "km-badge-pos" : "km-badge-neg";
    chgEl.textContent = `${sign}${d.expected_return_pct}% vs Spot`;

    document.getElementById("kmAnyDay80").textContent = `${formatCurrency(d.q10)} – ${formatCurrency(d.q90)}`;
    document.getElementById("kmAnyDay95").textContent = `${formatCurrency(d.q025)} – ${formatCurrency(d.q975)}`;

    const liveMcxInr = convertPrice(
      d.q50,
      state.metal,
      state.metal === "gold" || state.metal === "platinum" ? "INR_10G_24K" : "INR_1KG"
    );
    document.getElementById("kmAnyDayMcx").textContent = "₹" + Math.round(liveMcxInr).toLocaleString("en-IN");
    const liveEtf = convertPrice(d.q50, state.metal, "ETF_NAV");
    document.getElementById("kmAnyDayEtf").textContent =
      state.metal === "silver"
        ? `Silver BeES ₹${liveEtf.toFixed(2)}`
        : state.metal === "gold"
        ? `Gold BeES ₹${liveEtf.toFixed(2)}`
        : `ETF Unit $${liveEtf.toFixed(2)}`;

    const catList = document.getElementById("kmAnyDayCatalysts");
    catList.innerHTML = (d.catalysts.active_catalysts || [])
      .map((c) => `<div class="km-catalyst-item">${c}</div>`)
      .join("");

    const g = d.gtt_zone;
    document.getElementById("kmGttAccumulate").textContent = `${formatCurrency(
      g.accumulate_low_usd
    )} – ${formatCurrency(g.accumulate_high_usd)}`;
    document.getElementById("kmGttTarget").textContent = formatCurrency(g.target_exit_usd);
    document.getElementById("kmGttStop").textContent = formatCurrency(g.stop_loss_usd);
  }

  function renderShapAndRegimes() {
    if (!state.horizonPayload) return;
    const drivers =
      (state.horizonPayload.shap_by_horizon && state.horizonPayload.shap_by_horizon[state.horizon]) ||
      state.horizonPayload.shap_drivers ||
      [];
    const barsEl = document.getElementById("kmShapBars");
    if (barsEl) {
      barsEl.innerHTML = drivers
        .map((d) => {
          const isBull = d.signed_weight_pct >= 0;
          const sign = isBull ? "+" : "";
          const width = Math.min(100, Math.max(12, Math.abs(d.weight_pct) * 2.4));
          return `
            <div class="km-shap-row">
              <div class="km-shap-row__top">
                <span>${d.label}</span>
                <span class="${isBull ? "km-badge-pos" : "km-badge-neg"}">${sign}${d.signed_weight_pct}%</span>
              </div>
              <div class="km-shap-track">
                <div class="km-shap-fill ${isBull ? "is-bull" : "is-bear"}" data-target-width="${width}%"></div>
              </div>
            </div>
          `;
        })
        .join("");

      requestAnimationFrame(() => {
        barsEl.querySelectorAll(".km-shap-fill").forEach((bar) => {
          bar.style.width = bar.getAttribute("data-target-width") || "50%";
        });
      });
    }

    const narrative =
      (state.horizonPayload.narratives_by_horizon && state.horizonPayload.narratives_by_horizon[state.horizon]) || "";
    const narEl = document.getElementById("kmShapNarrative");
    if (narEl) narEl.textContent = narrative;

    const regimes = state.stressedPayload
      ? state.stressedPayload.stressed_regime_probabilities
      : state.horizonPayload.horizons[state.horizon].regime_probabilities ||
        state.horizonPayload.horizons["6M"].regime_probabilities ||
        {};
    const regEl = document.getElementById("kmRegimeGrid");
    if (regEl) {
      regEl.innerHTML = Object.entries(regimes)
        .map(([k, v]) => `<div class="km-regime-item"><span>${k}</span><strong>${v}%</strong></div>`)
        .join("");
    }
  }

  function scheduleMacroShockSimulation() {
    if (state.stressDebounceTimer) clearTimeout(state.stressDebounceTimer);
    state.stressDebounceTimer = setTimeout(() => {
      triggerMacroShockSimulation();
    }, 65);
  }

  async function triggerMacroShockSimulation() {
    const seq = ++state.stressSeq;
    const body = {
      metal: state.metal,
      horizon: state.horizon,
      usdinr_rate: state.usdinr,
      us_cpi_pct: state.cpi,
      rate_cuts_bps: state.rateCutsBps,
      real_yield_delta_bps: state.realYieldDeltaBps,
      customs_duty_pct: state.bcdPct,
      aidc_pct: state.aidcPct,
      gst_pct: state.gstPct,
    };
    const sim = await fetchJson("/api/simulate/macro-shock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (seq !== state.stressSeq) return;
    state.stressedPayload = sim;

    document.getElementById("kmStressBaseVal").textContent = formatCurrency(sim.baseline_terminal_q50);
    document.getElementById("kmStressNewVal").textContent = formatCurrency(sim.stressed_terminal.q50);
    const deltaEl = document.getElementById("kmStressDeltaVal");
    const sign = sim.delta_vs_baseline_pct >= 0 ? "+" : "";
    deltaEl.className = sim.delta_vs_baseline_pct >= 0 ? "km-badge-pos" : "km-badge-neg";
    deltaEl.textContent = `${sign}${sim.delta_vs_baseline_pct}% (${
      sim.annualized_macro_impulse_pct >= 0 ? "+" : ""
    }${sim.annualized_macro_impulse_pct}%/yr)`;

    renderKpiStrip();
    drawFanCanvas();
    renderParityLedger();
    renderShapAndRegimes();
    if (state.anyDayPayload) renderAnyDayCard(state.anyDayPayload);
  }

  function renderParityLedger() {
    if (!state.horizonPayload) return;
    const parity = state.stressedPayload
      ? state.stressedPayload.stressed_parity
      : state.horizonPayload.parity;
    const tax = parity.tax_breakdown_per_10g;
    const grid = document.getElementById("kmParityGrid");
    if (!grid) return;

    grid.innerHTML = `
      <div class="km-kpi-card">
        <span class="km-kpi-card__stamp" lang="ja">印</span>
        <div class="km-kpi-card__label">MCX Landed 24K (10g)</div>
        <div class="km-kpi-card__value">₹${Number(parity.inr_per_10g_24k).toLocaleString("en-IN")}</div>
        <div class="km-kpi-card__sub">22K Retail: ₹${Number(parity.inr_per_10g_22k).toLocaleString("en-IN")} / 10g</div>
      </div>
      <div class="km-kpi-card">
        <span class="km-kpi-card__stamp" lang="ja">税</span>
        <div class="km-kpi-card__label">Customs Duty + AIDC + GST</div>
        <div class="km-kpi-card__value">${tax.effective_tax_rate_pct}%</div>
        <div class="km-kpi-card__sub">Duty ₹${Math.round(tax.customs_duty_inr).toLocaleString("en-IN")} • AIDC ₹${Math.round(
      tax.aidc_cess_inr
    ).toLocaleString("en-IN")} • GST ₹${Math.round(tax.gst_3pct_inr).toLocaleString("en-IN")}</div>
      </div>
      <div class="km-kpi-card">
        <span class="km-kpi-card__stamp" lang="ja">塊</span>
        <div class="km-kpi-card__label">Wholesale Bullion (1 kg)</div>
        <div class="km-kpi-card__value">₹${Math.round(parity.inr_per_1kg).toLocaleString("en-IN")}</div>
        <div class="km-kpi-card__sub">CIF Base: ₹${Math.round(tax.cif_base_inr).toLocaleString("en-IN")} / 10g</div>
      </div>
      <div class="km-kpi-card">
        <span class="km-kpi-card__stamp" lang="ja">証</span>
        <div class="km-kpi-card__label">Domestic &amp; US ETF Parity</div>
        <div class="km-kpi-card__value">₹${
          state.metal === "silver" ? parity.etf.silver_bees_nav : parity.etf.gold_bees_nav
        }</div>
        <div class="km-kpi-card__sub">Gold BeES ₹${parity.etf.gold_bees_nav} • Silver BeES ₹${
      parity.etf.silver_bees_nav
    }</div>
      </div>
    `;
  }

  async function runPortfolioSipSimulation() {
    const sipEl = document.getElementById("mpSipMonthly");
    const lumpEl = document.getElementById("mpSipLumpSum");
    const yrsEl = document.getElementById("mpSipYears");
    const wGoldEl = document.getElementById("mpWeightGold");
    const wSilverEl = document.getElementById("mpWeightSilver");
    const wPlatEl = document.getElementById("mpWeightPlatinum");
    const wCopEl = document.getElementById("mpWeightCopper");
    if (!sipEl || !lumpEl) return;

    const payload = {
      monthly_sip: Number(sipEl.value) || 0,
      lump_sum: Number(lumpEl.value) || 0,
      horizon_years: Number(yrsEl.value) || 3,
      currency: "INR",
      gold_weight: Number(wGoldEl.value) || 50,
      silver_weight: Number(wSilverEl.value) || 25,
      platinum_weight: Number(wPlatEl.value) || 10,
      copper_weight: Number(wCopEl.value) || 15,
    };

    try {
      const res = await fetchJson("/api/simulate/portfolio-sip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      document.getElementById("mpSipInvested").textContent = "₹" + Number(res.total_invested).toLocaleString("en-IN");
      document.getElementById("mpSipQ50").textContent = "₹" + Number(res.projected_median_q50).toLocaleString("en-IN");
      document.getElementById("mpSipQ90").textContent = "₹" + Number(res.projected_bull_q90).toLocaleString("en-IN");
      document.getElementById("mpSipMetrics").textContent = `CAGR: ${res.expected_cagr_pct}% • Vol: ${res.portfolio_volatility_pct}% • Sharpe: ${res.sharpe_ratio}`;
    } catch (e) {
      console.error("SIP simulation error:", e);
    }
  }

  function animateBacktestSweep() {
    if (state.btAnimFrame) cancelAnimationFrame(state.btAnimFrame);
    const start = performance.now();
    const duration = 560;
    function step(now) {
      const t = Math.min(1.0, (now - start) / duration);
      state.btAnimProgress = 1 - Math.pow(1 - t, 3);
      drawBacktestCanvas();
      if (t < 1.0) {
        state.btAnimFrame = requestAnimationFrame(step);
      }
    }
    state.btAnimFrame = requestAnimationFrame(step);
  }

  function renderBacktestSection(animate = false) {
    if (!state.backtestPayload) return;
    const bt = state.backtestPayload;
    const foldList = document.getElementById("kmFoldList");
    if (foldList) {
      foldList.innerHTML = bt.available_folds
        .map((f) => {
          const active = f.fold_index === bt.selected_fold_index ? "is-active" : "";
          return `
            <button type="button" class="km-fold-btn ${active}" data-fold="${f.fold_index}">
              <span class="km-fold-btn__title">${f.label}</span>
              <span class="km-fold-btn__meta">Anchor: ${f.anchor_date} (${formatCurrency(
            f.anchor_price
          )}) • MAPE: ${f.fold_mape}% • DA: ${f.fold_da}%</span>
            </button>
          `;
        })
        .join("");

      foldList.querySelectorAll(".km-fold-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const idx = Number(btn.getAttribute("data-fold")) || 0;
          state.foldIndex = idx;
          state.backtestHover = null;
          state.backtestPayload = await fetchJson(
            `/api/backtest/replay?metal=${encodeURIComponent(state.metal)}&fold_index=${idx}`
          );
          renderBacktestSection(true);
        });
      });
    }

    if (animate) {
      animateBacktestSweep();
    } else {
      state.btAnimProgress = 1.0;
      drawBacktestCanvas();
    }
  }

  function drawBacktestCanvas() {
    const canvas = document.getElementById("kmBacktestChart");
    if (!canvas || !state.backtestPayload) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 28, right: 78, bottom: 32, left: 18 };
    const plotW = Math.max(10, W - pad.left - pad.right);
    const plotH = Math.max(10, H - pad.top - pad.bottom);

    ctx.clearRect(0, 0, W, H);
    const bt = state.backtestPayload;
    const before = bt.history_before || [];
    const preds = bt.predicted_series || [];
    const total = before.length + preds.length;

    const vals = [];
    before.forEach((b) => vals.push(convertPrice(b.close)));
    preds.forEach((p) => vals.push(convertPrice(p.q10), convertPrice(p.q90), convertPrice(p.actual)));
    const minP = Math.min(...vals) * 0.985;
    const maxP = Math.max(...vals) * 1.015;

    const xOf = (i) => pad.left + (i / Math.max(total - 1, 1)) * plotW;
    const yOf = (v) => pad.top + (1 - (v - minP) / Math.max(maxP - minP, 1e-6)) * plotH;

    ctx.strokeStyle = "rgba(39, 39, 39, 0.11)";
    ctx.lineWidth = 1;
    ctx.font = "600 11px Kanit, sans-serif";
    ctx.fillStyle = "#4a4144";
    for (let i = 0; i <= 3; i++) {
      const y = pad.top + (i / 3) * plotH;
      const val = maxP - (i / 3) * (maxP - minP);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.fillText(
        val >= 1000 ? Math.round(val).toLocaleString() : val.toFixed(2),
        W - pad.right + 8,
        y + 4
      );
    }

    const splitIdx = before.length - 1;
    const splitX = xOf(splitIdx);
    const visCount = Math.max(1, Math.round(preds.length * state.btAnimProgress));
    const visPreds = preds.slice(0, visCount);

    // 80% Out-of-Sample Forecast Cone
    ctx.beginPath();
    ctx.moveTo(splitX, yOf(convertPrice(before[splitIdx].close)));
    visPreds.forEach((p, i) => ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(p.q90))));
    for (let i = visPreds.length - 1; i >= 0; i--) {
      ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(visPreds[i].q10)));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(219, 91, 127, 0.22)";
    ctx.fill();

    // Pre-anchor training curve
    ctx.strokeStyle = "#202020";
    ctx.lineWidth = 2;
    ctx.beginPath();
    before.forEach((b, i) => {
      if (i === 0) ctx.moveTo(xOf(i), yOf(convertPrice(b.close)));
      else ctx.lineTo(xOf(i), yOf(convertPrice(b.close)));
    });
    ctx.stroke();

    // Realized Actual Future Curve
    ctx.strokeStyle = "#21464b";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(splitX, yOf(convertPrice(before[splitIdx].close)));
    visPreds.forEach((p, i) => ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(p.actual))));
    ctx.stroke();

    // Out-of-Sample Predicted Median
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#d62828";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(splitX, yOf(convertPrice(before[splitIdx].close)));
    visPreds.forEach((p, i) => ctx.lineTo(xOf(splitIdx + 1 + i), yOf(convertPrice(p.q50))));
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "#9d3835";
    ctx.font = "600 11px Kanit, sans-serif";
    ctx.fillText(`WALK-FORWARD CUTOFF: ${bt.anchor_date}`, Math.max(16, splitX - 70), pad.top - 8);

    ctx.fillStyle = "#4a4144";
    ctx.font = "400 11px Poppins, sans-serif";
    ctx.fillText(before[0].date, pad.left, H - 8);
    if (preds.length) {
      ctx.fillText(preds[preds.length - 1].date, W - pad.right - 72, H - 8);
    }

    if (state.backtestHover && preds[state.backtestHover.index]) {
      const pIdx = state.backtestHover.index;
      const pt = preds[pIdx];
      const hx = xOf(splitIdx + 1 + pIdx);
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#202020";
      ctx.beginPath();
      ctx.moveTo(hx, pad.top);
      ctx.lineTo(hx, H - pad.bottom);
      ctx.stroke();
      ctx.restore();

      const errPct = (((pt.q50 - pt.actual) / Math.max(pt.actual, 1e-6)) * 100).toFixed(2);
      ctx.fillStyle = "rgba(255,255,255,0.94)";
      ctx.fillRect(pad.left + 8, pad.top + 6, 280, 52);
      ctx.strokeStyle = "#202020";
      ctx.strokeRect(pad.left + 8, pad.top + 6, 280, 52);
      ctx.fillStyle = "#202020";
      ctx.font = "600 11px Kanit, sans-serif";
      ctx.fillText(`Replay Date: ${pt.date} (+${pt.days_ahead}D)`, pad.left + 16, pad.top + 22);
      ctx.font = "400 11px Poppins, sans-serif";
      ctx.fillText(
        `Pred: ${formatCurrency(pt.q50)} vs Actual: ${formatCurrency(pt.actual)} (${errPct}%)`,
        pad.left + 16,
        pad.top + 42
      );
    }
  }

  function renderPulseSection() {
    if (!state.pulsePayload) return;
    const { gsr_radar, physical_squeeze, central_bank_tracker, correlations, arbitrage_hubs, news } = state.pulsePayload;

    document.getElementById("kmGsrVal").textContent = `${gsr_radar.current_gsr}x`;
    document.getElementById("kmGsrMarker").style.left = `${gsr_radar.percentile_15y}%`;
    document.getElementById("kmGsrDesc").textContent = gsr_radar.swap_recommendation;

    document.getElementById("kmSqueezeVal").textContent = `${physical_squeeze.squeeze_index} / 100`;
    document.getElementById("kmSqueezeMarker").style.left = `${Math.min(96, physical_squeeze.squeeze_index)}%`;
    document.getElementById("kmSqueezeDesc").textContent = `${physical_squeeze.paper_to_registered_coverage} • Registered YoY: ${physical_squeeze.comex_shfe_drawdown_yoy_pct}%`;

    document.getElementById("kmCbVal").textContent = `${central_bank_tracker.monthly_net_purchases_tonnes} Tonnes/Mo`;
    document.getElementById("kmCbDesc").textContent = `US Debt/GDP: ${central_bank_tracker.us_debt_to_gdp_pct}% • Buyers: ${central_bank_tracker.top_accumulators.slice(0, 3).join(", ")}`;

    // Render Cross-Asset Correlations
    const corrEl = document.getElementById("kmCorrList");
    if (corrEl && correlations) {
      corrEl.innerHTML = correlations
        .map((c) => {
          const isPos = c.long_1y >= 0;
          const sign = isPos ? "+" : "";
          return `
            <div class="mp-corr-row">
              <strong>${c.asset}</strong>
              <span class="${isPos ? "km-badge-pos" : "km-badge-neg"}">30D: ${c.short_30d >= 0 ? "+" : ""}${
            c.short_30d
          } | 1Y: ${sign}${c.long_1y}</span>
              <span style="color: var(--ink-soft);">${c.regime}</span>
            </div>
          `;
        })
        .join("");
    }

    // Render 5-Hub Global Bullion Arbitrage Table
    const arbBody = document.getElementById("kmArbBody");
    if (arbBody && arbitrage_hubs) {
      arbBody.innerHTML = arbitrage_hubs
        .map(
          (row) => `
          <tr>
            <td><strong>${row.hub}</strong><br/><span style="font-size:0.72rem;color:var(--ink-soft);">${row.contract}</span></td>
            <td style="font-family:Kanit,sans-serif;font-weight:600;">${row.quote}</td>
            <td>${row.spread_vs_spot}</td>
            <td style="font-size:0.75rem;color:var(--ink-soft);">${row.status}</td>
          </tr>
        `
        )
        .join("");
    }

    const newsGrid = document.getElementById("kmNewsGrid");
    if (newsGrid && news) {
      newsGrid.innerHTML = news.headlines
        .map((h) => {
          const cls = h.polarity > 0 ? "is-pos" : h.polarity < 0 ? "is-neg" : "is-neu";
          const bpsSign = h.bps_impact >= 0 ? "+" : "";
          return `
            <article class="km-news-card">
              <div class="km-news-card__top">
                <span>${h.source} • ${h.category} • ${h.timestamp}</span>
                <span class="km-finbert-badge ${cls}">FinBERT ${h.sentiment_label} • ${bpsSign}${h.bps_impact} bps</span>
              </div>
              <h4>${h.headline}</h4>
              <p>${h.summary}</p>
            </article>
          `;
        })
        .join("");
    }
  }

  function updateLiveClockDisplay() {
    const el = document.getElementById("mpLiveClock");
    if (!el) return;
    const now = new Date();
    const locale = state.lang === "ja" ? "ja-JP" : state.lang === "hi" ? "hi-IN" : "en-US";
    const datePart = now.toLocaleDateString(locale, {
      weekday: "short",
      year: "numeric",
      month: "short",
      day: "2-digit",
    });
    const timePart = now.toLocaleTimeString(locale, {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    el.textContent = `${datePart.toUpperCase()} • ${timePart} IST`;
  }

  function initLiveClock() {
    updateLiveClockDisplay();
    setInterval(updateLiveClockDisplay, 1000);
  }

  function updateChatContextBadge() {
    const ctxEl = document.getElementById("mpChatContext");
    if (ctxEl) {
      ctxEl.textContent = `${state.metal.toUpperCase()} • ${state.horizon} • ${state.lang.toUpperCase()}`;
    }
  }

  function applyLanguage(lang) {
    state.lang = ["en", "ja", "hi"].includes(lang) ? lang : "en";
    document.documentElement.lang = state.lang;
    const dict = I18N_DICT[state.lang] || I18N_DICT.en;

    document.querySelectorAll(".mp-lang-btn").forEach((btn) => {
      btn.classList.toggle("is-active", btn.getAttribute("data-lang") === state.lang);
    });

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (key && dict[key]) {
        el.innerHTML = dict[key];
      }
    });

    const rotEl = document.getElementById("mpRotatingHeroWord");
    if (rotEl && dict.rotatingWords && dict.rotatingWords.length) {
      if (state.rotTimeout) clearTimeout(state.rotTimeout);
      rotEl.classList.remove("is-switching");
      rotEl.textContent = dict.rotatingWords[0];
    }

    const chatInput = document.getElementById("mpChatInput");
    if (chatInput && dict.chat_placeholder) {
      chatInput.placeholder = dict.chat_placeholder;
    }

    const welcomeEl = document.getElementById("mpChatWelcomeMsg");
    if (welcomeEl && dict.chat_welcome) {
      welcomeEl.innerHTML = dict.chat_welcome;
    }

    updateLiveClockDisplay();
    updateChatContextBadge();
    renderMetalTabs();
    renderKpiStrip();
    renderMainFanChart(false);
  }

  function formatBotMarkdown(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  }

  async function sendChatQuestion(questionText) {
    const q = String(questionText || "").trim();
    if (!q) return;
    const msgBox = document.getElementById("mpChatMessages");
    const input = document.getElementById("mpChatInput");
    if (input) input.value = "";

    if (msgBox) {
      const userBubble = document.createElement("div");
      userBubble.className = "mp-chat-msg mp-chat-msg--user";
      userBubble.textContent = q;
      msgBox.appendChild(userBubble);
      msgBox.scrollTop = msgBox.scrollHeight;
    }

    const loadingBubble = document.createElement("div");
    loadingBubble.className = "mp-chat-msg mp-chat-msg--bot";
    loadingBubble.textContent =
      state.lang === "ja"
        ? "マルチホライズンMLエンジンを照会中..."
        : state.lang === "hi"
        ? "मल्टी-होराइजन ML इंजन से गणना की जा रही है..."
        : "Querying Three-Tier ML & TreeSHAP Engine...";
    if (msgBox) {
      msgBox.appendChild(loadingBubble);
      msgBox.scrollTop = msgBox.scrollHeight;
    }

    try {
      const res = await fetchJson("/api/chat/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: q,
          metal: state.metal,
          horizon: state.horizon,
          lang: state.lang,
          usdinr: state.usdinr,
        }),
      });
      loadingBubble.innerHTML = formatBotMarkdown(res.answer);
      if (msgBox) msgBox.scrollTop = msgBox.scrollHeight;
    } catch (err) {
      loadingBubble.textContent = `Error: ${err.message || "Unable to reach AI Analyst"}`;
    }
  }

  function initChatbotWidget() {
    const toggleBtn = document.getElementById("mpChatToggleBtn");
    const closeBtn = document.getElementById("mpChatCloseBtn");
    const win = document.getElementById("mpChatWindow");
    const form = document.getElementById("mpChatForm");
    const input = document.getElementById("mpChatInput");

    if (toggleBtn && win) {
      toggleBtn.addEventListener("click", () => {
        const isOpen = win.classList.toggle("is-open");
        toggleBtn.setAttribute("aria-expanded", isOpen ? "true" : "false");
        if (isOpen && input) {
          setTimeout(() => input.focus(), 120);
        }
      });
    }
    if (closeBtn && win && toggleBtn) {
      closeBtn.addEventListener("click", () => {
        win.classList.remove("is-open");
        toggleBtn.setAttribute("aria-expanded", "false");
      });
    }
    if (form && input) {
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        sendChatQuestion(input.value);
      });
    }
    document.querySelectorAll(".mp-chat-chip[data-chat-q]").forEach((chip) => {
      chip.addEventListener("click", () => {
        const q = chip.getAttribute("data-chat-q") || chip.textContent;
        if (win && !win.classList.contains("is-open")) {
          win.classList.add("is-open");
        }
        sendChatQuestion(q);
      });
    });
  }

  function initThemeAnimations() {
    // 1. Rotating Hero Accent Word (Always high-contrast & localized)
    let wIdx = 0;
    const rotEl = document.getElementById("mpRotatingHeroWord");
    if (rotEl) {
      setInterval(() => {
        const dict = I18N_DICT[state.lang] || I18N_DICT.en;
        const words = dict.rotatingWords || I18N_DICT.en.rotatingWords;
        rotEl.classList.add("is-switching");
        state.rotTimeout = setTimeout(() => {
          const curDict = I18N_DICT[state.lang] || I18N_DICT.en;
          const curWords = curDict.rotatingWords || I18N_DICT.en.rotatingWords;
          wIdx = (wIdx + 1) % curWords.length;
          rotEl.textContent = curWords[wIdx];
          rotEl.classList.remove("is-switching");
        }, 180);
      }, 3200);
    }

    // 2. Two-Frame Animated Ukiyo-e Dancer (from Design Source)
    const pose1 = document.getElementById("mpDancerPose1");
    const pose2 = document.getElementById("mpDancerPose2");
    if (pose1 && pose2) {
      let showFirst = true;
      setInterval(() => {
        showFirst = !showFirst;
        pose1.style.opacity = showFirst ? "1" : "0";
        pose2.style.opacity = showFirst ? "0" : "1";
      }, 950);
    }

    // 3. Scroll-Triggered Reveal & Parallax + Paper Scrollbar
    const revealEls = document.querySelectorAll("[data-mp-reveal]");
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add("is-visible");
            }
          });
        },
        { threshold: 0.08 }
      );
      revealEls.forEach((el) => io.observe(el));
    } else {
      revealEls.forEach((el) => el.classList.add("is-visible"));
    }

    const thumb = document.getElementById("mpScrollbarThumb");
    window.addEventListener(
      "scroll",
      () => {
        const scrollY = window.scrollY || window.pageYOffset;
        const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
        const pct = Math.min(1, Math.max(0, scrollY / maxScroll));
        if (thumb) {
          thumb.style.top = `${pct * 74}%`;
        }
        document.documentElement.style.setProperty("--mp-parallax-y", `${scrollY * 0.1}px`);
      },
      { passive: true }
    );

    // 4. Chamfered Floral FAQ / Methodology Accordion
    document.querySelectorAll(".mp-faq-q").forEach((btn) => {
      btn.addEventListener("click", () => {
        const item = btn.closest(".mp-faq-item");
        if (!item) return;
        const isOpen = item.classList.contains("is-open");
        document.querySelectorAll(".mp-faq-item").forEach((el) => el.classList.remove("is-open"));
        if (!isOpen) item.classList.add("is-open");
      });
    });
  }

  function bindEvents() {
    const skipBtn = document.getElementById("mpSkipPreloaderBtn");
    if (skipBtn) {
      skipBtn.addEventListener("click", () => {
        const pre = document.getElementById("mpPreloader");
        if (pre) pre.classList.add("is-dismissed");
        toggleSkeletonState(false);
      });
    }

    // Language switcher (EN / 日本語 / हिन्दी)
    document.querySelectorAll(".mp-lang-btn[data-lang]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const lang = btn.getAttribute("data-lang") || "en";
        applyLanguage(lang);
      });
    });

    // Horizon switcher
    document.querySelectorAll(".km-horizon-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".km-horizon-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.horizon = btn.getAttribute("data-horizon") || "1M";
        state.fanHover = null;
        updateChatContextBadge();
        if (state.stressedPayload) {
          triggerMacroShockSimulation();
        } else {
          renderKpiStrip();
          renderMainFanChart(true);
          renderShapAndRegimes();
        }
      });
    });

    // Unit/Currency switcher
    document.querySelectorAll(".km-unit-btn[data-unit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".km-unit-btn[data-unit]").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.unitMode = btn.getAttribute("data-unit") || "USD_OZ";
        renderKpiStrip();
        drawFanCanvas();
        drawBacktestCanvas();
        if (state.anyDayPayload) renderAnyDayCard(state.anyDayPayload);
      });
    });

    // Any-Day Date Picker & Catalyst Presets
    const dateBtn = document.getElementById("kmAnyDayBtn");
    const dateInput = document.getElementById("kmTargetDateInput");
    if (dateBtn && dateInput) {
      dateBtn.addEventListener("click", () => runAnyDayPrediction(dateInput.value));
      dateInput.addEventListener("change", () => runAnyDayPrediction(dateInput.value));
    }
    document.querySelectorAll(".km-preset-chip[data-date]").forEach((chip) => {
      chip.addEventListener("click", () => {
        const dt = chip.getAttribute("data-date");
        if (dt && dateInput) {
          dateInput.value = dt;
          runAnyDayPrediction(dt);
        }
      });
    });

    // Macro & Duty Sliders
    const bindSlider = (id, valId, key, fmt) => {
      const el = document.getElementById(id);
      const vEl = document.getElementById(valId);
      if (!el) return;
      el.addEventListener("input", () => {
        if (window.getSelection) {
          window.getSelection().removeAllRanges();
        }
        state[key] = Number(el.value);
        if (vEl) vEl.textContent = fmt(state[key]);
        scheduleMacroShockSimulation();
      });
    };
    bindSlider("kmSliderUsdinr", "kmValUsdinr", "usdinr", (v) => `₹${v.toFixed(2)}`);
    bindSlider("kmSliderCpi", "kmValCpi", "cpi", (v) => `${v.toFixed(1)}%`);
    bindSlider("kmSliderCuts", "kmValCuts", "rateCutsBps", (v) => `${v >= 0 ? "+" : ""}${v} bps`);
    bindSlider("kmSliderRealYield", "kmValRealYield", "realYieldDeltaBps", (v) => `${v >= 0 ? "+" : ""}${v} bps`);
    bindSlider("kmSliderDuty", "kmValDuty", "bcdPct", (v) => `${v.toFixed(1)}%`);
    bindSlider("kmSliderGst", "kmValGst", "gstPct", (v) => `${v.toFixed(1)}%`);

    const resetBtn = document.getElementById("kmResetStressBtn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        state.usdinr = 86.48;
        state.cpi = 2.7;
        state.rateCutsBps = 50;
        state.realYieldDeltaBps = 0;
        state.bcdPct = 6.0;
        state.gstPct = 3.0;
        state.stressedPayload = null;
        document.getElementById("kmSliderUsdinr").value = 86.48;
        document.getElementById("kmValUsdinr").textContent = "₹86.48";
        document.getElementById("kmSliderCpi").value = 2.7;
        document.getElementById("kmValCpi").textContent = "2.7%";
        document.getElementById("kmSliderCuts").value = 50;
        document.getElementById("kmValCuts").textContent = "+50 bps";
        document.getElementById("kmSliderRealYield").value = 0;
        document.getElementById("kmValRealYield").textContent = "0 bps";
        document.getElementById("kmSliderDuty").value = 6.0;
        document.getElementById("kmValDuty").textContent = "6.0%";
        document.getElementById("kmSliderGst").value = 3.0;
        document.getElementById("kmValGst").textContent = "3.0%";
        renderKpiStrip();
        renderMainFanChart(true);
        renderParityLedger();
        renderShapAndRegimes();
        if (state.anyDayPayload) renderAnyDayCard(state.anyDayPayload);
      });
    }

    // Portfolio SIP Simulator bindings
    ["mpSipMonthly", "mpSipLumpSum", "mpSipYears", "mpWeightGold", "mpWeightSilver", "mpWeightPlatinum", "mpWeightCopper"].forEach(
      (id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("input", () => runPortfolioSipSimulation());
      }
    );
    document.querySelectorAll("[data-sip-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const [g, s, p, c] = (btn.getAttribute("data-sip-preset") || "50,25,10,15").split(",").map(Number);
        document.getElementById("mpWeightGold").value = g;
        document.getElementById("mpWeightSilver").value = s;
        document.getElementById("mpWeightPlatinum").value = p;
        document.getElementById("mpWeightCopper").value = c;
        runPortfolioSipSimulation();
      });
    });

    // Main Fan Chart Crosshair
    const fanCanvas = document.getElementById("kmFanChart");
    if (fanCanvas) {
      fanCanvas.addEventListener("mousemove", (e) => {
        if (!state.horizonPayload) return;
        const geom = getFanChartGeometry(fanCanvas);
        const mouseX = e.clientX - geom.rect.left;
        const ratio = (mouseX - geom.pad.left) / geom.plotW;
        const globalIdx = Math.min(
          geom.totalPoints - 1,
          Math.max(0, Math.round(ratio * (geom.totalPoints - 1)))
        );
        if (globalIdx <= geom.splitIdx) {
          state.fanHover = { type: "hist", localIdx: globalIdx, globalIdx };
        } else {
          const fIdx = Math.min(
            geom.activeSeries.length - 1,
            Math.max(0, globalIdx - (geom.splitIdx + 1))
          );
          state.fanHover = { type: "forecast", localIdx: fIdx, globalIdx: geom.splitIdx + 1 + fIdx };
        }
        drawFanCanvas();
      });
      fanCanvas.addEventListener("mouseleave", () => {
        state.fanHover = null;
        drawFanCanvas();
      });
    }

    // Backtest Chart Crosshair
    const btCanvas = document.getElementById("kmBacktestChart");
    if (btCanvas) {
      btCanvas.addEventListener("mousemove", (e) => {
        if (!state.backtestPayload) return;
        const rect = btCanvas.getBoundingClientRect();
        const padLeft = 18;
        const padRight = 78;
        const plotW = Math.max(10, rect.width - padLeft - padRight);
        const beforeLen = (state.backtestPayload.history_before || []).length;
        const preds = state.backtestPayload.predicted_series || [];
        const total = beforeLen + preds.length;
        const ratio = (e.clientX - rect.left - padLeft) / plotW;
        const globalIdx = Math.round(ratio * (total - 1));
        const pIdx = Math.min(preds.length - 1, Math.max(0, globalIdx - beforeLen));
        state.backtestHover = { index: pIdx };
        drawBacktestCanvas();
      });
      btCanvas.addEventListener("mouseleave", () => {
        state.backtestHover = null;
        drawBacktestCanvas();
      });
    }

    window.addEventListener("resize", () => {
      drawFanCanvas();
      drawBacktestCanvas();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initLiveClock();
    initThemeAnimations();
    initChatbotWidget();
    bindEvents();
    loadAllForMetal("gold");
  });
})();
