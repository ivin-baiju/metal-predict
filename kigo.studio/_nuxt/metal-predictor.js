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
    await sleep(650);
    setPreloaderStep(1, 54, "02 / 04 • Calibrating Tier 1–2 XGBoost & LightGBM Quantiles...");
    await sleep(650);
    setPreloaderStep(2, 82, "03 / 04 • Simulating 1,000 VECM Jump-Diffusion Monte Carlo Paths...");
    await sleep(650);
    setPreloaderStep(3, 100, "04 / 04 • Extracting TreeSHAP & FinBERT Sentiment Pulse...");
    await sleep(450);
    const pre = document.getElementById("mpPreloader");
    if (pre) pre.classList.add("is-dismissed");
  }

  async function loadAllForMetal(metal) {
    state.metal = metal;
    state.stressedPayload = null;
    state.fanHover = null;
    toggleSkeletonState(true);

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
        // Let the slow Washi-Paper Shimmer Skeleton remain visible on the page for 1.1s after the curtain lifts
        await sleep(1100);
        state.initialLoadDone = true;
      } else {
        const [[horizonsData, pulseData, backtestData]] = await Promise.all([
          fetchPromise,
          sleep(850), // Deliberate skeleton shimmer transition on metal switch
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
    const summaries = state.horizonPayload.all_metals || [];
    container.innerHTML = summaries
      .map((item) => {
        const active = item.metal === state.metal ? "is-active" : "";
        const chgSign = item.change_24h_pct >= 0 ? "+" : "";
        return `
          <button type="button" class="km-metal-tab ${active}" data-metal="${item.metal}">
            <span class="km-metal-tab__kanji" lang="ja">${item.kanji}</span>
            <span class="km-metal-tab__meta">
              <span class="km-metal-tab__name">${item.name} • ${item.symbol}</span>
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

    document.getElementById("kmKpiHorizonLabel").textContent = `${state.horizon} Median Projection (${hObj.days}D)`;
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
    const p = state.horizonPayload;
    const hObj = p.horizons[state.horizon];
    document.getElementById("kmActiveChartTitle").textContent = `${p.spec.name} (${p.spec.symbol}) — ${state.horizon} Probability Fan`;
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

  function initThemeAnimations() {
    // 1. Rotating Hero Sakura Word (Never clips or disappears)
    const words = ["forecasts", "quantiles", "regimes", "parity", "conviction"];
    let wIdx = 0;
    const rotEl = document.getElementById("mpRotatingHeroWord");
    if (rotEl) {
      setInterval(() => {
        rotEl.classList.add("is-switching");
        setTimeout(() => {
          wIdx = (wIdx + 1) % words.length;
          rotEl.textContent = words[wIdx];
          rotEl.classList.remove("is-switching");
        }, 280);
      }, 3000);
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

    // Horizon switcher
    document.querySelectorAll(".km-horizon-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".km-horizon-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.horizon = btn.getAttribute("data-horizon") || "1M";
        state.fanHover = null;
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
    initThemeAnimations();
    bindEvents();
    loadAllForMetal("gold");
  });
})();
