// Macro-Financial Valuation Lab — JS port of src/economics_models/core.py.
// Keep math in lockstep with the Python module.

(function () {
  "use strict";

  // ---------- Model math ----------

  function production(K, A, L, alpha) {
    return Math.pow(K, alpha) * Math.pow(A * L, 1 - alpha);
  }

  function nextK(K, Y, p) {
    return (1 + p.n + p.g) * (K + p.s * Y - p.delta * K);
  }

  function simulateSolow(p) {
    const T = p.T;
    const K = new Float64Array(T);
    const Y = new Float64Array(T);
    const A = new Float64Array(T);
    const L = new Float64Array(T);
    const growth = new Float64Array(T - 1);
    K[0] = p.K0;
    A[0] = p.A0;
    L[0] = p.L0;
    Y[0] = production(K[0], A[0], L[0], p.alpha);
    for (let t = 1; t < T; t++) {
      K[t] = nextK(K[t - 1], Y[t - 1], p);
      L[t] = L[t - 1] * (1 + p.n);
      A[t] = A[t - 1] * (1 + p.g);
      Y[t] = production(K[t], A[t], L[t], p.alpha);
      growth[t - 1] = ((Y[t] - Y[t - 1]) / Y[t - 1]) * 100;
    }
    return { K: K, Y: Y, A: A, L: L, growth: growth };
  }

  function steadyStateGrowth(p) {
    const path = simulateSolow(p);
    return path.growth[path.growth.length - 1];
  }

  function justifiedPE(solow, fin) {
    const realG = steadyStateGrowth(solow) / 100;
    const nominalG = realG + fin.inflation;
    const rf = realG + fin.inflation + fin.termPremium;
    const rr = rf + fin.beta * fin.erp;
    const eg = nominalG * fin.egf;
    const pe = eg >= rr ? null : (1 - fin.retention) / (rr - eg);
    return {
      realGrowthRate: realG,
      nominalGrowthRate: nominalG,
      riskFreeRate: rf,
      requiredReturn: rr,
      earningsGrowthRate: eg,
      justifiedPE: pe,
    };
  }

  function gdpSensitivity(solow, gMin, gMax, steps) {
    const xs = new Float64Array(steps);
    const ys = new Float64Array(steps);
    for (let i = 0; i < steps; i++) {
      const g = gMin + ((gMax - gMin) * i) / (steps - 1);
      xs[i] = g;
      ys[i] = steadyStateGrowth(Object.assign({}, solow, { g: g }));
    }
    return { xs: xs, ys: ys };
  }

  function peSensitivity(solow, fin, gMin, gMax, steps) {
    const xs = new Float64Array(steps);
    const ys = new Array(steps);
    for (let i = 0; i < steps; i++) {
      const g = gMin + ((gMax - gMin) * i) / (steps - 1);
      xs[i] = g;
      const res = justifiedPE(Object.assign({}, solow, { g: g }), fin);
      ys[i] = res.justifiedPE;
    }
    return { xs: xs, ys: ys };
  }

  // Box-Muller normal(mean, std)
  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function monteCarlo(solow, fin, mc) {
    const n = mc.numSims;
    const pe = [];
    const eg = [];
    for (let i = 0; i < n; i++) {
      const gDraw = mc.gMean + mc.gStd * randn();
      const piDraw = mc.inflMean + mc.inflStd * randn();
      const tpDraw = mc.tpMean + mc.tpStd * randn();
      const finDraw = Object.assign({}, fin, {
        inflation: piDraw,
        termPremium: tpDraw,
      });
      const res = justifiedPE(Object.assign({}, solow, { g: gDraw }), finDraw);
      if (res.justifiedPE !== null && isFinite(res.justifiedPE) && res.justifiedPE > 0) {
        pe.push(res.justifiedPE);
      }
      eg.push(res.earningsGrowthRate * 100);
    }
    return { pe: pe, eg: eg };
  }

  // ---------- Formatting ----------

  const fmtPct = (v, d) => (v == null || !isFinite(v) ? "—" : (v * 100).toFixed(d || 2) + "%");
  const fmtNum = (v, d) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d == null ? 2 : d));

  // ---------- Defaults & state ----------

  const defaults = {
    s: 0.20, n: 0.005, g: 0.02, delta: 0.05, alpha: 0.35, T: 200,
    A0: 1, L0: 100, K0: 10000,
    beta: 1.0, erp: 0.05, inflation: 0.02, termPremium: 0.01,
    retention: 0.35, egf: 1.10,
    numSims: 5000, gStd: 0.005, inflStd: 0.005, tpStd: 0.002,
  };

  const formats = {
    s: { kind: "pct", d: 1 }, n: { kind: "pct", d: 2 }, g: { kind: "pct", d: 2 },
    delta: { kind: "pct", d: 1 }, alpha: { kind: "pct", d: 0 }, T: { kind: "int" },
    beta: { kind: "num", d: 2 }, erp: { kind: "pct", d: 2 },
    inflation: { kind: "pct", d: 2 }, termPremium: { kind: "pct", d: 2 },
    retention: { kind: "pct", d: 0 }, egf: { kind: "num", d: 2 },
    numSims: { kind: "int" }, gStd: { kind: "pct", d: 2 },
    inflStd: { kind: "pct", d: 2 }, tpStd: { kind: "pct", d: 2 },
  };

  function readControls() {
    const v = {};
    Object.keys(defaults).forEach((k) => {
      const el = document.getElementById(k);
      if (el) v[k] = parseFloat(el.value);
      else v[k] = defaults[k];
    });
    v.A0 = defaults.A0;
    v.L0 = defaults.L0;
    v.K0 = defaults.K0;
    return v;
  }

  function fmt(id, value) {
    const f = formats[id];
    if (!f) return value;
    if (f.kind === "pct") return (value * 100).toFixed(f.d) + "%";
    if (f.kind === "int") return Math.round(value).toString();
    return Number(value).toFixed(f.d);
  }

  function syncOutputs(state) {
    document.querySelectorAll(".param").forEach((p) => {
      const id = p.dataset.id;
      const out = p.querySelector("output");
      if (out) out.textContent = fmt(id, state[id]);
    });
  }

  function setKpi(id, text, invalid) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    const card = el.parentElement;
    if (id === "kpi-pe" && card) {
      card.classList.toggle("invalid", !!invalid);
    }
  }

  // ---------- Plotly layout ----------

  function plotlyLayout(title, xtitle, ytitle) {
    const isLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    const fontColor = isLight ? "#0f172a" : "#e6ecff";
    const grid = isLight ? "#dbe3f5" : "#243260";
    return {
      title: { text: title, font: { size: 14, color: fontColor } },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: fontColor, family: "Inter, system-ui, sans-serif" },
      margin: { l: 60, r: 24, t: 50, b: 50 },
      xaxis: { title: xtitle, gridcolor: grid, zerolinecolor: grid },
      yaxis: { title: ytitle, gridcolor: grid, zerolinecolor: grid },
      legend: { orientation: "h", y: -0.18 },
    };
  }

  const plotlyConfig = { displaylogo: false, responsive: true };

  // ---------- Renderers ----------

  function renderSolow(state) {
    const path = simulateSolow(state);
    const xs = Array.from({ length: state.T - 1 }, (_, i) => i + 1);
    const trace = {
      x: xs,
      y: Array.from(path.growth),
      type: "scattergl",
      mode: "lines",
      name: "Output growth",
      line: { color: "#5eead4", width: 2.5 },
    };
    const target = ((1 + state.n) * (1 + state.g) - 1) * 100;
    const ref = {
      x: [xs[0], xs[xs.length - 1]],
      y: [target, target],
      type: "scatter",
      mode: "lines",
      name: "(1+n)(1+g)−1",
      line: { color: "#818cf8", width: 1.5, dash: "dash" },
    };
    const layout = plotlyLayout(
      "Solow-Swan output growth path",
      "Period",
      "YoY growth (%)"
    );
    Plotly.react("chart-solow", [trace, ref], layout, plotlyConfig);
  }

  function renderGdp(state) {
    const sweep = gdpSensitivity(state, 0.005, 0.04, 36);
    const trace = {
      x: Array.from(sweep.xs).map((g) => g * 100),
      y: Array.from(sweep.ys),
      type: "scatter",
      mode: "lines+markers",
      name: "Steady-state growth",
      line: { color: "#818cf8", width: 2 },
      marker: { color: "#818cf8", size: 6 },
    };
    Plotly.react(
      "chart-gdp",
      [trace],
      plotlyLayout("Steady-state growth vs. tech progress", "Tech progress g (%)", "Growth (%)"),
      plotlyConfig
    );
  }

  function renderPe(state) {
    const fin = {
      beta: state.beta, erp: state.erp, inflation: state.inflation,
      termPremium: state.termPremium, retention: state.retention, egf: state.egf,
    };
    const sweep = peSensitivity(state, fin, 0.005, 0.04, 36);
    const trace = {
      x: Array.from(sweep.xs).map((g) => g * 100),
      y: sweep.ys.map((v) => (v === null ? null : v)),
      type: "scatter",
      mode: "lines+markers",
      name: "Justified P/E",
      line: { color: "#5eead4", width: 2 },
      marker: { color: "#5eead4", size: 6 },
      connectgaps: false,
    };
    Plotly.react(
      "chart-pe",
      [trace],
      plotlyLayout("Justified P/E vs. tech progress", "Tech progress g (%)", "Justified forward P/E"),
      plotlyConfig
    );
  }

  function renderMonteCarlo(state) {
    const fin = {
      beta: state.beta, erp: state.erp, inflation: state.inflation,
      termPremium: state.termPremium, retention: state.retention, egf: state.egf,
    };
    const mc = {
      numSims: Math.round(state.numSims),
      gMean: state.g, gStd: state.gStd,
      inflMean: state.inflation, inflStd: state.inflStd,
      tpMean: state.termPremium, tpStd: state.tpStd,
    };
    const out = monteCarlo(state, fin, mc);
    const summarize = (xs) => {
      if (!xs.length) return { mean: NaN, std: NaN };
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
      const v = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / xs.length;
      return { mean: mean, std: Math.sqrt(v) };
    };
    const peStats = summarize(out.pe);
    const egStats = summarize(out.eg);

    const peTrace = {
      x: out.pe, type: "histogram", nbinsx: 35, name: "P/E",
      marker: { color: "#818cf8", line: { color: "rgba(255,255,255,0.6)", width: 1 } },
    };
    const peLayout = plotlyLayout(
      `P/E distribution — μ=${fmtNum(peStats.mean)}, σ=${fmtNum(peStats.std)}`,
      "P/E ratio",
      "Frequency"
    );
    if (isFinite(peStats.mean)) {
      peLayout.shapes = [
        { type: "line", x0: peStats.mean, x1: peStats.mean, yref: "paper", y0: 0, y1: 1, line: { color: "#e6ecff", dash: "dash", width: 1 } },
      ];
    }
    Plotly.react("chart-mc-pe", [peTrace], peLayout, plotlyConfig);

    const egTrace = {
      x: out.eg, type: "histogram", nbinsx: 35, name: "Earnings growth",
      marker: { color: "#5eead4", line: { color: "rgba(255,255,255,0.6)", width: 1 } },
    };
    const egLayout = plotlyLayout(
      `Earnings growth distribution — μ=${fmtNum(egStats.mean)}%, σ=${fmtNum(egStats.std)}%`,
      "Earnings growth (%)",
      "Frequency"
    );
    if (isFinite(egStats.mean)) {
      egLayout.shapes = [
        { type: "line", x0: egStats.mean, x1: egStats.mean, yref: "paper", y0: 0, y1: 1, line: { color: "#e6ecff", dash: "dash", width: 1 } },
      ];
    }
    Plotly.react("chart-mc-eg", [egTrace], egLayout, plotlyConfig);
  }

  // ---------- KPI update ----------

  function updateKpis(state) {
    const fin = {
      beta: state.beta, erp: state.erp, inflation: state.inflation,
      termPremium: state.termPremium, retention: state.retention, egf: state.egf,
    };
    const res = justifiedPE(state, fin);
    setKpi("kpi-real", fmtPct(res.realGrowthRate));
    setKpi("kpi-nominal", fmtPct(res.nominalGrowthRate));
    setKpi("kpi-rf", fmtPct(res.riskFreeRate));
    setKpi("kpi-rr", fmtPct(res.requiredReturn));
    if (res.justifiedPE === null) {
      setKpi("kpi-pe", "Invalid (g_e ≥ k_e)", true);
    } else {
      setKpi("kpi-pe", fmtNum(res.justifiedPE), false);
    }
  }

  // ---------- Tab switching ----------

  const renderers = {
    solow: renderSolow,
    gdp: renderGdp,
    pe: renderPe,
    mc: renderMonteCarlo,
  };
  const tabContainer = {
    solow: "chart-solow", gdp: "chart-gdp", pe: "chart-pe", mc: "chart-mc",
  };
  let activeTab = "solow";

  function showTab(tab) {
    activeTab = tab;
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.tab === tab)
    );
    Object.entries(tabContainer).forEach(([key, id]) => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle("hidden", key !== tab);
    });
    const state = readControls();
    renderers[tab](state);
    // resize plotly when becoming visible
    requestAnimationFrame(() => {
      const id = tabContainer[tab];
      if (tab === "mc") {
        Plotly.Plots.resize(document.getElementById("chart-mc-pe"));
        Plotly.Plots.resize(document.getElementById("chart-mc-eg"));
      } else {
        Plotly.Plots.resize(document.getElementById(id));
      }
    });
  }

  // ---------- Wiring ----------

  function debounce(fn, ms) {
    let t = null;
    return function () {
      const args = arguments;
      const ctx = this;
      clearTimeout(t);
      t = setTimeout(() => fn.apply(ctx, args), ms);
    };
  }

  function refreshActive() {
    const state = readControls();
    syncOutputs(state);
    updateKpis(state);
    if (activeTab !== "mc") {
      renderers[activeTab](state);
    }
  }
  const debouncedRefresh = debounce(refreshActive, 30);

  function setDefaults() {
    Object.keys(defaults).forEach((k) => {
      const el = document.getElementById(k);
      if (el) el.value = defaults[k];
    });
    refreshActive();
    if (activeTab === "mc") renderers.mc(readControls());
  }

  function init() {
    Object.keys(defaults).forEach((k) => {
      const el = document.getElementById(k);
      if (el) el.addEventListener("input", debouncedRefresh);
    });
    document.querySelectorAll(".tab").forEach((t) =>
      t.addEventListener("click", () => showTab(t.dataset.tab))
    );
    document.getElementById("run-mc").addEventListener("click", () =>
      renderers.mc(readControls())
    );
    document.getElementById("reset-btn").addEventListener("click", setDefaults);

    refreshActive();
    showTab("solow");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // expose for console exploration / cross-checking with Python
  window.MacroFin = {
    simulateSolow, steadyStateGrowth, justifiedPE,
    gdpSensitivity, peSensitivity, monteCarlo, defaults,
  };
})();
