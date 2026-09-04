// Macro-Financial Valuation Lab — JS port of src/economics_models/core.py.
// Parity with the Python module is enforced by tests/test_js_parity.py.
// The model section is pure (no DOM); the UI section only runs in a browser.

(function (root) {
  "use strict";

  // =========================================================================
  // Model
  // =========================================================================

  function production(K, A, L, alpha) {
    return Math.pow(K, alpha) * Math.pow(A * L, 1 - alpha);
  }

  // Textbook aggregate law of motion: K_{t+1} = (1 - delta) K_t + s Y_t
  function nextK(K, Y, p) {
    return (1 - p.delta) * K + p.s * Y;
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
    return { K, Y, A, L, growth };
  }

  // Final-period growth for a vector of g's, one pass over T (mirrors Python).
  function simulatedFinalGrowthVec(gArr, p) {
    const n = gArr.length;
    const K = new Float64Array(n).fill(p.K0);
    const A = new Float64Array(n).fill(p.A0);
    const L = new Float64Array(n).fill(p.L0);
    let Y = new Float64Array(n);
    for (let i = 0; i < n; i++) Y[i] = production(K[i], A[i], L[i], p.alpha);
    let Yprev = Y;
    for (let t = 1; t < p.T; t++) {
      Yprev = Y;
      Y = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        K[i] = nextK(K[i], Yprev[i], p);
        L[i] = L[i] * (1 + p.n);
        A[i] = A[i] * (1 + gArr[i]);
        Y[i] = production(K[i], A[i], L[i], p.alpha);
      }
    }
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = ((Y[i] - Yprev[i]) / Yprev[i]) * 100;
    return out;
  }

  function closedFormGrowth(n, g) {
    return ((1 + n) * (1 + g) - 1) * 100;
  }

  function steadyStateGrowth(p, method) {
    if (method === "simulated") {
      const path = simulateSolow(p);
      return path.growth[path.growth.length - 1];
    }
    return closedFormGrowth(p.n, p.g);
  }

  function convergenceGap(p) {
    return steadyStateGrowth(p, "simulated") - steadyStateGrowth(p, "closed_form");
  }

  function steadyStateOutputCapitalRatio(p) {
    if (p.s === 0) return Infinity;
    return ((1 + p.n) * (1 + p.g) - (1 - p.delta)) / p.s;
  }

  // Justified forward P/E (P0/E1). NaN when the terminal growth is not below k.
  function peFromRates(rr, eg, gTerm, fin) {
    const payout = 1 - fin.retention;
    const N = fin.highGrowthYears;
    if (N == null) {
      return eg < rr ? payout / (rr - eg) : NaN;
    }
    if (!(gTerm < rr)) return NaN;
    const r = (1 + eg) / (1 + rr);
    const annuity =
      Math.abs(r - 1.0) <= 1e-8 + 1e-5 * 1.0
        ? N / (1 + rr)
        : (1 - Math.pow(r, N)) / (1 - r) / (1 + rr);
    const terminal =
      (Math.pow(1 + eg, N - 1) * (1 + gTerm)) / ((rr - gTerm) * Math.pow(1 + rr, N));
    return payout * (annuity + terminal);
  }

  function valuation(realG, inflation, termPremium, erp, fin) {
    const nominalG = realG + inflation;
    const rf = nominalG + termPremium;
    const rr = rf + fin.beta * erp;
    const eg = nominalG * fin.egf;
    const gTerm = fin.highGrowthYears == null ? eg : nominalG;
    const pe = peFromRates(rr, eg, gTerm, fin);
    const h = 1e-6;
    const peUp = peFromRates(rr + h, eg, gTerm, fin);
    const peDn = peFromRates(rr - h, eg, gTerm, fin);
    const duration = -(peUp - peDn) / (2 * h) / pe;
    return {
      realGrowthRate: realG,
      nominalGrowthRate: nominalG,
      riskFreeRate: rf,
      requiredReturn: rr,
      earningsGrowthRate: eg,
      terminalGrowthRate: gTerm,
      justifiedPE: pe,
      peDuration: duration,
    };
  }

  function justifiedPE(solow, fin, method) {
    const realG = steadyStateGrowth(solow, method) / 100;
    const v = valuation(realG, fin.inflation, fin.termPremium, fin.erp, fin);
    v.justifiedPE = Number.isFinite(v.justifiedPE) ? v.justifiedPE : null;
    v.peDuration = Number.isFinite(v.peDuration) ? v.peDuration : null;
    v.earningsOutgrowGdp = fin.highGrowthYears == null && fin.egf > 1.0;
    return v;
  }

  // ---------- Sensitivities ----------

  function linspace(a, b, steps) {
    const xs = new Float64Array(steps);
    const step = (b - a) / (steps - 1);
    for (let i = 0; i < steps; i++) xs[i] = a + step * i;
    xs[steps - 1] = b;
    return xs;
  }

  function peSensitivity(solow, fin, gMin, gMax, steps, method) {
    const xs = linspace(gMin, gMax, steps);
    const ys = new Float64Array(steps);
    let growth;
    if (method === "simulated") growth = simulatedFinalGrowthVec(xs, solow);
    else {
      growth = new Float64Array(steps);
      for (let i = 0; i < steps; i++) growth[i] = closedFormGrowth(solow.n, xs[i]);
    }
    for (let i = 0; i < steps; i++) {
      ys[i] = valuation(growth[i] / 100, fin.inflation, fin.termPremium, fin.erp, fin).justifiedPE;
    }
    return { xs, ys };
  }

  // Python parameter names -> JS object keys.
  const PARAM_MAP = {
    g: ["solow", "g"],
    n: ["solow", "n"],
    s: ["solow", "s"],
    delta: ["solow", "delta"],
    alpha: ["solow", "alpha"],
    expected_inflation: ["fin", "inflation"],
    term_premium: ["fin", "termPremium"],
    equity_risk_premium: ["fin", "erp"],
    beta: ["fin", "beta"],
    retention_rate: ["fin", "retention"],
    earnings_growth_factor: ["fin", "egf"],
  };

  const DEFAULT_TORNADO_SHOCKS = {
    g: 0.005,
    n: 0.0025,
    expected_inflation: 0.005,
    term_premium: 0.0025,
    equity_risk_premium: 0.01,
    beta: 0.2,
    retention_rate: 0.1,
    earnings_growth_factor: 0.2,
  };

  function withParam(solow, fin, name, value) {
    const [which, key] = PARAM_MAP[name];
    if (which === "solow") return [Object.assign({}, solow, { [key]: value }), fin];
    return [solow, Object.assign({}, fin, { [key]: value })];
  }

  function getParam(solow, fin, name) {
    const [which, key] = PARAM_MAP[name];
    return which === "solow" ? solow[key] : fin[key];
  }

  function tornado(solow, fin, shocks, method) {
    shocks = shocks || DEFAULT_TORNADO_SHOCKS;
    const rows = [];
    Object.keys(shocks).forEach((name) => {
      const h = shocks[name];
      const base = getParam(solow, fin, name);
      const vals = [];
      const pes = [];
      [base - h, base + h].forEach((v) => {
        if (v < 0) v = 0; // mirrors Python's clip-to-valid-range
        const [s2, f2] = withParam(solow, fin, name, v);
        vals.push(v);
        pes.push(justifiedPE(s2, f2, method).justifiedPE);
      });
      const swing = pes[0] == null || pes[1] == null ? Infinity : Math.abs(pes[1] - pes[0]);
      rows.push({ parameter: name, lowValue: vals[0], highValue: vals[1], peLow: pes[0], peHigh: pes[1], swing });
    });
    rows.sort((a, b) => (b.swing === a.swing ? 0 : b.swing - a.swing));
    return rows;
  }

  // ---------- Inversion ----------

  function bisect(f, lo, hi, tol, maxIter) {
    tol = tol == null ? 1e-10 : tol;
    maxIter = maxIter == null ? 200 : maxIter;
    let flo = f(lo);
    let fhi = f(hi);
    if (Number.isNaN(flo) || Number.isNaN(fhi) || flo * fhi > 0) return null;
    if (flo === 0) return lo;
    if (fhi === 0) return hi;
    for (let i = 0; i < maxIter; i++) {
      const mid = 0.5 * (lo + hi);
      const fmid = f(mid);
      if (Number.isNaN(fmid)) return null;
      if (Math.abs(hi - lo) < tol) return mid;
      if (flo * fmid <= 0) {
        hi = mid;
        fhi = fmid;
      } else {
        lo = mid;
        flo = fmid;
      }
    }
    return 0.5 * (lo + hi);
  }

  function peOrInf(solow, fin, method) {
    const pe = justifiedPE(solow, fin, method).justifiedPE;
    return pe == null ? Infinity : pe;
  }

  function impliedG(targetPE, solow, fin, gMin, gMax, method) {
    gMin = gMin == null ? 0.0 : gMin;
    gMax = gMax == null ? 0.1 : gMax;
    if (!(targetPE > 0)) throw new Error("targetPE must be positive");
    const f = (g) => peOrInf(Object.assign({}, solow, { g }), fin, method) - targetPE;
    let hi = gMax;
    if (!Number.isFinite(f(hi))) {
      const edge = bisect((g) => (Number.isFinite(f(g)) ? 0.0 : 1.0) - 0.5, gMin, gMax, 1e-9);
      if (edge == null) return null;
      hi = Math.max(gMin, edge - 1e-9);
    }
    return bisect(f, gMin, hi);
  }

  function impliedERP(targetPE, solow, fin, erpMin, erpMax, method) {
    erpMin = erpMin == null ? 0.0 : erpMin;
    erpMax = erpMax == null ? 0.2 : erpMax;
    if (!(targetPE > 0)) throw new Error("targetPE must be positive");
    const f = (erp) => peOrInf(solow, Object.assign({}, fin, { erp }), method) - targetPE;
    let lo = erpMin;
    if (!Number.isFinite(f(lo))) {
      const edge = bisect((e) => (Number.isFinite(f(e)) ? 1.0 : 0.0) - 0.5, erpMin, erpMax, 1e-9);
      if (edge == null) return null;
      lo = Math.min(erpMax, edge + 1e-9);
    }
    return bisect(f, lo, erpMax);
  }

  // ---------- Monte Carlo ----------

  // mulberry32: small seeded PRNG so UI runs are reproducible.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function standardNormalDraws(n, seed) {
    const rand = mulberry32(seed == null ? 42 : seed);
    const z = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(4);
      for (let j = 0; j < 4; j++) {
        let u = 0;
        let v = 0;
        while (u === 0) u = rand();
        while (v === 0) v = rand();
        row[j] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      z[i] = row;
    }
    return z;
  }

  function cholesky(m, jitter) {
    const n = m.length;
    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = m[i][j] + (i === j && jitter ? jitter : 0);
        for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
        if (i === j) {
          if (sum <= 0) {
            if (!jitter) return cholesky(m, 1e-12);
            throw new Error("correlation matrix is not positive semi-definite");
          }
          L[i][j] = Math.sqrt(sum);
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
    }
    return L;
  }

  function percentile(sorted, q) {
    const idx = (q / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  function summarize(values) {
    const valid = [];
    for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) valid.push(values[i]);
    const n = values.length;
    const nv = valid.length;
    if (nv === 0) {
      return { n, nValid: 0, invalidShare: 1, mean: NaN, std: NaN, stdError: NaN, p5: NaN, p25: NaN, p50: NaN, p75: NaN, p95: NaN };
    }
    let mean = 0;
    for (let i = 0; i < nv; i++) mean += valid[i];
    mean /= nv;
    let ss = 0;
    for (let i = 0; i < nv; i++) ss += (valid[i] - mean) * (valid[i] - mean);
    const std = nv > 1 ? Math.sqrt(ss / (nv - 1)) : 0;
    valid.sort((a, b) => a - b);
    return {
      n, nValid: nv, invalidShare: 1 - nv / n, mean, std, stdError: std / Math.sqrt(nv),
      p5: percentile(valid, 5), p25: percentile(valid, 25), p50: percentile(valid, 50),
      p75: percentile(valid, 75), p95: percentile(valid, 95),
    };
  }

  const IDENTITY4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

  // draws: (g, inflation, term premium, erp). `z` optionally injects N(0,1) rows.
  function monteCarlo(solow, fin, mc, z, method) {
    const n = mc.numSims;
    z = z || standardNormalDraws(n, mc.seed);
    const chol = cholesky(mc.corr || IDENTITY4);
    const means = [
      mc.gMean == null ? solow.g : mc.gMean,
      mc.inflMean == null ? fin.inflation : mc.inflMean,
      mc.tpMean == null ? fin.termPremium : mc.tpMean,
      fin.erp,
    ];
    const stds = [mc.gStd, mc.inflStd, mc.tpStd, mc.erpStd == null ? 0 : mc.erpStd];
    const draws = new Array(n);
    const gArr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(4);
      for (let j = 0; j < 4; j++) {
        let x = 0;
        for (let k = 0; k <= j; k++) x += z[i][k] * chol[j][k];
        row[j] = means[j] + stds[j] * x;
      }
      if (row[0] < 0) row[0] = 0;
      if (row[3] < 0) row[3] = 0;
      draws[i] = row;
      gArr[i] = row[0];
    }
    let real;
    if (method === "simulated") {
      real = simulatedFinalGrowthVec(gArr, solow);
    } else {
      real = new Float64Array(n);
      for (let i = 0; i < n; i++) real[i] = closedFormGrowth(solow.n, gArr[i]);
    }
    const pe = new Float64Array(n);
    const eg = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = valuation(real[i] / 100, draws[i][1], draws[i][2], draws[i][3], fin);
      pe[i] = v.justifiedPE;
      eg[i] = v.earningsGrowthRate;
    }
    return { pe, eg, draws, summary: summarize(pe) };
  }

  const defaults = {
    s: 0.2, n: 0.005, g: 0.02, delta: 0.05, alpha: 0.35, T: 200,
    A0: 1, L0: 100, K0: 200,
    beta: 1.0, erp: 0.05, inflation: 0.02, termPremium: 0.01,
    retention: 0.35, egf: 1.1, highGrowthYears: 0,
    numSims: 5000, gStd: 0.005, inflStd: 0.005, tpStd: 0.002, erpStd: 0, rho: 0, seed: 42,
    targetPE: 18,
  };

  const MacroFin = {
    production, nextK, simulateSolow, simulatedFinalGrowthVec, closedFormGrowth,
    steadyStateGrowth, convergenceGap, steadyStateOutputCapitalRatio,
    justifiedPE, peSensitivity, tornado, DEFAULT_TORNADO_SHOCKS,
    impliedG, impliedERP, monteCarlo, summarize, standardNormalDraws, cholesky,
    defaults,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = MacroFin;
  if (root) root.MacroFin = MacroFin;

  // =========================================================================
  // UI (browser only)
  // =========================================================================

  if (typeof document === "undefined") return;

  const fmtPct = (v, d) => (v == null || !Number.isFinite(v) ? "—" : (v * 100).toFixed(d == null ? 2 : d) + "%");
  const fmtNum = (v, d) => (v == null || !Number.isFinite(v) ? "—" : Number(v).toFixed(d == null ? 2 : d));

  const formats = {
    s: { kind: "pct", d: 1 }, n: { kind: "pct", d: 2 }, g: { kind: "pct", d: 2 },
    delta: { kind: "pct", d: 1 }, alpha: { kind: "pct", d: 0 }, T: { kind: "int" },
    beta: { kind: "num", d: 2 }, erp: { kind: "pct", d: 2 },
    inflation: { kind: "pct", d: 2 }, termPremium: { kind: "pct", d: 2 },
    retention: { kind: "pct", d: 0 }, egf: { kind: "num", d: 2 },
    highGrowthYears: { kind: "years" },
    numSims: { kind: "int" }, gStd: { kind: "pct", d: 2 },
    inflStd: { kind: "pct", d: 2 }, tpStd: { kind: "pct", d: 2 }, erpStd: { kind: "pct", d: 2 },
    rho: { kind: "num", d: 2 }, seed: { kind: "int" }, targetPE: { kind: "num", d: 1 },
  };

  function fmt(id, value) {
    const f = formats[id];
    if (!f) return value;
    if (f.kind === "pct") return (value * 100).toFixed(f.d) + "%";
    if (f.kind === "int") return Math.round(value).toString();
    if (f.kind === "years") return value === 0 ? "off (single-stage)" : Math.round(value) + " yrs";
    return Number(value).toFixed(f.d);
  }

  function readControls() {
    const v = {};
    Object.keys(defaults).forEach((k) => {
      const el = document.getElementById(k);
      const parsed = el ? parseFloat(el.value) : NaN;
      v[k] = Number.isFinite(parsed) ? parsed : defaults[k];
    });
    v.A0 = defaults.A0;
    v.L0 = defaults.L0;
    v.K0 = defaults.K0;
    v.T = Math.round(v.T);
    return v;
  }

  function finFrom(state) {
    return {
      beta: state.beta, erp: state.erp, inflation: state.inflation,
      termPremium: state.termPremium, retention: state.retention, egf: state.egf,
      highGrowthYears: state.highGrowthYears > 0 ? Math.round(state.highGrowthYears) : null,
    };
  }

  function mcFrom(state) {
    const rho = Math.max(-0.99, Math.min(0.99, state.rho));
    const corr = [[1, 0, 0, 0], [0, 1, rho, 0], [0, rho, 1, 0], [0, 0, 0, 1]];
    return {
      numSims: Math.round(state.numSims), gMean: null, gStd: state.gStd,
      inflMean: null, inflStd: state.inflStd, tpMean: null, tpStd: state.tpStd,
      erpStd: state.erpStd, corr, seed: Math.round(state.seed),
    };
  }

  function syncOutputs(state) {
    document.querySelectorAll(".param").forEach((p) => {
      const id = p.dataset.id;
      const out = p.querySelector("output");
      if (out) out.textContent = fmt(id, state[id]);
    });
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function setKpi(id, text, invalid) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    const card = el.parentElement;
    if (card) card.classList.toggle("invalid", !!invalid);
  }

  // ---------- Plotly ----------

  function isLight() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  }

  function plotlyLayout(title, xtitle, ytitle) {
    const light = isLight();
    const fontColor = light ? "#0f172a" : "#e6ecff";
    const grid = light ? "#dbe3f5" : "#243260";
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

  function vline(x, color) {
    return { type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color, dash: "dash", width: 1 } };
  }

  // ---------- Renderers ----------

  function renderSolow(state) {
    const path = simulateSolow(state);
    const xs = Array.from({ length: state.T - 1 }, (_, i) => i + 1);
    const trace = {
      x: xs, y: Array.from(path.growth), type: "scatter", mode: "lines",
      name: "Simulated output growth", line: { color: "#5eead4", width: 2.5 },
    };
    const target = closedFormGrowth(state.n, state.g);
    const ref = {
      x: [xs[0], xs[xs.length - 1]], y: [target, target], type: "scatter", mode: "lines",
      name: "Steady state (1+n)(1+g)−1", line: { color: "#818cf8", width: 1.5, dash: "dash" },
    };
    const gap = convergenceGap(state);
    Plotly.react(
      "chart-solow", [trace, ref],
      plotlyLayout(`Solow-Swan convergence — gap at T: ${gap >= 0 ? "+" : ""}${gap.toFixed(4)} pp`, "Period", "YoY growth (%)"),
      plotlyConfig
    );
  }

  function renderPe(state) {
    const sweep = peSensitivity(state, finFrom(state), 0.005, 0.04, 36);
    const trace = {
      x: Array.from(sweep.xs).map((g) => g * 100),
      y: Array.from(sweep.ys).map((v) => (Number.isFinite(v) ? v : null)),
      type: "scatter", mode: "lines+markers", name: "Justified P/E",
      line: { color: "#5eead4", width: 2 }, marker: { color: "#5eead4", size: 6 }, connectgaps: false,
    };
    const layout = plotlyLayout("Justified P/E vs. tech progress", "Tech progress g (%)", "Justified forward P/E");
    layout.shapes = [vline(state.g * 100, "#818cf8")];
    Plotly.react("chart-pe", [trace], layout, plotlyConfig);
  }

  const TORNADO_LABELS = {
    g: "Tech progress g", n: "Population growth n", expected_inflation: "Inflation π",
    term_premium: "Term premium", equity_risk_premium: "Equity risk premium", beta: "Beta β",
    retention_rate: "Retention b", earnings_growth_factor: "Earnings growth factor",
  };

  function renderTornado(state) {
    const fin = finFrom(state);
    const base = justifiedPE(state, fin).justifiedPE;
    const rows = tornado(state, fin).slice().reverse();
    const labels = rows.map((r) => TORNADO_LABELS[r.parameter] || r.parameter);
    const light = isLight();
    const layout = plotlyLayout("Tornado — one-at-a-time ± shocks", "Justified forward P/E", "");
    layout.barmode = "overlay";
    layout.margin.l = 170;
    if (base == null) {
      Plotly.react("chart-tornado", [], Object.assign(layout, { title: { text: "Tornado — base case invalid (g_e ≥ k_e)" } }), plotlyConfig);
      return;
    }
    const mk = (key, name, color) => ({
      type: "bar", orientation: "h", name, y: labels,
      x: rows.map((r) => (r[key] == null ? null : r[key] - base)),
      base: rows.map(() => base),
      marker: { color },
      hovertemplate: rows.map((r) => `${name}: P/E %{customdata:.2f}<extra></extra>`),
      customdata: rows.map((r) => r[key]),
    });
    layout.shapes = [vline(base, light ? "#0f172a" : "#e6ecff")];
    Plotly.react(
      "chart-tornado",
      [mk("peLow", "− shock", "#f87171"), mk("peHigh", "+ shock", "#5eead4")],
      layout, plotlyConfig
    );
  }

  function renderMonteCarlo(state) {
    const out = monteCarlo(state, finFrom(state), mcFrom(state));
    const s = out.summary;
    const pe = Array.from(out.pe).filter(Number.isFinite);
    const eg = Array.from(out.eg).map((v) => v * 100);
    const light = isLight();
    const lineColor = light ? "#0f172a" : "#e6ecff";

    const peLayout = plotlyLayout(
      `P/E — μ=${fmtNum(s.mean)}, σ=${fmtNum(s.std)}, P5–P95 ${fmtNum(s.p5)}–${fmtNum(s.p95)}, invalid ${fmtPct(s.invalidShare, 1)}`,
      "P/E ratio", "Frequency"
    );
    if (Number.isFinite(s.mean)) {
      peLayout.shapes = [vline(s.mean, lineColor), vline(s.p5, "#f87171"), vline(s.p95, "#f87171")];
    }
    Plotly.react(
      "chart-mc-pe",
      [{ x: pe, type: "histogram", nbinsx: 35, name: "P/E", marker: { color: "#818cf8", line: { color: "rgba(255,255,255,0.6)", width: 1 } } }],
      peLayout, plotlyConfig
    );

    const egStats = summarize(eg);
    const egLayout = plotlyLayout(
      `Earnings growth — μ=${fmtNum(egStats.mean)}%, σ=${fmtNum(egStats.std)}%`,
      "Earnings growth (%)", "Frequency"
    );
    if (Number.isFinite(egStats.mean)) egLayout.shapes = [vline(egStats.mean, lineColor)];
    Plotly.react(
      "chart-mc-eg",
      [{ x: eg, type: "histogram", nbinsx: 35, name: "Earnings growth", marker: { color: "#5eead4", line: { color: "rgba(255,255,255,0.6)", width: 1 } } }],
      egLayout, plotlyConfig
    );
    return s;
  }

  // ---------- KPIs & implied panel ----------

  function updateKpis(state) {
    const fin = finFrom(state);
    const res = justifiedPE(state, fin);
    setKpi("kpi-real", fmtPct(res.realGrowthRate));
    setKpi("kpi-nominal", fmtPct(res.nominalGrowthRate));
    setKpi("kpi-rf", fmtPct(res.riskFreeRate));
    setKpi("kpi-rr", fmtPct(res.requiredReturn));
    setKpi("kpi-duration", res.peDuration == null ? "—" : fmtNum(res.peDuration, 1) + " yrs");
    const gap = convergenceGap(state);
    setKpi("kpi-gap", (gap >= 0 ? "+" : "") + gap.toFixed(3) + " pp", Math.abs(gap) > 0.05);
    if (res.justifiedPE === null) {
      setKpi("kpi-pe", "Invalid (g_term ≥ k_e)", true);
    } else {
      setKpi("kpi-pe", fmtNum(res.justifiedPE), false);
    }
    const note = document.getElementById("pe-note");
    if (note) {
      note.hidden = !res.earningsOutgrowGdp;
    }
    return res;
  }

  function updateImplied(state) {
    const fin = finFrom(state);
    const target = state.targetPE;
    const g = impliedG(target, state, fin, 0, 0.1);
    const erp = impliedERP(target, state, fin, 0, 0.2);
    const peAt = (s2, f2) => justifiedPE(s2, f2).justifiedPE;
    const gLo = peAt(Object.assign({}, state, { g: 0 }), fin);
    const gHi = peAt(Object.assign({}, state, { g: 0.1 }), fin);
    const eLo = peAt(state, Object.assign({}, fin, { erp: 0.2 }));
    const eHi = peAt(state, Object.assign({}, fin, { erp: 0 }));
    setText("implied-g", g == null ? "not reachable" : fmtPct(g));
    setText("implied-g-range", `reachable P/E for g ∈ [0, 10%]: ${fmtNum(gLo)} – ${gHi == null ? "∞" : fmtNum(gHi)}`);
    setText("implied-erp", erp == null ? "not reachable" : fmtPct(erp));
    setText("implied-erp-range", `reachable P/E for ERP ∈ [0, 20%]: ${fmtNum(eLo)} – ${eHi == null ? "∞" : fmtNum(eHi)}`);
    return { g, erp };
  }

  // ---------- Tabs ----------

  const renderers = { solow: renderSolow, pe: renderPe, tornado: renderTornado, mc: renderMonteCarlo };
  const tabContainer = { solow: "chart-solow", pe: "chart-pe", tornado: "chart-tornado", mc: "chart-mc" };
  let activeTab = "solow";

  function showTab(tab) {
    activeTab = tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tab));
    Object.entries(tabContainer).forEach(([key, id]) => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle("hidden", key !== tab);
    });
    renderers[tab](readControls());
    requestAnimationFrame(() => {
      if (tab === "mc") {
        Plotly.Plots.resize(document.getElementById("chart-mc-pe"));
        Plotly.Plots.resize(document.getElementById("chart-mc-eg"));
      } else {
        Plotly.Plots.resize(document.getElementById(tabContainer[tab]));
      }
    });
  }

  // ---------- Export ----------

  function buildExport(state) {
    const fin = finFrom(state);
    const mc = mcFrom(state);
    const out = monteCarlo(state, fin, mc);
    return {
      exportedAt: new Date().toISOString(),
      parameters: { solow: { s: state.s, n: state.n, g: state.g, delta: state.delta, alpha: state.alpha, T: state.T, A0: state.A0, L0: state.L0, K0: state.K0 }, financial: fin, monteCarlo: mc },
      valuation: justifiedPE(state, fin),
      convergenceGapPp: convergenceGap(state),
      steadyStateOutputCapitalRatio: steadyStateOutputCapitalRatio(state),
      tornado: tornado(state, fin),
      implied: { targetPE: state.targetPE, technologyGrowth: impliedG(state.targetPE, state, fin, 0, 0.1), equityRiskPremium: impliedERP(state.targetPE, state, fin, 0, 0.2) },
      monteCarloSummary: out.summary,
    };
  }

  function exportJson() {
    const text = JSON.stringify(buildExport(readControls()), null, 2);
    try {
      const blob = new Blob([text], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "macrofin-scenario.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (e) {
      if (navigator.clipboard) navigator.clipboard.writeText(text);
    }
  }

  // ---------- Wiring ----------

  function debounce(fn, ms) {
    let t = null;
    return function () {
      const args = arguments;
      clearTimeout(t);
      t = setTimeout(() => fn.apply(null, args), ms);
    };
  }

  function refreshActive() {
    const state = readControls();
    syncOutputs(state);
    updateKpis(state);
    updateImplied(state);
    renderers[activeTab](state);
  }
  const debouncedRefresh = debounce(refreshActive, 30);

  function setDefaults() {
    Object.keys(defaults).forEach((k) => {
      const el = document.getElementById(k);
      if (el) el.value = defaults[k];
    });
    refreshActive();
  }

  function init() {
    Object.keys(defaults).forEach((k) => {
      const el = document.getElementById(k);
      if (el) el.addEventListener("input", debouncedRefresh);
    });
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
    const reset = document.getElementById("reset-btn");
    if (reset) reset.addEventListener("click", setDefaults);
    const exp = document.getElementById("export-btn");
    if (exp) exp.addEventListener("click", exportJson);
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", refreshActive);
    }
    refreshActive();
    showTab("solow");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : null);
