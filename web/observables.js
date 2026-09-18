// Latest observable counterparts for the model inputs and headline outputs.
// Reference economy: United States / S&P 500. Values are in the same units the
// sliders use (rates as decimals). Every entry carries its source and as-of
// date so a reader can judge how stale it is. Parameters that are pure
// modelling choices (simulation horizon, Monte Carlo settings) are listed in
// `notApplicable` with the reason, so the UI can say so explicitly.
//
// Update this file when you refresh the numbers; app.js only reads it.

(function (root) {
  "use strict";

  const observables = {
    asOf: "2026-09-18",
    region: "United States / S&P 500",
    params: {
      n: {
        value: 0.005,
        label: "0.5%",
        source: "Census Bureau, Vintage 2025 estimates",
        asOf: "Jul 2025",
        note: "US resident population grew 0.5% between July 2024 and July 2025 (+1.8M), the slowest pace since 2021 as net migration fell.",
        url: "https://www.census.gov/newsroom/press-releases/2026/population-growth-slows.html",
      },
      g: {
        value: 0.012,
        label: "≈1.2%",
        source: "BLS Total Factor Productivity, 2025 (released Mar 2026)",
        asOf: "2025",
        note: "Private non-farm TFP rose 0.8% in 2025 (1.5% in 2024). TFP is Hicks-neutral; the model's g is labour-augmenting, so g ≈ TFP ÷ (1−α) ≈ 0.8% ÷ 0.65 ≈ 1.2%.",
        url: "https://www.bls.gov/news.release/prod3.nr0.htm",
        derived: true,
      },
      s: {
        value: 0.17,
        label: "17.0%",
        source: "BEA, gross saving as % of gross national income",
        asOf: "Q3 2025",
        note: "US gross saving was 17.0% of gross national income in Q3 2025 (16.7% for full-year 2024).",
        url: "https://fred.stlouisfed.org/series/W206RC1Q156SBEA",
      },
      delta: {
        value: 0.06,
        label: "≈6%",
        source: "BEA Fixed Assets accounts (consumption of fixed capital ÷ net stock)",
        asOf: "2025",
        note: "Private consumption of fixed capital was ≈$4.2T in 2025 against a private net fixed-asset stock of roughly $70T, an implied aggregate depreciation rate of ≈6% per year.",
        url: "https://fred.stlouisfed.org/series/A024RC1A027NBEA",
        derived: true,
      },
      alpha: {
        value: 0.4,
        label: "0.40",
        source: "BLS productivity accounts, capital's share of income",
        asOf: "Q1 2026",
        note: "Capital's share of income in the BLS TFP accounts was 40.4% in Q1 2026. The narrower non-farm business labour share hit a record-low 52.8% in Q2 2026, implying an even higher capital share.",
        url: "https://www.bls.gov/opub/ted/2026/labor-share-at-its-lowest-level-52-8-percent-in-second-quarter-2026.htm",
      },
      beta: {
        value: 1.0,
        label: "1.00",
        source: "Definition",
        asOf: "—",
        note: "The market index has a beta of 1.00 by construction. Use a sector or stock beta when valuing something narrower than the index.",
      },
      erp: {
        value: 0.0423,
        label: "4.23%",
        source: "Damodaran, implied ERP for the S&P 500",
        asOf: "1 Jan 2026",
        note: "Implied equity risk premium over the 10-year Treasury at the start of 2026. Damodaran refreshes the estimate monthly on damodaran.com.",
        url: "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html",
      },
      inflation: {
        value: 0.0236,
        label: "2.36%",
        source: "10-year TIPS breakeven inflation rate",
        asOf: "mid-Sep 2026",
        note: "Market-implied 10-year inflation was 2.36% in mid-September 2026. Cross-checks: headline CPI +3.4% y/y and core +2.4% (Aug 2026); Michigan 5-year household expectations 3.4% (Sep 2026 prelim).",
        url: "https://fred.stlouisfed.org/series/T10YIE",
      },
      termPremium: {
        value: 0.008,
        label: "0.80%",
        source: "NY Fed ACM 10-year term premium",
        asOf: "Aug 2026",
        note: "The ACM 10-year term premium was 0.80% on 13 Aug 2026 and touched 0.89% on 17 Aug, the highest 2026 print, after five years near zero.",
        url: "https://www.newyorkfed.org/research/data_indicators/term-premia-tabs",
      },
      retention: {
        value: 0.3,
        label: "≈30%",
        source: "S&P Dow Jones Indices, dividends + buybacks vs. earnings",
        asOf: "mid-2026",
        note: "Dividends alone are 35–42% of S&P 500 earnings (retention ≈60%). Counting buybacks, total payout is ≈70% of earnings, so the retention consistent with a payout-based P/E is ≈30%.",
        url: "https://www.spglobal.com/spdji/en/indices/dividends-factors/sp-500-buyback-index/",
        derived: true,
      },
      egf: {
        value: 1.3,
        label: "≈1.3×",
        source: "Long-run S&P 500 EPS growth vs. nominal GDP",
        asOf: "1990–2025",
        note: "S&P 500 EPS compounded ≈6–7% a year over 1990–2025 against ≈5% nominal GDP, a ratio of ≈1.3×. Near-term consensus is far higher (FactSet CY2027 EPS +15.1% as of Sep 2026) but cannot persist forever; use high-growth years for that.",
        url: "https://www.factset.com/earningsinsight",
        derived: true,
      },
      targetPE: {
        value: 19.1,
        label: "19.1×",
        source: "FactSet Earnings Insight, S&P 500 forward 12-month P/E",
        asOf: "Sep 2026",
        note: "Forward 12-month P/E of 19.1, below the 5-year average (19.8) and above the 10-year average (19.0).",
        url: "https://www.factset.com/earningsinsight",
      },
    },
    // Headline outputs with a market counterpart, for the KPI cards.
    kpis: {
      "kpi-real": {
        label: "2.1%",
        short: "CBO potential",
        source: "CBO potential real GDP growth, 2026–2030 average",
        asOf: "Feb 2026",
        url: "https://www.cbo.gov/publication/61882",
      },
      "kpi-nominal": {
        label: "≈4.5%",
        short: "CBO + breakeven",
        source: "CBO potential growth + 10-year breakeven inflation",
        asOf: "Sep 2026",
      },
      "kpi-rf": {
        label: "4.80%",
        short: "10-yr Treasury",
        source: "10-year Treasury yield, 17 Sep 2026",
        asOf: "17 Sep 2026",
        url: "https://fred.stlouisfed.org/series/DGS10",
      },
      "kpi-pe": {
        label: "19.1×",
        short: "S&P 500 fwd",
        source: "S&P 500 forward P/E (FactSet)",
        asOf: "Sep 2026",
        url: "https://www.factset.com/earningsinsight",
      },
    },
    notApplicable: {
      T: "Simulation horizon — a modelling choice, no observable counterpart.",
      highGrowthYears: "Length of the high-growth stage is a modelling choice. Analysts' explicit-forecast horizon is typically 3–5 years.",
      numSims: "Monte Carlo sample size — a modelling choice.",
      gStd: "Uncertainty band — calibrate to your own view; no single observable.",
      inflStd: "Uncertainty band — calibrate to your own view; no single observable.",
      tpStd: "Uncertainty band — calibrate to your own view; no single observable.",
      erpStd: "Uncertainty band — calibrate to your own view; no single observable.",
      rho: "Correlation assumption — no single observable.",
      seed: "Random seed — reproducibility only.",
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = observables;
  if (root) root.MacroFinObservables = observables;
})(typeof window !== "undefined" ? window : null);
