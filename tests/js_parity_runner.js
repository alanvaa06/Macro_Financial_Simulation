// Node.js harness for tests/test_js_parity.py: reads a JSON spec on stdin,
// evaluates the JS model, prints JSON results on stdout.
"use strict";
const path = require("path");
const M = require(path.join(__dirname, "..", "web", "app.js"));

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (c) => (input += c));
process.stdin.on("end", () => {
  const spec = JSON.parse(input);
  const { solow, fin, mc, z, targetPE } = spec;
  const nanToNull = (arr) => Array.from(arr, (v) => (Number.isFinite(v) ? v : null));
  const val = M.justifiedPE(solow, fin);
  const sens = M.peSensitivity(solow, fin, 0.005, 0.04, 36);
  const res = M.monteCarlo(solow, fin, mc, z);
  const out = {
    valuation: val,
    solowGrowth: Array.from(M.simulateSolow(solow).growth),
    peSensitivity: nanToNull(sens.ys),
    tornado: M.tornado(solow, fin),
    impliedG: M.impliedG(targetPE, solow, fin),
    impliedERP: M.impliedERP(targetPE, solow, fin),
    mc: {
      pe: nanToNull(res.pe),
      eg: Array.from(res.eg),
      draws: res.draws,
      summary: res.summary,
    },
  };
  process.stdout.write(JSON.stringify(out));
});
