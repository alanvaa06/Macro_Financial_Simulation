"""Pin the JS port (web/app.js) to the Python model on identical inputs.

Runs ``tests/js_parity_runner.js`` under Node.js with a JSON spec on stdin and
compares every number it returns. Skipped when Node.js is not on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from economics_models import (
    FinancialParameters,
    MonteCarloParameters,
    SolowParameters,
    implied_equity_risk_premium,
    implied_technology_growth,
    justified_pe,
    monte_carlo,
    pe_sensitivity,
    simulate_solow,
    tornado,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "js_parity_runner.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

CASES = [
    {"solow": {}, "fin": {}},
    {
        "solow": {"g": 0.03, "n": 0.01, "s": 0.25, "alpha": 0.4, "T": 120},
        "fin": {"beta": 1.2, "retention_rate": 0.5},
    },
    {"solow": {}, "fin": {"earnings_growth_factor": 1.4, "high_growth_years": 10}},
    {"solow": {"g": 0.04}, "fin": {"earnings_growth_factor": 2.0, "equity_risk_premium": 0.02}},
]


def _js_fin(f: FinancialParameters) -> dict:
    return {
        "beta": f.beta,
        "erp": f.equity_risk_premium,
        "inflation": f.expected_inflation,
        "termPremium": f.term_premium,
        "retention": f.retention_rate,
        "egf": f.earnings_growth_factor,
        "highGrowthYears": f.high_growth_years,
    }


def _js_solow(p: SolowParameters) -> dict:
    return {
        "s": p.s,
        "n": p.n,
        "g": p.g,
        "delta": p.delta,
        "alpha": p.alpha,
        "A0": p.A0,
        "L0": p.L0,
        "K0": p.K0,
        "T": p.T,
    }


def _run_js(spec: dict) -> dict:
    out = subprocess.run(
        ["node", str(RUNNER)], input=json.dumps(spec), capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize("case", CASES)
def test_js_matches_python(case):
    solow = SolowParameters(**case["solow"])
    fin = FinancialParameters(**case["fin"])
    rng = np.random.default_rng(123)
    z = rng.standard_normal((400, 4))
    corr = np.eye(4)
    corr[1, 2] = corr[2, 1] = 0.4
    mc = MonteCarloParameters(num_simulations=400, seed=0, correlation=corr, erp_std=0.01)
    target_pe = 12.0

    js = _run_js(
        {
            "solow": _js_solow(solow),
            "fin": _js_fin(fin),
            "mc": {
                "numSims": 400,
                "gMean": None,
                "gStd": mc.g_std,
                "inflMean": None,
                "inflStd": mc.inflation_std,
                "tpMean": None,
                "tpStd": mc.term_premium_std,
                "erpStd": mc.erp_std,
                "corr": corr.tolist(),
                "seed": 0,
            },
            "z": z.tolist(),
            "targetPE": target_pe,
        }
    )

    py = justified_pe(solow, fin)
    for py_key, js_key in [
        ("real_growth_rate", "realGrowthRate"),
        ("nominal_growth_rate", "nominalGrowthRate"),
        ("risk_free_rate", "riskFreeRate"),
        ("required_return", "requiredReturn"),
        ("earnings_growth_rate", "earningsGrowthRate"),
        ("terminal_growth_rate", "terminalGrowthRate"),
        ("justified_pe", "justifiedPE"),
        ("pe_duration", "peDuration"),
    ]:
        a, b = getattr(py, py_key), js["valuation"][js_key]
        if a is None:
            assert b is None
        else:
            assert b == pytest.approx(a, rel=1e-9, abs=1e-9), py_key

    path = simulate_solow(solow)
    np.testing.assert_allclose(js["solowGrowth"], path.growth_rates, rtol=1e-9)

    _, pe = pe_sensitivity(solow, fin, 0.005, 0.04, 36)
    js_pe = np.array([np.nan if v is None else v for v in js["peSensitivity"]])
    np.testing.assert_allclose(js_pe, pe, rtol=1e-9, equal_nan=True)

    py_rows = tornado(solow, fin)
    assert [r["parameter"] for r in js["tornado"]] == [r.parameter for r in py_rows]
    for jr, pr in zip(js["tornado"], py_rows):
        for a, b in ((jr["peLow"], pr.pe_low), (jr["peHigh"], pr.pe_high)):
            assert (a is None) == (b is None)
            if a is not None:
                assert a == pytest.approx(b, rel=1e-9)

    for py_fn, key in (
        (implied_technology_growth, "impliedG"),
        (implied_equity_risk_premium, "impliedERP"),
    ):
        a, b = py_fn(target_pe, solow, fin), js[key]
        assert (a is None) == (b is None)
        if a is not None:
            assert b == pytest.approx(a, abs=1e-7)

    res = monte_carlo(solow, fin, mc, z=z)
    js_mc_pe = np.array([np.nan if v is None else v for v in js["mc"]["pe"]])
    np.testing.assert_allclose(js_mc_pe, res.pe, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(js["mc"]["eg"], res.earnings_growth, rtol=1e-9)
    np.testing.assert_allclose(js["mc"]["draws"], res.draws, rtol=1e-9)
    assert js["mc"]["summary"]["invalidShare"] == pytest.approx(res.summary.invalid_share)
    if res.summary.n_valid:
        assert js["mc"]["summary"]["mean"] == pytest.approx(res.summary.mean, rel=1e-9)
        assert js["mc"]["summary"]["p50"] == pytest.approx(res.summary.percentiles[50], rel=1e-9)
