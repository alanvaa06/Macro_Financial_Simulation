"""Verify the model math and the integration with the financial layer."""

from __future__ import annotations

import numpy as np
import pytest

from economics_models import (
    FinancialParameters,
    MonteCarloParameters,
    SolowParameters,
    closed_form_growth_rate,
    compare_scenarios,
    convergence_gap,
    gdp_sensitivity,
    implied_equity_risk_premium,
    implied_technology_growth,
    justified_pe,
    monte_carlo,
    monte_carlo_pe,
    monte_carlo_summary,
    pe_sensitivity,
    production,
    scenarios_to_csv,
    simulate_solow,
    steady_state_growth_rate,
    steady_state_output_capital_ratio,
    tornado,
)

# --------------------------------------------------------------------------- #
# Solow
# --------------------------------------------------------------------------- #


def test_production_constant_returns_to_K_and_L():
    y1 = production(100, 1, 100, 0.35)
    y2 = production(200, 1, 200, 0.35)
    assert y2 == pytest.approx(2 * y1, rel=1e-12)


def test_closed_form_is_default_and_exact():
    p = SolowParameters()
    expected = ((1 + p.n) * (1 + p.g) - 1) * 100
    assert steady_state_growth_rate(p) == expected
    assert steady_state_growth_rate(p, "closed_form") == expected


@pytest.mark.parametrize("g", [0.005, 0.02, 0.04])
def test_simulated_path_converges_to_closed_form(g):
    p = SolowParameters(g=g, T=600)
    assert steady_state_growth_rate(p, "simulated") == pytest.approx(
        steady_state_growth_rate(p, "closed_form"), rel=1e-4
    )
    assert abs(convergence_gap(p)) < 1e-3


def test_closed_form_is_invariant_to_s_delta_alpha_K0_T():
    base = steady_state_growth_rate(SolowParameters())
    for kw in ({"s": 0.05}, {"s": 0.6}, {"delta": 0.2}, {"alpha": 0.7}, {"K0": 1.0}, {"T": 50}):
        assert steady_state_growth_rate(SolowParameters(**kw)) == base


def test_steady_state_output_capital_ratio_matches_textbook():
    p = SolowParameters(T=2000)
    path = simulate_solow(p)
    assert path.Y[-1] / path.K[-1] == pytest.approx(steady_state_output_capital_ratio(p), rel=1e-6)
    assert steady_state_output_capital_ratio(p) == pytest.approx(
        ((1 + p.n) * (1 + p.g) - (1 - p.delta)) / p.s
    )


def test_simulate_solow_shape_and_growth_length():
    p = SolowParameters(T=50)
    path = simulate_solow(p)
    assert path.K.shape == (50,)
    assert path.growth_rates.shape == (49,)
    assert np.all(np.isfinite(path.Y))


def test_gdp_sensitivity_is_linear_in_g_closed_form():
    g_grid, gdp = gdp_sensitivity(SolowParameters(), g_min=0.01, g_max=0.03, steps=11)
    np.testing.assert_allclose(gdp, closed_form_growth_rate(0.005, g_grid))


def test_gdp_sensitivity_simulated_vectorised_matches_loop():
    p = SolowParameters(T=120)
    g_grid, vec = gdp_sensitivity(p, g_min=0.0, g_max=0.05, steps=6, method="simulated")
    loop = [
        steady_state_growth_rate(SolowParameters(T=120, g=float(g)), "simulated") for g in g_grid
    ]
    np.testing.assert_allclose(vec, loop, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #


def test_justified_pe_basic():
    res = justified_pe(SolowParameters(), FinancialParameters())
    assert res.required_return > res.earnings_growth_rate
    assert res.justified_pe is not None and res.justified_pe > 0
    # Single-stage Gordon closed forms
    assert res.justified_pe == pytest.approx(
        0.65 / (res.required_return - res.earnings_growth_rate)
    )
    assert res.pe_duration == pytest.approx(
        1 / (res.required_return - res.earnings_growth_rate), rel=1e-6
    )
    assert res.earnings_outgrow_gdp is True  # factor 1.1 assumed forever


def test_justified_pe_invalid_when_growth_exceeds_required_return():
    p = SolowParameters(g=0.10)
    f = FinancialParameters(beta=0.1, equity_risk_premium=0.001, earnings_growth_factor=2.5)
    res = justified_pe(p, f)
    assert res.justified_pe is None
    assert res.pe_duration is None


def test_two_stage_collapses_to_gordon_when_factor_is_one():
    p = SolowParameters()
    two = justified_pe(p, FinancialParameters(earnings_growth_factor=1.0, high_growth_years=7))
    one = justified_pe(p, FinancialParameters(earnings_growth_factor=1.0))
    assert two.justified_pe == pytest.approx(one.justified_pe, rel=1e-12)
    assert two.earnings_outgrow_gdp is False


def test_two_stage_is_between_single_stage_bounds():
    p = SolowParameters()
    hi = justified_pe(p, FinancialParameters(earnings_growth_factor=1.3)).justified_pe
    lo = justified_pe(p, FinancialParameters(earnings_growth_factor=1.0)).justified_pe
    mid = justified_pe(p, FinancialParameters(earnings_growth_factor=1.3, high_growth_years=10))
    assert lo < mid.justified_pe < hi
    assert mid.terminal_growth_rate == pytest.approx(mid.nominal_growth_rate)


def test_two_stage_valid_when_stage1_growth_exceeds_required_return():
    # Stage-1 growth above k is fine as long as the terminal rate is below k.
    p = SolowParameters(g=0.03)
    f = FinancialParameters(earnings_growth_factor=2.5, high_growth_years=5)
    res = justified_pe(p, f)
    assert res.earnings_growth_rate > res.required_return
    assert res.justified_pe is not None


def test_pe_increases_with_g_when_in_valid_region():
    g_grid, pe = pe_sensitivity(SolowParameters(), FinancialParameters(), 0.01, 0.025, 8)
    finite = pe[~np.isnan(pe)]
    assert np.all(np.diff(finite) >= -1e-9)


def test_simulated_method_is_close_to_closed_form_at_default_T():
    p, f = SolowParameters(), FinancialParameters()
    a = justified_pe(p, f).justified_pe
    b = justified_pe(p, f, method="simulated").justified_pe
    assert a == pytest.approx(b, rel=1e-3)


# --------------------------------------------------------------------------- #
# Sensitivity, inversion, scenarios
# --------------------------------------------------------------------------- #


def test_tornado_sorted_and_brackets_base():
    p, f = SolowParameters(), FinancialParameters()
    base = justified_pe(p, f).justified_pe
    rows = tornado(p, f)
    swings = [r.swing for r in rows]
    assert swings == sorted(swings, reverse=True)
    assert {r.parameter for r in rows} == {
        "g",
        "n",
        "expected_inflation",
        "term_premium",
        "equity_risk_premium",
        "beta",
        "retention_rate",
        "earnings_growth_factor",
    }
    for r in rows:
        assert min(r.pe_low, r.pe_high) <= base + 1e-9
        assert max(r.pe_low, r.pe_high) >= base - 1e-9


def test_tornado_clips_shocks_into_valid_range():
    rows = tornado(SolowParameters(n=0.0), FinancialParameters(), {"n": 0.01})
    assert rows[0].low_value == 0.0


def test_implied_growth_round_trips():
    p, f = SolowParameters(g=0.023), FinancialParameters()
    target = justified_pe(p, f).justified_pe
    g = implied_technology_growth(target, SolowParameters(), f)
    assert g == pytest.approx(0.023, abs=1e-7)


def test_implied_erp_round_trips():
    p, f = SolowParameters(), FinancialParameters(equity_risk_premium=0.062)
    target = justified_pe(p, f).justified_pe
    erp = implied_equity_risk_premium(target, p, FinancialParameters())
    assert erp == pytest.approx(0.062, abs=1e-7)


def test_implied_returns_none_when_unreachable():
    p, f = SolowParameters(), FinancialParameters()
    assert implied_technology_growth(1000.0, p, f) is None
    assert implied_equity_risk_premium(1000.0, p, f, erp_min=0.02) is None
    with pytest.raises(ValueError):
        implied_technology_growth(-1.0, p, f)


def test_implied_erp_handles_invalid_lower_bracket():
    # At erp=0 the Gordon denominator closes; the bracket must be shrunk to the valid region.
    p = SolowParameters(g=0.04)
    f = FinancialParameters(earnings_growth_factor=1.5)
    assert (
        justified_pe(
            p, FinancialParameters(earnings_growth_factor=1.5, equity_risk_premium=0.0)
        ).justified_pe
        is None
    )
    erp = implied_equity_risk_premium(15.0, p, f)
    assert erp is not None
    assert justified_pe(
        p, FinancialParameters(earnings_growth_factor=1.5, equity_risk_premium=erp)
    ).justified_pe == pytest.approx(15.0, rel=1e-6)


def test_compare_scenarios_and_csv():
    p = SolowParameters()
    res = compare_scenarios(
        {
            "bear": (p, FinancialParameters(equity_risk_premium=0.07)),
            "base": (p, FinancialParameters()),
            "bull": (p, FinancialParameters(equity_risk_premium=0.04)),
        }
    )
    assert res["bear"].justified_pe < res["base"].justified_pe < res["bull"].justified_pe
    csv_text = scenarios_to_csv(res)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("scenario,real_growth_rate")
    assert len(lines) == 4


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #


def test_monte_carlo_returns_expected_shape_and_finite_mean():
    pe, eg = monte_carlo_pe(
        SolowParameters(), FinancialParameters(), MonteCarloParameters(num_simulations=300, seed=42)
    )
    assert pe.shape == (300,) and eg.shape == (300,)
    assert np.isfinite(np.nanmean(pe))


def test_monte_carlo_is_seeded_and_centred_on_point_estimates():
    p, f = SolowParameters(g=0.03), FinancialParameters(expected_inflation=0.03)
    mc = MonteCarloParameters(num_simulations=20000, seed=7)
    a = monte_carlo(p, f, mc)
    b = monte_carlo(p, f, mc)
    np.testing.assert_array_equal(a.pe, b.pe)
    assert a.draws[:, 0].mean() == pytest.approx(0.03, abs=2e-4)  # g centred on solow.g
    assert a.draws[:, 1].mean() == pytest.approx(0.03, abs=2e-4)  # inflation centred on fin
    assert a.draws[:, 3].std() == pytest.approx(0.0, abs=1e-12)  # erp_std defaults to 0


def test_monte_carlo_vectorised_matches_scalar_pipeline():
    p, f = SolowParameters(), FinancialParameters(high_growth_years=8)
    res = monte_carlo(p, f, MonteCarloParameters(num_simulations=50, seed=3))
    for i in range(50):
        g, pi, tp, erp = res.draws[i]
        scalar = justified_pe(
            SolowParameters(g=g),
            FinancialParameters(
                expected_inflation=pi, term_premium=tp, equity_risk_premium=erp, high_growth_years=8
            ),
        )
        assert res.pe[i] == pytest.approx(scalar.justified_pe, rel=1e-12)


def test_monte_carlo_simulated_method_matches_closed_form_closely():
    p, f = SolowParameters(), FinancialParameters()
    mc = MonteCarloParameters(num_simulations=2000, seed=11)
    a = monte_carlo(p, f, mc).summary.mean
    b = monte_carlo(p, f, mc, method="simulated").summary.mean
    assert a == pytest.approx(b, rel=1e-3)


def test_monte_carlo_correlation_is_honoured():
    corr = np.eye(4)
    corr[1, 2] = corr[2, 2 - 1] = 0.6
    mc = MonteCarloParameters(num_simulations=20000, seed=5, correlation=corr, erp_std=0.01)
    res = monte_carlo(SolowParameters(), FinancialParameters(), mc)
    assert np.corrcoef(res.draws[:, 1], res.draws[:, 2])[0, 1] == pytest.approx(0.6, abs=0.03)
    assert res.draws[:, 3].std() == pytest.approx(0.01, rel=0.05)


def test_monte_carlo_summary_reports_invalid_share():
    s = monte_carlo_summary(np.array([1.0, 2.0, np.nan, 3.0]))
    assert s.n == 4 and s.n_valid == 3
    assert s.invalid_share == pytest.approx(0.25)
    assert s.mean == pytest.approx(2.0)
    assert s.percentiles[50] == pytest.approx(2.0)
    empty = monte_carlo_summary(np.array([np.nan, np.nan]))
    assert empty.invalid_share == 1.0 and np.isnan(empty.mean)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_validation_rejects_bad_inputs():
    for kw in ({"s": -0.1}, {"alpha": 1.5}, {"T": 1}, {"K0": 0}):
        with pytest.raises(ValueError):
            SolowParameters(**kw)
    for kw in (
        {"retention_rate": 1.0},
        {"earnings_growth_factor": 0},
        {"high_growth_years": 0},
        {"equity_risk_premium": -0.01},
    ):
        with pytest.raises(ValueError):
            FinancialParameters(**kw)
    for kw in (
        {"num_simulations": 0},
        {"g_std": -1},
        {"correlation": np.eye(3)},
        {"correlation": [[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
    ):
        with pytest.raises(ValueError):
            MonteCarloParameters(**kw)
