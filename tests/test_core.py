"""Verify the model math and the integration with the financial layer."""

from __future__ import annotations

import numpy as np
import pytest

from economics_models import (
    FinancialParameters,
    MonteCarloParameters,
    SolowParameters,
    gdp_sensitivity,
    justified_pe,
    monte_carlo_pe,
    pe_sensitivity,
    production,
    simulate_solow,
    steady_state_growth_rate,
)


def test_production_constant_returns_to_K_and_L():
    # Doubling K and L (with A fixed) doubles Y — constant returns to scale.
    y1 = production(100, 1, 100, 0.35)
    y2 = production(200, 1, 200, 0.35)
    assert y2 == pytest.approx(2 * y1, rel=1e-12)


def test_solow_steady_state_matches_geometric_rate():
    # In this model A_t = A_0(1+g)^t, L_t = L_0(1+n)^t, so Y grows at (1+n)(1+g)-1.
    p = SolowParameters(T=600)
    expected = ((1 + p.n) * (1 + p.g) - 1) * 100
    assert steady_state_growth_rate(p) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("g", [0.005, 0.02, 0.04])
def test_steady_state_tracks_g(g):
    p = SolowParameters(g=g, T=600)
    expected = ((1 + p.n) * (1 + g) - 1) * 100
    assert steady_state_growth_rate(p) == pytest.approx(expected, rel=2e-3)


def test_simulate_solow_shape_and_growth_length():
    p = SolowParameters(T=50)
    path = simulate_solow(p)
    assert path.K.shape == (50,)
    assert path.growth_rates.shape == (49,)
    assert np.all(np.isfinite(path.Y))


def test_justified_pe_basic():
    p = SolowParameters()
    f = FinancialParameters()
    res = justified_pe(p, f)
    # Required return must exceed earnings growth in this regime
    assert res.required_return > res.earnings_growth_rate
    assert res.justified_pe is not None
    assert res.justified_pe > 0


def test_justified_pe_invalid_when_growth_exceeds_required_return():
    p = SolowParameters(g=0.10)  # very high growth
    f = FinancialParameters(beta=0.1, equity_risk_premium=0.001, earnings_growth_factor=2.5)
    res = justified_pe(p, f)
    assert res.justified_pe is None


def test_pe_increases_with_g_when_in_valid_region():
    p = SolowParameters()
    f = FinancialParameters()
    g_grid, pe = pe_sensitivity(p, f, g_min=0.01, g_max=0.025, steps=8)
    finite = pe[~np.isnan(pe)]
    # P/E should be monotone non-decreasing in g while valid
    assert np.all(np.diff(finite) >= -1e-9)


def test_gdp_sensitivity_is_linear_in_g():
    p = SolowParameters(T=500)
    g_grid, gdp = gdp_sensitivity(p, g_min=0.01, g_max=0.03, steps=11)
    # Linearity: corr should be ~ 1
    corr = np.corrcoef(g_grid, gdp)[0, 1]
    assert corr > 0.999


def test_monte_carlo_returns_expected_shape_and_finite_mean():
    p = SolowParameters(T=200)
    f = FinancialParameters()
    mc = MonteCarloParameters(num_simulations=300, seed=42)
    pe, eg = monte_carlo_pe(p, f, mc)
    assert pe.shape == (300,)
    assert eg.shape == (300,)
    assert np.isfinite(np.nanmean(pe))


def test_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        SolowParameters(s=-0.1)
    with pytest.raises(ValueError):
        SolowParameters(alpha=1.5)
    with pytest.raises(ValueError):
        SolowParameters(T=1)
