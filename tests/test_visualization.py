"""Smoke tests: every renderer produces a figure without raising."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from economics_models import (  # noqa: E402
    FinancialParameters,
    MonteCarloParameters,
    SolowParameters,
    dashboard,
    plot_gdp_sensitivity,
    plot_monte_carlo,
    plot_pe_sensitivity,
    plot_solow_growth,
    plot_tornado,
)


def teardown_function(_):
    plt.close("all")


def test_single_panel_plots_return_axes():
    p, f = SolowParameters(T=60), FinancialParameters()
    assert plot_solow_growth(p).get_title()
    assert plot_gdp_sensitivity(p).get_title()
    assert plot_pe_sensitivity(p, f).get_title()
    assert plot_tornado(p, f).get_title()


def test_monte_carlo_and_dashboard_return_figures():
    p, f = SolowParameters(T=60), FinancialParameters()
    mc = MonteCarloParameters(num_simulations=200, seed=1)
    assert len(plot_monte_carlo(p, f, mc).axes) == 2
    assert len(dashboard(p, f, mc).axes) == 4


def test_monte_carlo_plot_handles_no_valid_draws():
    p = SolowParameters(g=0.10)
    f = FinancialParameters(beta=0.1, equity_risk_premium=0.001, earnings_growth_factor=2.5)
    fig = plot_monte_carlo(p, f, MonteCarloParameters(num_simulations=50, seed=1))
    assert fig is not None
