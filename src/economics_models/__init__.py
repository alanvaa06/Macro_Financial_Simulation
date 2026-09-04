"""Integrated macro-financial valuation model.

Public API:
    SolowParameters, FinancialParameters, MonteCarloParameters
    simulate_solow, steady_state_growth_rate, convergence_gap,
    steady_state_output_capital_ratio
    justified_pe, gdp_sensitivity, pe_sensitivity, tornado
    implied_technology_growth, implied_equity_risk_premium
    compare_scenarios, scenarios_to_csv
    monte_carlo, monte_carlo_pe, monte_carlo_summary
    plot_solow_growth, plot_gdp_sensitivity, plot_pe_sensitivity,
    plot_tornado, plot_monte_carlo, dashboard
"""

from .core import (
    DEFAULT_TORNADO_SHOCKS,
    FinancialParameters,
    MonteCarloParameters,
    MonteCarloResult,
    MonteCarloSummary,
    SolowParameters,
    SolowPath,
    TornadoRow,
    ValuationResult,
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
    next_period_capital,
    pe_sensitivity,
    production,
    scenarios_to_csv,
    simulate_solow,
    steady_state_growth_rate,
    steady_state_output_capital_ratio,
    tornado,
)
from .visualization import (
    dashboard,
    plot_gdp_sensitivity,
    plot_monte_carlo,
    plot_pe_sensitivity,
    plot_solow_growth,
    plot_tornado,
)

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_TORNADO_SHOCKS",
    "SolowParameters",
    "FinancialParameters",
    "MonteCarloParameters",
    "MonteCarloResult",
    "MonteCarloSummary",
    "SolowPath",
    "TornadoRow",
    "ValuationResult",
    "closed_form_growth_rate",
    "compare_scenarios",
    "convergence_gap",
    "gdp_sensitivity",
    "implied_equity_risk_premium",
    "implied_technology_growth",
    "justified_pe",
    "monte_carlo",
    "monte_carlo_pe",
    "monte_carlo_summary",
    "next_period_capital",
    "pe_sensitivity",
    "production",
    "scenarios_to_csv",
    "simulate_solow",
    "steady_state_growth_rate",
    "steady_state_output_capital_ratio",
    "tornado",
    "plot_solow_growth",
    "plot_gdp_sensitivity",
    "plot_pe_sensitivity",
    "plot_tornado",
    "plot_monte_carlo",
    "dashboard",
    "__version__",
]
