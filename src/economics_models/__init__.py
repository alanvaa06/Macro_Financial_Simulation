"""Integrated macro-financial valuation model.

Public API:
    SolowParameters, FinancialParameters, MonteCarloParameters
    simulate_solow, steady_state_growth_rate
    justified_pe, gdp_sensitivity, pe_sensitivity, monte_carlo_pe
    plot_solow_growth, plot_gdp_sensitivity, plot_pe_sensitivity,
    plot_monte_carlo, dashboard
"""

from .core import (
    SolowParameters,
    FinancialParameters,
    MonteCarloParameters,
    SolowPath,
    ValuationResult,
    simulate_solow,
    steady_state_growth_rate,
    justified_pe,
    gdp_sensitivity,
    pe_sensitivity,
    monte_carlo_pe,
    production,
    next_period_capital,
)
from .visualization import (
    plot_solow_growth,
    plot_gdp_sensitivity,
    plot_pe_sensitivity,
    plot_monte_carlo,
    dashboard,
)

__version__ = "1.1.0"

__all__ = [
    "SolowParameters",
    "FinancialParameters",
    "MonteCarloParameters",
    "SolowPath",
    "ValuationResult",
    "simulate_solow",
    "steady_state_growth_rate",
    "justified_pe",
    "gdp_sensitivity",
    "pe_sensitivity",
    "monte_carlo_pe",
    "production",
    "next_period_capital",
    "plot_solow_growth",
    "plot_gdp_sensitivity",
    "plot_pe_sensitivity",
    "plot_monte_carlo",
    "dashboard",
    "__version__",
]
