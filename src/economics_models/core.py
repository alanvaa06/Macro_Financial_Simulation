"""Core economic models: Solow-Swan growth, CAPM, Gordon Growth, Monte Carlo.

The math here mirrors `web/app.js` line-for-line. If you change one, change the
other and re-run `tests/test_core.py` and the JS console check in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class SolowParameters:
    s: float = 0.20          # savings rate
    n: float = 0.005         # population growth
    g: float = 0.02          # technological progress
    delta: float = 0.05      # depreciation
    alpha: float = 0.35      # capital elasticity of output
    A0: float = 1.0          # initial technology level
    L0: float = 100.0        # initial labor force
    K0: float = 10000.0      # initial capital stock
    T: int = 200             # simulation periods

    def __post_init__(self) -> None:
        if not 0 <= self.s <= 1:
            raise ValueError("s must be in [0, 1]")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.delta < 0 or self.n < 0 or self.g < 0:
            raise ValueError("delta, n, g must be non-negative")
        if self.T < 2:
            raise ValueError("T must be >= 2")


@dataclass
class FinancialParameters:
    beta: float = 1.0
    equity_risk_premium: float = 0.05
    expected_inflation: float = 0.02
    term_premium: float = 0.01
    retention_rate: float = 0.35      # b; payout ratio = 1 - b
    earnings_growth_factor: float = 1.1  # tfp multiplier on nominal growth


@dataclass
class MonteCarloParameters:
    num_simulations: int = 5000
    g_mean: float = 0.02
    g_std: float = 0.005
    inflation_mean: float = 0.02
    inflation_std: float = 0.005
    term_premium_mean: float = 0.01
    term_premium_std: float = 0.002
    seed: Optional[int] = None


@dataclass
class SolowPath:
    K: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    L: np.ndarray
    growth_rates: np.ndarray  # YoY % growth of Y, length T-1


def production(K: float, A: float, L: float, alpha: float) -> float:
    """Cobb-Douglas: Y = K^alpha * (A*L)^(1-alpha)."""
    return (K ** alpha) * (A * L) ** (1 - alpha)


def next_period_capital(K: float, Y: float, p: SolowParameters) -> float:
    """K_{t+1} = (1 + n + g) * (K_t + s*Y_t - delta*K_t)."""
    return (1 + p.n + p.g) * (K + p.s * Y - p.delta * K)


def simulate_solow(p: SolowParameters) -> SolowPath:
    T = p.T
    K = np.zeros(T)
    Y = np.zeros(T)
    A = np.zeros(T)
    L = np.zeros(T)

    K[0], A[0], L[0] = p.K0, p.A0, p.L0
    Y[0] = production(K[0], A[0], L[0], p.alpha)

    for t in range(1, T):
        K[t] = next_period_capital(K[t - 1], Y[t - 1], p)
        L[t] = L[t - 1] * (1 + p.n)
        A[t] = A[t - 1] * (1 + p.g)
        Y[t] = production(K[t], A[t], L[t], p.alpha)

    growth = (Y[1:] - Y[:-1]) / Y[:-1] * 100.0
    return SolowPath(K=K, Y=Y, A=A, L=L, growth_rates=growth)


def steady_state_growth_rate(p: SolowParameters) -> float:
    """Final-period YoY % growth rate. Approaches (n + g)*100 as T grows."""
    return float(simulate_solow(p).growth_rates[-1])


@dataclass
class ValuationResult:
    real_growth_rate: float
    nominal_growth_rate: float
    risk_free_rate: float
    required_return: float
    earnings_growth_rate: float
    justified_pe: Optional[float]   # None when nominal_growth >= required_return

    def as_dict(self) -> dict:
        return asdict(self)


def justified_pe(solow: SolowParameters, fin: FinancialParameters) -> ValuationResult:
    """Compute the justified forward P/E from macro + financial inputs.

    Pipeline:
      real growth = Solow steady-state YoY %
      nominal growth = real + expected inflation
      risk-free = real + inflation + term premium      (Fisher + term structure)
      required return = risk-free + beta * ERP          (CAPM)
      earnings growth = nominal growth * earnings_growth_factor
      P/E = (1 - b) / (required_return - earnings_growth)  (Gordon)
    """
    real_g = steady_state_growth_rate(solow) / 100.0
    nominal_g = real_g + fin.expected_inflation
    rf = real_g + fin.expected_inflation + fin.term_premium
    rr = rf + fin.beta * fin.equity_risk_premium
    eg = nominal_g * fin.earnings_growth_factor

    if eg >= rr:
        pe: Optional[float] = None
    else:
        pe = (1 - fin.retention_rate) / (rr - eg)

    return ValuationResult(
        real_growth_rate=real_g,
        nominal_growth_rate=nominal_g,
        risk_free_rate=rf,
        required_return=rr,
        earnings_growth_rate=eg,
        justified_pe=pe,
    )


def gdp_sensitivity(
    solow: SolowParameters,
    g_min: float = 0.01,
    g_max: float = 0.03,
    steps: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (g_grid, steady-state growth %) sweeping technological progress."""
    g_grid = np.linspace(g_min, g_max, steps)
    out = np.empty_like(g_grid)
    for i, g in enumerate(g_grid):
        p = SolowParameters(**{**asdict(solow), "g": float(g)})
        out[i] = steady_state_growth_rate(p)
    return g_grid, out


def pe_sensitivity(
    solow: SolowParameters,
    fin: FinancialParameters,
    g_min: float = 0.015,
    g_max: float = 0.03,
    steps: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    g_grid = np.linspace(g_min, g_max, steps)
    out = np.empty_like(g_grid)
    for i, g in enumerate(g_grid):
        p = SolowParameters(**{**asdict(solow), "g": float(g)})
        res = justified_pe(p, fin)
        out[i] = np.nan if res.justified_pe is None else res.justified_pe
    return g_grid, out


def monte_carlo_pe(
    solow: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (g, inflation, term_premium) ~ Normal and compute P/E + earnings growth.

    Returns (pe_ratios, earnings_growth_rates). Failed draws (where the Gordon
    denominator is non-positive) appear as NaN — filter at call sites.
    """
    rng = np.random.default_rng(mc.seed)
    n = mc.num_simulations
    pe = np.empty(n)
    eg = np.empty(n)

    g_draws = rng.normal(mc.g_mean, mc.g_std, size=n)
    pi_draws = rng.normal(mc.inflation_mean, mc.inflation_std, size=n)
    tp_draws = rng.normal(mc.term_premium_mean, mc.term_premium_std, size=n)

    for i in range(n):
        p = SolowParameters(**{**asdict(solow), "g": float(g_draws[i])})
        f = FinancialParameters(
            beta=fin.beta,
            equity_risk_premium=fin.equity_risk_premium,
            expected_inflation=float(pi_draws[i]),
            term_premium=float(tp_draws[i]),
            retention_rate=fin.retention_rate,
            earnings_growth_factor=fin.earnings_growth_factor,
        )
        res = justified_pe(p, f)
        pe[i] = np.nan if res.justified_pe is None else res.justified_pe
        eg[i] = res.earnings_growth_rate

    return pe, eg
