"""Core economic models: Solow-Swan growth, CAPM, Gordon Growth, Monte Carlo.

The math here mirrors ``web/app.js``. Parity between the two implementations is
enforced by ``tests/test_js_parity.py`` (skipped when Node.js is unavailable).

Model conventions
-----------------
* **Steady-state real growth** is the closed form ``(1+n)(1+g) - 1``. The
  simulated Solow path is a *convergence diagnostic*; it never feeds the
  valuation unless you explicitly ask for ``method="simulated"``.
* **Capital accumulation** follows the textbook aggregate law of motion
  ``K_{t+1} = (1 - delta) K_t + s Y_t``.
* **Earnings growth** may run above nominal GDP growth only for a finite
  number of years (``FinancialParameters.high_growth_years``). With
  ``high_growth_years=None`` the model collapses to the single-stage Gordon
  formula and the ``earnings_outgrow_gdp`` flag warns when the stage-1 rate is
  assumed to persist forever.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

GrowthMethod = Literal["closed_form", "simulated"]

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass
class SolowParameters:
    s: float = 0.20  # savings rate
    n: float = 0.005  # population growth
    g: float = 0.02  # technological progress
    delta: float = 0.05  # depreciation
    alpha: float = 0.35  # capital elasticity of output
    A0: float = 1.0  # initial technology level
    L0: float = 100.0  # initial labor force
    K0: float = 200.0  # initial capital stock (below steady state -> catch-up path)
    T: int = 200  # simulation periods

    def __post_init__(self) -> None:
        if not 0 <= self.s <= 1:
            raise ValueError("s must be in [0, 1]")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.delta < 0 or self.n < 0 or self.g < 0:
            raise ValueError("delta, n, g must be non-negative")
        if self.delta > 1:
            raise ValueError("delta must be <= 1")
        if self.T < 2:
            raise ValueError("T must be >= 2")
        if self.A0 <= 0 or self.L0 <= 0 or self.K0 <= 0:
            raise ValueError("A0, L0, K0 must be positive")


@dataclass
class FinancialParameters:
    beta: float = 1.0
    equity_risk_premium: float = 0.05
    expected_inflation: float = 0.02
    term_premium: float = 0.01
    retention_rate: float = 0.35  # b; payout ratio = 1 - b
    earnings_growth_factor: float = 1.1  # stage-1 multiplier on nominal growth
    high_growth_years: int | None = None  # None -> single-stage Gordon

    def __post_init__(self) -> None:
        if not 0 <= self.retention_rate < 1:
            raise ValueError("retention_rate must be in [0, 1)")
        if self.earnings_growth_factor <= 0:
            raise ValueError("earnings_growth_factor must be positive")
        if self.equity_risk_premium < 0:
            raise ValueError("equity_risk_premium must be non-negative")
        if self.high_growth_years is not None and self.high_growth_years < 1:
            raise ValueError("high_growth_years must be >= 1 or None")
        for name in ("beta", "expected_inflation", "term_premium"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


@dataclass
class MonteCarloParameters:
    """Distributional assumptions for the stochastic inputs.

    Means default to the point estimates in the ``SolowParameters`` /
    ``FinancialParameters`` they are run against (``None``), which is what the
    web UI does. ``correlation`` is a 4x4 matrix over ``(g, inflation,
    term_premium, equity_risk_premium)``; ``None`` means independent draws.
    """

    num_simulations: int = 5000
    g_mean: float | None = None
    g_std: float = 0.005
    inflation_mean: float | None = None
    inflation_std: float = 0.005
    term_premium_mean: float | None = None
    term_premium_std: float = 0.002
    erp_std: float = 0.0
    correlation: Sequence[Sequence[float]] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        for name in ("g_std", "inflation_std", "term_premium_std", "erp_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.correlation is not None:
            c = np.asarray(self.correlation, dtype=float)
            if c.shape != (4, 4):
                raise ValueError("correlation must be 4x4 over (g, inflation, tp, erp)")
            if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
                raise ValueError("correlation must be symmetric with unit diagonal")
            if np.min(np.linalg.eigvalsh(c)) < -1e-10:
                raise ValueError("correlation must be positive semi-definite")

    def correlation_matrix(self) -> np.ndarray:
        if self.correlation is None:
            return np.eye(4)
        return np.asarray(self.correlation, dtype=float)


# --------------------------------------------------------------------------- #
# Solow-Swan
# --------------------------------------------------------------------------- #


@dataclass
class SolowPath:
    K: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    L: np.ndarray
    growth_rates: np.ndarray  # YoY % growth of Y, length T-1


def production(K, A, L, alpha: float):
    """Cobb-Douglas: Y = K^alpha * (A*L)^(1-alpha). Works on scalars or arrays."""
    return (K**alpha) * (A * L) ** (1 - alpha)


def next_period_capital(K, Y, p: SolowParameters):
    """Textbook aggregate law of motion: K_{t+1} = (1 - delta) K_t + s Y_t."""
    return (1 - p.delta) * K + p.s * Y


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


def _simulated_final_growth_vec(g: np.ndarray, p: SolowParameters) -> np.ndarray:
    """Final-period YoY % growth for a *vector* of technology growth rates.

    Runs the T-step recurrence once with every draw carried along axis 0.
    """
    g = np.asarray(g, dtype=float)
    K = np.full_like(g, p.K0)
    A = np.full_like(g, p.A0)
    L = np.full_like(g, p.L0)
    Y = production(K, A, L, p.alpha)
    Y_prev = Y
    for _ in range(1, p.T):
        Y_prev = Y
        K = next_period_capital(K, Y, p)
        L = L * (1 + p.n)
        A = A * (1 + g)
        Y = production(K, A, L, p.alpha)
    return (Y - Y_prev) / Y_prev * 100.0


def closed_form_growth_rate(n: float, g) -> np.ndarray | float:
    """Steady-state YoY % growth of aggregate output: ((1+n)(1+g) - 1) * 100."""
    return ((1 + n) * (1 + np.asarray(g, dtype=float)) - 1) * 100.0


def steady_state_growth_rate(p: SolowParameters, method: GrowthMethod = "closed_form") -> float:
    """Real growth rate (%) that feeds the valuation.

    ``closed_form`` (default) returns the analytical asymptote, which is
    independent of ``s``, ``delta``, ``alpha``, ``K0`` and ``T``.
    ``simulated`` returns the final-period growth of the simulated path, which
    is still on the transition path for small ``T`` or extreme ``K0``.
    """
    if method == "closed_form":
        return float(closed_form_growth_rate(p.n, p.g))
    if method == "simulated":
        return float(simulate_solow(p).growth_rates[-1])
    raise ValueError(f"unknown method {method!r}")


def steady_state_output_capital_ratio(p: SolowParameters) -> float:
    """Analytical steady-state Y/K = ((1+n)(1+g) - (1-delta)) / s."""
    if p.s == 0:
        return float("inf")
    return ((1 + p.n) * (1 + p.g) - (1 - p.delta)) / p.s


def convergence_gap(p: SolowParameters) -> float:
    """Simulated final-period growth minus the closed form, in percentage points.

    A large |gap| means ``T`` is too short or ``K0`` too far from steady state
    for the simulated path to represent the long run.
    """
    return steady_state_growth_rate(p, "simulated") - steady_state_growth_rate(p, "closed_form")


# --------------------------------------------------------------------------- #
# Valuation (vectorised core)
# --------------------------------------------------------------------------- #


def _pe_from_rates(rr, eg, g_term, fin: FinancialParameters):
    """Justified forward P/E (P0/E1) from required return and growth rates.

    All inputs broadcast. Returns NaN where the terminal growth rate is not
    below the required return.
    """
    rr = np.asarray(rr, dtype=float)
    eg = np.asarray(eg, dtype=float)
    g_term = np.asarray(g_term, dtype=float)
    payout = 1 - fin.retention_rate
    with np.errstate(divide="ignore", invalid="ignore"):
        if fin.high_growth_years is None:
            valid = eg < rr
            pe = np.where(valid, payout / (rr - eg), np.nan)
        else:
            N = fin.high_growth_years
            valid = g_term < rr
            r = (1 + eg) / (1 + rr)
            # sum_{t=1}^{N} (1+eg)^(t-1) / (1+rr)^t
            annuity = np.where(
                np.isclose(r, 1.0),
                N / (1 + rr),
                (1 - r**N) / (1 - r) / (1 + rr),
            )
            terminal = (1 + eg) ** (N - 1) * (1 + g_term) / ((rr - g_term) * (1 + rr) ** N)
            pe = np.where(valid, payout * (annuity + terminal), np.nan)
    return pe


def _valuation_arrays(real_g, inflation, term_premium, erp, fin: FinancialParameters) -> dict:
    """Vectorised pipeline from real growth (decimal) to P/E. Inputs broadcast."""
    real_g = np.asarray(real_g, dtype=float)
    nominal_g = real_g + inflation
    rf = nominal_g + term_premium
    rr = rf + fin.beta * erp
    eg = nominal_g * fin.earnings_growth_factor
    g_term = eg if fin.high_growth_years is None else nominal_g
    pe = _pe_from_rates(rr, eg, g_term, fin)
    h = 1e-6
    pe_up = _pe_from_rates(rr + h, eg, g_term, fin)
    pe_dn = _pe_from_rates(rr - h, eg, g_term, fin)
    with np.errstate(divide="ignore", invalid="ignore"):
        duration = -(pe_up - pe_dn) / (2 * h) / pe
    return {
        "real_growth_rate": real_g,
        "nominal_growth_rate": nominal_g,
        "risk_free_rate": rf,
        "required_return": rr,
        "earnings_growth_rate": eg,
        "terminal_growth_rate": g_term,
        "justified_pe": pe,
        "pe_duration": duration,
    }


@dataclass
class ValuationResult:
    real_growth_rate: float
    nominal_growth_rate: float
    risk_free_rate: float
    required_return: float
    earnings_growth_rate: float  # stage-1 (or perpetual) earnings growth
    terminal_growth_rate: float  # growth the perpetuity is discounted at
    justified_pe: float | None  # None when terminal growth >= required return
    pe_duration: float | None  # -(dPE/dk)/PE, in years; None when invalid
    earnings_outgrow_gdp: bool  # earnings assumed to grow faster than nominal GDP forever

    def as_dict(self) -> dict:
        return asdict(self)


def justified_pe(
    solow: SolowParameters,
    fin: FinancialParameters,
    method: GrowthMethod = "closed_form",
) -> ValuationResult:
    """Compute the justified forward P/E from macro + financial inputs.

    Pipeline:
      real growth = Solow steady state (closed form by default)
      nominal growth = real + expected inflation
      risk-free = nominal growth + term premium              (Fisher + term structure)
      required return = risk-free + beta * ERP               (CAPM)
      earnings growth = nominal growth * earnings_growth_factor
      P/E = (1 - b) / (k - g_e)                              (Gordon, single stage)
          or the two-stage equivalent when high_growth_years is set.
    """
    real_g = steady_state_growth_rate(solow, method) / 100.0
    v = _valuation_arrays(
        real_g, fin.expected_inflation, fin.term_premium, fin.equity_risk_premium, fin
    )
    pe = float(v["justified_pe"])
    dur = float(v["pe_duration"])
    return ValuationResult(
        real_growth_rate=float(v["real_growth_rate"]),
        nominal_growth_rate=float(v["nominal_growth_rate"]),
        risk_free_rate=float(v["risk_free_rate"]),
        required_return=float(v["required_return"]),
        earnings_growth_rate=float(v["earnings_growth_rate"]),
        terminal_growth_rate=float(v["terminal_growth_rate"]),
        justified_pe=None if np.isnan(pe) else pe,
        pe_duration=None if np.isnan(dur) else dur,
        earnings_outgrow_gdp=bool(
            fin.high_growth_years is None and fin.earnings_growth_factor > 1.0
        ),
    )


# --------------------------------------------------------------------------- #
# Sensitivities
# --------------------------------------------------------------------------- #


def gdp_sensitivity(
    solow: SolowParameters,
    g_min: float = 0.01,
    g_max: float = 0.03,
    steps: int = 21,
    method: GrowthMethod = "closed_form",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (g_grid, steady-state growth %) sweeping technological progress."""
    g_grid = np.linspace(g_min, g_max, steps)
    if method == "closed_form":
        return g_grid, np.asarray(closed_form_growth_rate(solow.n, g_grid))
    return g_grid, _simulated_final_growth_vec(g_grid, solow)


def pe_sensitivity(
    solow: SolowParameters,
    fin: FinancialParameters,
    g_min: float = 0.015,
    g_max: float = 0.03,
    steps: int = 31,
    method: GrowthMethod = "closed_form",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (g_grid, justified P/E) sweeping technological progress. Invalid -> NaN."""
    g_grid, growth = gdp_sensitivity(solow, g_min, g_max, steps, method)
    v = _valuation_arrays(
        growth / 100.0, fin.expected_inflation, fin.term_premium, fin.equity_risk_premium, fin
    )
    return g_grid, v["justified_pe"]


@dataclass
class TornadoRow:
    parameter: str
    low_value: float
    high_value: float
    pe_low: float | None
    pe_high: float | None
    swing: float  # |pe_high - pe_low|; inf when one side is invalid


DEFAULT_TORNADO_SHOCKS: dict[str, float] = {
    "g": 0.005,
    "n": 0.0025,
    "expected_inflation": 0.005,
    "term_premium": 0.0025,
    "equity_risk_premium": 0.01,
    "beta": 0.2,
    "retention_rate": 0.10,
    "earnings_growth_factor": 0.2,
}


def _with_param(
    solow: SolowParameters, fin: FinancialParameters, name: str, value: float
) -> tuple[SolowParameters, FinancialParameters]:
    if hasattr(solow, name):
        return SolowParameters(**{**asdict(solow), name: value}), fin
    if hasattr(fin, name):
        return solow, FinancialParameters(**{**asdict(fin), name: value})
    raise KeyError(f"unknown parameter {name!r}")


def _get_param(solow: SolowParameters, fin: FinancialParameters, name: str) -> float:
    if hasattr(solow, name):
        return float(getattr(solow, name))
    if hasattr(fin, name):
        return float(getattr(fin, name))
    raise KeyError(f"unknown parameter {name!r}")


def tornado(
    solow: SolowParameters,
    fin: FinancialParameters,
    shocks: Mapping[str, float] | None = None,
    method: GrowthMethod = "closed_form",
) -> list[TornadoRow]:
    """One-at-a-time +/- shocks on every driver, sorted by P/E swing (largest first).

    ``shocks`` maps parameter name -> absolute half-width. Parameters whose
    shocked value would fail validation are clipped to the valid range.
    """
    shocks = dict(DEFAULT_TORNADO_SHOCKS if shocks is None else shocks)
    rows: list[TornadoRow] = []
    for name, h in shocks.items():
        base = _get_param(solow, fin, name)
        pes: list[float | None] = []
        vals: list[float] = []
        for v in (base - h, base + h):
            try:
                s2, f2 = _with_param(solow, fin, name, v)
            except ValueError:
                v = max(v, 0.0)
                s2, f2 = _with_param(solow, fin, name, v)
            vals.append(v)
            pes.append(justified_pe(s2, f2, method).justified_pe)
        lo, hi = pes
        swing = float("inf") if lo is None or hi is None else abs(hi - lo)
        rows.append(TornadoRow(name, vals[0], vals[1], lo, hi, swing))
    rows.sort(key=lambda r: r.swing, reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# Inversion: what does a market multiple imply?
# --------------------------------------------------------------------------- #


def _bisect(f, lo: float, hi: float, tol: float = 1e-10, max_iter: int = 200) -> float | None:
    """Root of a monotone ``f`` on [lo, hi]; ``None`` if no sign change."""
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return None
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if np.isnan(fmid):
            return None
        if abs(hi - lo) < tol:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def _pe_or_inf(solow: SolowParameters, fin: FinancialParameters, method: GrowthMethod) -> float:
    pe = justified_pe(solow, fin, method).justified_pe
    return float("inf") if pe is None else pe


def implied_technology_growth(
    target_pe: float,
    solow: SolowParameters,
    fin: FinancialParameters,
    g_min: float = 0.0,
    g_max: float = 0.10,
    method: GrowthMethod = "closed_form",
) -> float | None:
    """Technology growth ``g`` at which the justified P/E equals ``target_pe``.

    ``None`` if no ``g`` in [g_min, g_max] reproduces the target (for example
    the target is below the zero-growth P/E, or above the P/E at which the
    Gordon denominator closes).
    """
    if target_pe <= 0:
        raise ValueError("target_pe must be positive")

    def f(g: float) -> float:
        return _pe_or_inf(replace(solow, g=g), fin, method) - target_pe

    # Restrict to the valid region so bisection sees a finite, monotone function.
    hi = g_max
    if not np.isfinite(f(hi)):
        edge = _bisect(lambda g: (0.0 if np.isfinite(f(g)) else 1.0) - 0.5, g_min, g_max, 1e-9)
        if edge is None:
            return None
        hi = max(g_min, edge - 1e-9)
    return _bisect(f, g_min, hi)


def implied_equity_risk_premium(
    target_pe: float,
    solow: SolowParameters,
    fin: FinancialParameters,
    erp_min: float = 0.0,
    erp_max: float = 0.20,
    method: GrowthMethod = "closed_form",
) -> float | None:
    """Equity risk premium at which the justified P/E equals ``target_pe``.

    ``None`` if no ERP in [erp_min, erp_max] reproduces the target.
    """
    if target_pe <= 0:
        raise ValueError("target_pe must be positive")

    def f(erp: float) -> float:
        return _pe_or_inf(solow, replace(fin, equity_risk_premium=erp), method) - target_pe

    lo = erp_min
    if not np.isfinite(f(lo)):
        edge = _bisect(lambda e: (1.0 if np.isfinite(f(e)) else 0.0) - 0.5, erp_min, erp_max, 1e-9)
        if edge is None:
            return None
        lo = min(erp_max, edge + 1e-9)
    return _bisect(f, lo, erp_max)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def compare_scenarios(
    scenarios: Mapping[str, tuple[SolowParameters, FinancialParameters]],
    method: GrowthMethod = "closed_form",
) -> dict[str, ValuationResult]:
    """Run the valuation for each named (solow, fin) pair."""
    return {name: justified_pe(s, f, method) for name, (s, f) in scenarios.items()}


def scenarios_to_csv(results: Mapping[str, ValuationResult]) -> str:
    """Render ``compare_scenarios`` output as CSV text (one row per scenario)."""
    buf = io.StringIO()
    fields = ["scenario", *ValuationResult.__dataclass_fields__.keys()]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for name, res in results.items():
        writer.writerow({"scenario": name, **res.as_dict()})
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #


@dataclass
class MonteCarloSummary:
    n: int
    n_valid: int
    invalid_share: float
    mean: float
    std: float
    std_error: float
    percentiles: dict[int, float] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    pe: np.ndarray  # NaN where the Gordon denominator is non-positive
    earnings_growth: np.ndarray  # decimal
    draws: np.ndarray  # shape (n, 4): g, inflation, term_premium, erp
    summary: MonteCarloSummary


def monte_carlo_summary(
    values: np.ndarray, percentiles: Sequence[int] = (5, 25, 50, 75, 95)
) -> MonteCarloSummary:
    """Summarise a (possibly NaN-containing) sample, reporting the invalid share."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    n, nv = int(values.size), int(valid.size)
    if nv == 0:
        return MonteCarloSummary(n, 0, 1.0, float("nan"), float("nan"), float("nan"), {})
    std = float(np.std(valid, ddof=1)) if nv > 1 else 0.0
    return MonteCarloSummary(
        n=n,
        n_valid=nv,
        invalid_share=1.0 - nv / n,
        mean=float(np.mean(valid)),
        std=std,
        std_error=std / np.sqrt(nv),
        percentiles={
            int(q): float(v) for q, v in zip(percentiles, np.percentile(valid, percentiles))
        },
    )


def _cholesky(corr: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor; falls back to a jittered factor for singular PSD input."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(corr + 1e-12 * np.eye(corr.shape[0]))


def standard_normal_draws(mc: MonteCarloParameters) -> np.ndarray:
    """Independent N(0,1) draws, shape (num_simulations, 4)."""
    rng = np.random.default_rng(mc.seed)
    return rng.standard_normal(size=(mc.num_simulations, 4))


def monte_carlo(
    solow: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
    method: GrowthMethod = "closed_form",
    z: np.ndarray | None = None,
) -> MonteCarloResult:
    """Draw (g, inflation, term_premium, erp) and compute the P/E distribution.

    The whole simulation is vectorised: one pass over the draws, no Python
    loop per draw. ``z`` lets callers inject standard-normal draws of shape
    (n, 4) for reproducibility across implementations.
    """
    z = standard_normal_draws(mc) if z is None else np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 4:
        raise ValueError("z must have shape (n, 4)")
    chol = _cholesky(mc.correlation_matrix())
    x = z @ chol.T
    means = np.array(
        [
            solow.g if mc.g_mean is None else mc.g_mean,
            fin.expected_inflation if mc.inflation_mean is None else mc.inflation_mean,
            fin.term_premium if mc.term_premium_mean is None else mc.term_premium_mean,
            fin.equity_risk_premium,
        ]
    )
    stds = np.array([mc.g_std, mc.inflation_std, mc.term_premium_std, mc.erp_std])
    draws = means + stds * x
    g, pi, tp, erp = draws.T
    g = np.clip(g, 0.0, None)
    erp = np.clip(erp, 0.0, None)

    if method == "closed_form":
        real = np.asarray(closed_form_growth_rate(solow.n, g)) / 100.0
    else:
        real = _simulated_final_growth_vec(g, solow) / 100.0
    v = _valuation_arrays(real, pi, tp, erp, fin)
    pe = np.asarray(v["justified_pe"], dtype=float)
    return MonteCarloResult(
        pe=pe,
        earnings_growth=np.asarray(v["earnings_growth_rate"], dtype=float),
        draws=np.column_stack([g, pi, tp, erp]),
        summary=monte_carlo_summary(pe),
    )


def monte_carlo_pe(
    solow: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
    method: GrowthMethod = "closed_form",
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper: returns (pe_ratios, earnings_growth_rates).

    Failed draws appear as NaN in ``pe_ratios`` -- filter at call sites, or use
    :func:`monte_carlo` which reports the invalid share explicitly.
    """
    res = monte_carlo(solow, fin, mc, method)
    return res.pe, res.earnings_growth
