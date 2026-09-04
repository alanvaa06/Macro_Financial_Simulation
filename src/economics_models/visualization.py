"""Matplotlib visualizations for the Solow + valuation model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .core import (
    FinancialParameters,
    MonteCarloParameters,
    SolowParameters,
    closed_form_growth_rate,
    convergence_gap,
    gdp_sensitivity,
    justified_pe,
    monte_carlo,
    pe_sensitivity,
    simulate_solow,
    tornado,
)


def plot_solow_growth(p: SolowParameters, ax=None):
    """Simulated growth path against the closed-form steady state."""
    path = simulate_solow(p)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, p.T), path.growth_rates, color="#0ea5a4", linewidth=2, label="simulated")
    target = float(closed_form_growth_rate(p.n, p.g))
    ax.axhline(target, ls="--", color="#94a3b8", label="(1+n)(1+g)−1")
    ax.set_xlabel("Period")
    ax.set_ylabel("YoY output growth (%)")
    ax.set_title(f"Solow-Swan convergence (gap at T: {convergence_gap(p):+.4f} pp)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_gdp_sensitivity(p: SolowParameters, ax=None, method="closed_form"):
    g_grid, gdp = gdp_sensitivity(p, method=method)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g_grid * 100, gdp, marker="o", color="#6366f1")
    ax.set_xlabel("Technological progress rate g (%)")
    ax.set_ylabel("Steady-state growth (%)")
    ax.set_title("GDP growth vs. technological progress")
    ax.grid(True, alpha=0.3)
    return ax


def plot_pe_sensitivity(p: SolowParameters, fin: FinancialParameters, ax=None):
    g_grid, pe = pe_sensitivity(p, fin)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g_grid * 100, pe, marker="o", color="#0ea5a4")
    ax.set_xlabel("Technological progress rate g (%)")
    ax.set_ylabel("Justified forward P/E")
    ax.set_title("Justified P/E vs. technological progress")
    ax.grid(True, alpha=0.3)
    return ax


def plot_tornado(p: SolowParameters, fin: FinancialParameters, ax=None, shocks=None):
    """Horizontal tornado chart: P/E swing per driver around the base case."""
    rows = tornado(p, fin, shocks)
    base = justified_pe(p, fin).justified_pe
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    labels = [r.parameter for r in rows][::-1]
    lows = [np.nan if r.pe_low is None else r.pe_low for r in rows][::-1]
    highs = [np.nan if r.pe_high is None else r.pe_high for r in rows][::-1]
    y = np.arange(len(rows))
    if base is not None:
        ax.barh(y, np.array(lows) - base, left=base, color="#ef4444", alpha=0.8, label="− shock")
        ax.barh(y, np.array(highs) - base, left=base, color="#0ea5a4", alpha=0.8, label="+ shock")
        ax.axvline(base, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Justified forward P/E")
    ax.set_title("Tornado: one-at-a-time sensitivity")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    return ax


def plot_monte_carlo(
    p: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
):
    res = monte_carlo(p, fin, mc)
    pe = res.pe[~np.isnan(res.pe)]
    eg = res.earnings_growth * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, data, color, title, xlabel in (
        (axes[0], pe, "#6366f1", "Distribution of justified P/E", "P/E ratio"),
        (axes[1], eg, "#0ea5a4", "Distribution of earnings growth", "Earnings growth (%)"),
    ):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No valid draws", ha="center", transform=ax.transAxes)
            continue
        mean, std = float(np.mean(data)), float(np.std(data))
        ax.hist(data, bins=30, alpha=0.75, color=color, edgecolor="white")
        ax.axvline(mean, color="black", ls="--", lw=1, label=f"mean={mean:.2f}")
        ax.axvline(mean + 2 * std, color="#ef4444", ls="--", lw=1, label="±2σ")
        ax.axvline(mean - 2 * std, color="#ef4444", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend()
    s = res.summary
    fig.suptitle(
        f"n={s.n}, invalid={s.invalid_share:.1%}, "
        f"P5/P50/P95 = {s.percentiles.get(5, float('nan')):.2f} / "
        f"{s.percentiles.get(50, float('nan')):.2f} / {s.percentiles.get(95, float('nan')):.2f}"
    )
    fig.tight_layout()
    return fig


def dashboard(
    p: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
):
    """Single 2x2 figure: convergence, P/E sensitivity, tornado, Monte Carlo."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_solow_growth(p, ax=axes[0, 0])
    plot_pe_sensitivity(p, fin, ax=axes[0, 1])
    plot_tornado(p, fin, ax=axes[1, 0])

    res = monte_carlo(p, fin, mc)
    pe = res.pe[~np.isnan(res.pe)]
    ax = axes[1, 1]
    if len(pe):
        s = res.summary
        ax.hist(pe, bins=30, alpha=0.75, color="#6366f1", edgecolor="white")
        ax.axvline(s.mean, color="black", ls="--", lw=1, label=f"mean={s.mean:.2f}")
        ax.axvline(s.percentiles[5], color="#ef4444", ls=":", lw=1, label="P5 / P95")
        ax.axvline(s.percentiles[95], color="#ef4444", ls=":", lw=1)
        ax.legend()
    ax.set_title(f"Monte Carlo P/E distribution (invalid {res.summary.invalid_share:.1%})")
    ax.set_xlabel("P/E ratio")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
