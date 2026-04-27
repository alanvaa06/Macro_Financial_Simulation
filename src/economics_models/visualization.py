"""Matplotlib visualizations for the Solow + valuation model."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .core import (
    SolowParameters,
    FinancialParameters,
    MonteCarloParameters,
    simulate_solow,
    gdp_sensitivity,
    pe_sensitivity,
    monte_carlo_pe,
)


def plot_solow_growth(p: SolowParameters, ax=None):
    path = simulate_solow(p)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, p.T), path.growth_rates, color="#0ea5a4", linewidth=2)
    ax.axhline((p.n + p.g) * 100, ls="--", color="#94a3b8", label="n+g (theoretical)")
    ax.set_xlabel("Period")
    ax.set_ylabel("YoY output growth (%)")
    ax.set_title("Solow-Swan output growth path")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_gdp_sensitivity(p: SolowParameters, ax=None):
    g_grid, gdp = gdp_sensitivity(p)
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


def plot_monte_carlo(
    p: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
):
    pe, eg = monte_carlo_pe(p, fin, mc)
    pe = pe[~np.isnan(pe)]
    eg = eg[~np.isnan(eg)] * 100

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

    fig.tight_layout()
    return fig


def dashboard(
    p: SolowParameters,
    fin: FinancialParameters,
    mc: MonteCarloParameters,
):
    """Single 2x2 figure with all four panels."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_solow_growth(p, ax=axes[0, 0])
    plot_gdp_sensitivity(p, ax=axes[0, 1])
    plot_pe_sensitivity(p, fin, ax=axes[1, 0])

    pe, _ = monte_carlo_pe(p, fin, mc)
    pe = pe[~np.isnan(pe)]
    ax = axes[1, 1]
    if len(pe):
        mean = float(np.mean(pe))
        ax.hist(pe, bins=30, alpha=0.75, color="#6366f1", edgecolor="white")
        ax.axvline(mean, color="black", ls="--", lw=1, label=f"mean={mean:.2f}")
        ax.legend()
    ax.set_title("Monte Carlo P/E distribution")
    ax.set_xlabel("P/E ratio")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
