# Macro-Financial Valuation Lab

> Solow-Swan growth → CAPM → Gordon Growth → Monte Carlo, in one integrated model.
> Move sliders, watch the **justified forward P/E** update in real time — then ask the model what a market multiple *implies*.

An integrated macro-financial valuation framework. Long-run real growth from a Solow-Swan model feeds a CAPM required return, which a Gordon Growth model (single- or two-stage) turns into a justified forward P/E. Around that point estimate you get a tornado sensitivity, a vectorised Monte Carlo distribution, P/E duration, scenario comparison, and inversion (implied technology growth and implied equity risk premium for an observed P/E).

You can use it three ways:

1. **Open `web/index.html` in a browser** — zero install, fully interactive.
2. **`import economics_models`** — typed Python API for analysts and notebooks.
3. **Open `EconomicGrowth.ipynb`** — the original research notebook. *Historical reference only*: it predates the v2 model conventions below and its defaults differ from the package.

---

## 1. Web interface

```bash
python -m http.server 8000 --directory web   # then visit http://localhost:8000
```

or just open the file. The `web/` folder is self-contained apart from Plotly (partial `cartesian` bundle via CDN).

### What you get

- **Live KPIs**: real and nominal growth, risk-free rate, required return, justified P/E, **P/E duration**, and a **convergence gap** that turns red when the simulated Solow path has not reached its steady state.
- **What does the market imply?** Enter an observed forward P/E and read off the implied technology growth and the implied equity risk premium, with the reachable range shown when the target is out of bounds.
- **Convergence path**: simulated growth against the closed-form steady state.
- **P/E sensitivity** to technology growth.
- **Tornado**: one-at-a-time ± shocks on all eight drivers, ranked by P/E swing.
- **Monte Carlo**: live, seeded, vectorised; reports μ, σ, P5–P95 and the **share of invalid draws**; optional ERP uncertainty and inflation/term-premium correlation.
- **Export JSON**: parameters, valuation, tornado, implied values and Monte Carlo summary in one file.
- **Parameter sidebar that fits one screen**: the three groups (Solow-Swan, Financial, Monte Carlo) form an exclusive accordion whose collapsed headers show their key values; convergence-path settings sit in a nested "advanced" fold. Every value can be typed as well as dragged, two-stage growth / ERP uncertainty / inflation–term-premium correlation are switches, and on narrow screens the sidebar becomes a drawer.
- **Latest observed values**: each input carries a note with its most recent observable counterpart (US / S&P 500), its source and as-of date, a ▲ marker on the slider, and a one-click apply; "Use observed" applies them all. Inputs with no observable counterpart say so. The numbers live in `web/observables.js`; `tests/test_observables.py` checks they stay inside the slider ranges.
- Dark / light mode, reset to defaults.

`web/app.js` is a line-for-line port of `src/economics_models/core.py`. Parity is enforced by `tests/test_js_parity.py`, which runs the JS under Node.js on identical inputs (including injected Monte Carlo draws) and compares every number.

---

## 2. Python package

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # runtime only
pip install -e .[dev]           # + tests, ruff, mypy
pip install -e .[notebook]      # + jupyter & ipywidgets for the notebook
```

### Quick start

```python
from economics_models import (
    SolowParameters, FinancialParameters, MonteCarloParameters,
    justified_pe, tornado, monte_carlo,
    implied_technology_growth, implied_equity_risk_premium,
    compare_scenarios, scenarios_to_csv, dashboard,
)

solow = SolowParameters(n=0.005, g=0.02)
fin   = FinancialParameters(beta=1.0, equity_risk_premium=0.05,
                            expected_inflation=0.02, term_premium=0.01,
                            retention_rate=0.35, earnings_growth_factor=1.10)

res = justified_pe(solow, fin)
print(f"Justified forward P/E: {res.justified_pe:.2f}   duration: {res.pe_duration:.1f} yrs")

# What does a market P/E of 18 imply, holding everything else fixed?
print(implied_technology_growth(18.0, solow, fin))     # None if unreachable for g in [0, 10%]
print(implied_equity_risk_premium(18.0, solow, fin))   # ERP that reproduces 18x

# Which assumptions matter most?
for row in tornado(solow, fin)[:3]:
    print(row.parameter, round(row.swing, 2))

# Uncertainty, with the invalid share reported instead of silently dropped
mc = monte_carlo(solow, fin, MonteCarloParameters(num_simulations=20_000, seed=42))
print(mc.summary.mean, mc.summary.percentiles[5], mc.summary.percentiles[95], mc.summary.invalid_share)

# Scenarios
table = compare_scenarios({
    "bear": (solow, FinancialParameters(equity_risk_premium=0.07)),
    "base": (solow, fin),
    "bull": (solow, FinancialParameters(equity_risk_premium=0.04)),
})
print(scenarios_to_csv(table))

# Matplotlib dashboard: convergence, P/E sensitivity, tornado, Monte Carlo
dashboard(solow, fin, MonteCarloParameters(seed=42)).savefig("dashboard.png", dpi=120)
```

### Public API

| Symbol | Purpose |
|---|---|
| `SolowParameters`, `FinancialParameters`, `MonteCarloParameters` | Validated dataclass inputs |
| `steady_state_growth_rate(p, method="closed_form")` | Real growth % fed to the valuation (`"simulated"` = final-period path value) |
| `convergence_gap(p)`, `steady_state_output_capital_ratio(p)` | Solow diagnostics |
| `simulate_solow(p) -> SolowPath` | Full K, Y, A, L, growth-rate trajectory |
| `justified_pe(solow, fin) -> ValuationResult` | Valuation pipeline incl. `pe_duration` and `earnings_outgrow_gdp` flag |
| `gdp_sensitivity`, `pe_sensitivity` | Sweeps over technology growth `g` |
| `tornado(solow, fin, shocks=None)` | One-at-a-time ± shocks, sorted by swing |
| `implied_technology_growth`, `implied_equity_risk_premium` | Invert the model for an observed P/E |
| `compare_scenarios`, `scenarios_to_csv` | Named scenarios → results table |
| `monte_carlo(...) -> MonteCarloResult` | Vectorised, seeded, optionally correlated draws + summary |
| `monte_carlo_pe(...)` | Backward-compatible `(pe, eg)` tuple |
| `plot_*`, `dashboard` | Matplotlib renderers |

---

## 3. The math

### Cobb-Douglas production
\[ Y_t = K_t^{\alpha} (A_t L_t)^{1-\alpha} \]

### Capital accumulation (textbook aggregate form)
\[ K_{t+1} = (1-\delta)K_t + s\,Y_t,\qquad L_{t+1} = L_t(1+n),\qquad A_{t+1} = A_t(1+g) \]
Steady-state output/capital ratio: \(Y/K = \big((1+n)(1+g) - (1-\delta)\big)/s\).

### Steady-state growth
Aggregate output grows at the rate of effective labour: **\(g_{real} = (1+n)(1+g) − 1\)**. This closed form is what feeds the valuation. It does **not** depend on \(s, \delta, \alpha, K_0\) or \(T\) — those only shape the convergence path, which the UI shows as a diagnostic.

### Risk-free rate and CAPM
\[ R_f = g_{real} + \pi + TP,\qquad k_e = R_f + \beta \cdot ERP \]

### Justified forward P/E
Single stage (Gordon): \(P/E = (1-b)/(k_e - g_e)\) with \(g_e = g_{nominal}\cdot\text{factor}\).

Two stage (`high_growth_years = N`): earnings grow at \(g_e\) for \(N\) years, then at nominal GDP growth forever:
\[ \frac{P_0}{E_1} = (1-b)\left[\sum_{t=1}^{N}\frac{(1+g_e)^{t-1}}{(1+k_e)^t} + \frac{(1+g_e)^{N-1}(1+g_{nom})}{(k_e-g_{nom})(1+k_e)^N}\right] \]

The model is **invalid** (UI flags it red, Python returns `justified_pe=None`) when the *terminal* growth rate is ≥ \(k_e\).

### P/E duration
\(D = -\frac{\partial (P/E)/\partial k_e}{P/E}\), in years. Single stage: \(D = 1/(k_e - g_e)\).

---

## 4. Model conventions (v2) — and why

These were deliberate choices; each is reversible.

| Question | v1 behaviour | v2 behaviour | Why |
|---|---|---|---|
| What is "steady-state growth"? | Final-period growth of a simulated path with `K0=10000`, `T=200` | Closed form \((1+n)(1+g)-1\) | The simulated number depended on an arbitrary `K0` (changing it to 1e6 moved real growth by 21 bp) and on `T`. The closed form is exact and the simulation is kept as a **convergence diagnostic** (`convergence_gap`, `method="simulated"`). |
| Capital law of motion | \(K_{t+1}=(1+n+g)(K_t+sY_t-\delta K_t)\) | \(K_{t+1}=(1-\delta)K_t+sY_t\) | The v1 form gave a steady-state \(Y/K=\delta/s\) instead of the textbook \((\delta+n+g)/s\). Growth rates were unaffected; levels were not. |
| Can earnings outgrow GDP forever? | Yes, silently | Flagged (`earnings_outgrow_gdp`); bounded by the two-stage option | A perpetual factor > 1 contradicts the steady state the model just computed. |
| Monte Carlo means | Python defaulted `g_mean=0.02` regardless of `solow.g`; JS used the slider | Both default to the point estimates | Removed a Python/JS drift. |
| Invalid Monte Carlo draws | Dropped silently (survivorship bias) | Reported as `invalid_share`; percentiles and standard error included | The bias activates exactly near the \(g_e \ge k_e\) boundary where the model is most interesting. |
| Default `K0` | 10 000 | 200 | Puts the path below steady state so the convergence chart shows a classic catch-up; irrelevant to the valuation. |

---

## 5. Performance

Monte Carlo is fully vectorised in both implementations (one pass over the draws, no per-draw Solow loop). Measured on a single core:

| | v1 | v2 |
|---|---|---|
| Python, 5 000 draws | 1.08 s | ~10 ms |
| Python, 20 000 draws | 4.32 s | ~30 ms |
| JS, 20 000 draws | 406 ms | ~40 ms |

The web UI therefore runs Monte Carlo live with the sliders; the "Run" button is gone. Plotly loads the partial `cartesian` bundle instead of the full build.

---

## 6. Best practices

- **Only `n` and `g` move the valuation through the Solow block.** If the convergence gap is red, `T` is too short or `K0` too far from steady state for the *simulated* path to mean anything — but the valuation is unaffected because it uses the closed form.
- **Watch the terminal-growth constraint.** Gordon explodes near the boundary. Use `high_growth_years` if you want stage-1 growth above nominal GDP.
- **Use the tornado before the Monte Carlo.** At the defaults, ERP, beta, the earnings growth factor and retention dominate; `g` and inflation barely register. Calibrate std-devs where the swing is.
- **Inversion returns `None` when unreachable.** The UI shows the reachable P/E range so you can see *why* (e.g. at the defaults no `g` in [0, 10%] reproduces an 18x multiple, but an ERP of about 3.1% does).
- **This is a teaching/research tool.** Not investment advice.

---

## 7. Development

```bash
pip install -e .[dev]
pytest -q                      # Python tests + JS parity (parity skipped if node is missing)
ruff check src tests && ruff format --check src tests
mypy src
```

CI (`.github/workflows/ci.yml`) runs lint, type-check and the full suite on Python 3.9 and 3.12 with Node 20.

### Contributing

1. Add a test in `tests/test_core.py` for any math change.
2. **Update both `src/economics_models/core.py` and `web/app.js` together.** `tests/test_js_parity.py` will fail if they drift.
3. Run the checks above and open a PR.

## License

MIT.
