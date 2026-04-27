# Macro-Financial Valuation Lab

> Solow-Swan growth → CAPM → Gordon Growth → Monte Carlo, in one integrated model.
> Move sliders, watch the **justified forward P/E** update in real time.

This repository contains an integrated macro-financial valuation framework. A Solow-Swan growth model with a Cobb-Douglas production function feeds a steady-state real growth rate into a CAPM-based required return, which a Gordon Growth Model then turns into a justified forward P/E ratio. A Monte Carlo layer quantifies the uncertainty around that estimate.

You can use it three ways:

1. **Open `web/index.html` in a browser** — zero install, fully interactive.
2. **`import economics_models`** — clean, typed Python API for analysts and notebooks.
3. **Open `EconomicGrowth.ipynb`** — the original research notebook (kept for reference).

---

## 1. Web interface (recommended)

Just open the file:

```bash
# from the repo root
xdg-open web/index.html        # Linux
open web/index.html            # macOS
start web\index.html           # Windows
```

Or serve it locally (any static server works):

```bash
python -m http.server 8000 --directory web
# then visit http://localhost:8000
```

Or push to GitHub Pages — the `web/` folder is fully self-contained (only depends on Plotly via CDN).

### What you get

- **Live KPIs** for real growth, nominal growth, risk-free rate, required return, and justified P/E.
- **Solow growth path** chart with the theoretical `(1+n)(1+g) − 1` reference line.
- **Sensitivity sweeps**: GDP growth vs. tech progress, P/E vs. tech progress.
- **Monte Carlo** distributions for P/E and earnings growth.
- **Dark / light mode** that follows your OS preference.
- **Reset to defaults** button.

The JS implementation in `web/app.js` mirrors the Python math in `src/economics_models/core.py` line for line — both produce the same numbers (verified in tests).

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
    steady_state_growth_rate, justified_pe, monte_carlo_pe,
    dashboard,
)

solow = SolowParameters(s=0.20, n=0.005, g=0.02, delta=0.05, alpha=0.35)
fin   = FinancialParameters(beta=1.0, equity_risk_premium=0.05,
                            expected_inflation=0.02, term_premium=0.01,
                            retention_rate=0.35, earnings_growth_factor=1.10)

print(f"Steady-state growth: {steady_state_growth_rate(solow):.3f}%")

result = justified_pe(solow, fin)
print(f"Justified forward P/E: {result.justified_pe:.2f}")
print(f"Required return: {result.required_return:.2%}")

mc = MonteCarloParameters(num_simulations=5_000, seed=42)
pe, eg = monte_carlo_pe(solow, fin, mc)
print(f"P/E mean ± std (MC): {pe.mean():.2f} ± {pe.std():.2f}")

# All-in-one matplotlib dashboard
fig = dashboard(solow, fin, mc)
fig.savefig("dashboard.png", dpi=120)
```

### Public API

| Symbol | Purpose |
|---|---|
| `SolowParameters`, `FinancialParameters`, `MonteCarloParameters` | Validated dataclass inputs |
| `simulate_solow(p) -> SolowPath` | Full K, Y, A, L, growth-rate trajectory |
| `steady_state_growth_rate(p)` | Final-period YoY % growth |
| `justified_pe(solow, fin) -> ValuationResult` | Full valuation pipeline |
| `gdp_sensitivity(solow, ...)` | Sweep `g` → steady-state growth |
| `pe_sensitivity(solow, fin, ...)` | Sweep `g` → justified P/E |
| `monte_carlo_pe(solow, fin, mc)` | Distribution of P/E and earnings growth |
| `plot_*`, `dashboard` | Matplotlib renderers |

---

## 3. The math

### Cobb-Douglas production
\[ Y_t = K_t^{\alpha} (A_t L_t)^{1-\alpha} \]

### Capital accumulation (per the Solow-Swan formulation in this codebase)
\[ K_{t+1} = (1 + n + g)\,(K_t + s\,Y_t - \delta\,K_t) \]
with \(L_{t+1} = L_t(1+n)\) and \(A_{t+1} = A_t(1+g)\).

### Steady-state growth
Output grows at the rate at which effective labor grows: **\((1+n)(1+g) − 1\)** per period (≈ \(n+g\) for small values). This is the *real* growth rate fed into the financial layer.

### Risk-free rate (Fisher + term premium)
\[ R_f = g_{real} + \pi + TP \]

### CAPM required return
\[ k_e = R_f + \beta \cdot ERP \]

### Gordon Growth justified forward P/E
\[ P/E = \frac{1 - b}{k_e - g_e},\quad g_e = g_{nominal} \cdot \text{TFP factor} \]

The model is **invalid** (and the UI flags it) when \(g_e \ge k_e\) — no finite present value exists.

---

## 4. Project layout

```
.
├── web/                       # Static, zero-install browser UI
│   ├── index.html
│   ├── styles.css
│   └── app.js                 # JS port of the Python model
├── src/economics_models/      # Python package
│   ├── __init__.py
│   ├── core.py                # Model math + dataclasses (single source of truth)
│   └── visualization.py       # Matplotlib renderers
├── tests/
│   └── test_core.py           # 12 unit tests, all passing
├── EconomicGrowth.ipynb       # Original research notebook (reference)
├── pyproject.toml             # Project metadata, deps, ruff, pytest config
├── requirements.txt           # Runtime deps
├── requirements-dev.txt       # + dev/test deps
└── README.md
```

---

## 5. Best practices for using the model

- **`(1+n)(1+g) − 1` is the asymptote, not `n + g`.** They agree to 1 bp at typical values; for large `g` the difference grows. Tests use the geometric form.
- **Keep `T ≥ 200`** when you want the steady-state growth rate. Earlier periods are still on the transition path.
- **Watch the `g_e ≥ k_e` constraint.** Gordon explodes (and lies) near the boundary. The UI marks it red; the Python API returns `justified_pe=None`.
- **Calibrate Monte Carlo std-devs honestly.** Tiny `std`s create false precision in the resulting distribution. The defaults (`g_std=0.5%`, `π_std=0.5%`, `TP_std=0.2%`) are a starting point, not gospel.
- **`α` is capital's share of output**, typically 0.30–0.40 for advanced economies.
- **`b` is the retention rate**; payout ratio is `1 − b`.
- **`earnings_growth_factor`** scales nominal GDP growth into earnings growth (operating leverage / margin expansion / TFP). > 1 is bullish, < 1 is conservative.
- **This is a teaching/research tool.** Not investment advice. Treat outputs as conditional on your assumptions.

---

## 6. Development

```bash
pip install -e .[dev]
pytest -q                      # run all tests
ruff check src tests           # lint
ruff format src tests          # format
mypy src                       # type-check
```

CI-friendly. The `tests/` suite is deterministic (Monte Carlo uses a fixed seed).

### Cross-checking JS ↔ Python

The web UI exposes the model on `window.MacroFin`. To verify the JS port matches Python, open the browser console:

```js
const p = MacroFin.defaults;
const path = MacroFin.simulateSolow({...p, T: 200});
const last = path.growth[path.growth.length - 1];
console.log(last);   // ≈ 2.49997, same as Python
```

---

## 7. Contributing

1. Fork & branch (`feat/my-thing`).
2. Add a test in `tests/test_core.py` for any math change.
3. **Update both `src/economics_models/core.py` and `web/app.js` together** — they are intentionally redundant implementations. Drift between them is a bug.
4. Run `pytest -q` and `ruff check`.
5. Open a PR.

## License

MIT. See repository for details.
