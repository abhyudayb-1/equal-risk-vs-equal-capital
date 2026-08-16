# Risk Parity + Volatility Targeting

A systematic portfolio engine over four ETF sleeves — **SPY** (US equities), **TLT** (long
Treasuries), **DBC** (commodities) and **TIP** (inflation-linked bonds) — backtested daily from
February 2006 to the present, through the 2008 financial crisis, the March 2020 COVID crash and
the 2022 rate shock.

The engine does two things a conventional allocator does not:

1. **It allocates risk, not capital.** A 25%-each portfolio is not a balanced portfolio. Equities
   run roughly three times the volatility of TIP, so equal capital means equities dominate the
   portfolio's variance while the pie chart looks perfectly diversified. The engine instead solves
   each month for the weights at which every sleeve contributes an *equal share of portfolio
   volatility* — the **equal risk contribution (ERC)** portfolio.
2. **It then chooses how much risk to run.** ERC decides how risk is *split*; it says nothing about
   *how much*. A bond-heavy risk-parity book left alone runs far less volatility than the strategy
   is sold on, so a **volatility-targeting overlay** levers the whole portfolio toward a 10%
   annualised target, subject to a leverage cap. That leverage is **financed, not free**: the
   borrowed portion is charged at the 3-month T-bill rate, which is the single most common way this
   strategy gets oversold.

The deliverable is not only the engine. The analytical payload is **why risk parity failed in
2022** — and results that make the strategy look bad are stated plainly rather than buried. Both
`weights` and `risk_contributions` are carried through the whole pipeline side by side, because
their divergence *is* the thesis.

---

## Setup

Requires Python 3.10+. Everything is keyless — prices come from yfinance and the risk-free rate
from FRED's public CSV endpoint, so there are no API keys to obtain or configure.

```bash
git clone https://github.com/abhyudayb-1/portfolio-allocation-capstone-project.git
cd portfolio-allocation-capstone-project

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # macOS / Linux
```

On Windows, the interpreter lives elsewhere in the venv:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Every command below is written in the macOS/Linux form. Substitute `.\.venv\Scripts\python.exe`
for `.venv/bin/python` on Windows.

> **Always work inside the venv.** The dependency list includes numpy, scipy, cvxpy and matplotlib;
> a system Python without them will fail at import.

---

## Running it

```bash
.venv/bin/python run_backtest.py
```

That is the whole thing: it downloads prices and the risk-free series, runs the engine and four
benchmarks, prints a full-sample summary, a crisis table and the regime diagnostics, and writes a
CSV plus four charts to `reports/`.

The first run downloads about twenty years of daily data for four tickers and takes a couple of
minutes; the results are cached under `data/cache/`, so subsequent runs are fast and work offline.

| Flag | What it does |
|---|---|
| `--start YYYY-MM-DD` | First date of price history. Defaults to `2006-02-03`, DBC's inception. |
| `--target-vol SIGMA` | Annualised volatility the overlay targets, as a decimal. Default `0.10`. |
| `--max-leverage L` | Upper leverage cap. Default `3.0`; the floor stays at `0.5`. |
| `--refresh-prices` | Bypass the on-disk caches and re-download. |
| `--reports-dir DIR` | Where the table and charts are written. Default `reports/`. |
| `--no-charts` | Skip chart rendering — the slow part — and write the tables only. |
| `--verbose` | Show INFO logging from `src/`: downloads, PSD repairs, skipped windows. |
| `--help` | The above, from the CLI. |

Note that the first rebalance is roughly 252 trading days after `--start`, because the covariance
estimate needs a year of history before it is allowed to decide anything.

### Tests

Standalone scripts — no pytest, no test runner. Run a file to run its suite; each prints
`N checks passed` and exits 0.

```bash
.venv/bin/python tests/test_contract.py        # the BacktestResult contract, metrics, fixtures
.venv/bin/python tests/test_data.py            # windowing, calendar, look-ahead guard
.venv/bin/python tests/test_benchmarks.py      # equal weight / 60-40 / max Sharpe / naive 1-vol
.venv/bin/python tests/test_crisis_slices.py   # crisis-window slicing and summary metrics
.venv/bin/python tests/test_stress.py          # crisis table, diversification ratio, charts
.venv/bin/python tests/test_riskparity.py      # risk-contribution identity, ERC oracle, edge cases
```

Only `test_data.py` and `test_contract.py` touch the network; the suites are expected to pass
offline. Most engine modules also carry a `__main__` self-check — `.venv/bin/python -m src.riskparity`
verifies the analytic gradient against central differences and solves 500 random covariances,
`.venv/bin/python -m src.covariance` checks that an infinite halflife reproduces the sample
covariance and that PSD repair fires on an indefinite input.

---

## Modules

One module per pipeline stage under `src/`, wired end to end by `run_backtest.py`:

```
config.py → data.py → covariance.py → riskparity.py → overlay.py → backtest.py → BacktestResult
                                                                                       ↓
                                                          metrics.py / benchmarks.py / stress.py / viz.py
```

| File | What it does |
|---|---|
| `src/config.py` | Every magic number in the project: the universe, `DATA_START_DATE`, EWMA halflives, `TARGET_VOL`, `LEVERAGE_CAP`, solver tolerances and the three crisis windows. Nothing is hard-coded at a call site. Each constant carries the measurement that set it. |
| `src/cache.py` | On-disk cache for network pulls, in CSV/JSON so it stays human-inspectable. Atomic `.tmp`-then-`replace` writes, so an interrupted run cannot leave a half-written cache behind. |
| `src/types.py` | **The contract.** `BacktestResult` — a frozen dataclass of `weights`, `risk_contributions`, `leverage`, `returns` and `meta`, with a loud `.validate()`. Every strategy in the repo, engine and benchmark alike, returns one of these, so every metric and chart treats them identically. |
| `src/data.py` | Total-return ETF closes from yfinance (`auto_adjust=True` — TLT and TIP pay most of their return as coupons) and the daily risk-free rate from FRED's `DTB3`. Owns `asof_slice()`, the single chokepoint enforcing that date `t` never sees data stamped after `t`. Raises `DataUnavailableError` rather than silently dropping a sleeve. |
| `src/covariance.py` | The covariance pipeline: EWMA (so a regime change reaches the weights in weeks, not a year), then shrinkage toward the diagonal, then PSD repair. The repair works in *correlation* space and renormalises, so the asset volatilities — the well-estimated part — come back unchanged. |
| `src/riskparity.py` | The mathematical core. Risk contributions via the Euler identity `RC_i = w_i·(Cw)_i / σ_p`, and the ERC solver that equalises them. The solver reports `optimiser_succeeded`, `converged` and `restarts_used` separately, because SLSQP reports success on local minima that pin a sleeve at the weight floor with *negative* risk contribution. It restarts to escape them, and rejects a non-PSD covariance rather than quietly repairing it. |
| `src/overlay.py` | The volatility-targeting overlay: trailing EWMA portfolio volatility, `L_t = TARGET_VOL / σ̂_t` clipped to the leverage cap, and the financing leg `r_p = L_t·(r_assets − r_f) + r_f`. Enforces the one-day rule — a decision taken at the close of `t` earns returns from `t+1` onward. |
| `src/backtest.py` | The monthly rebalance loop that ties the above together into a `BacktestResult`. Owns no maths; its job is to walk the calendar in the right order and never look forward. Solver non-convergence is counted into `meta`; a genuinely broken price series raises and kills the run. |
| `src/metrics.py` | CAGR, annualised volatility, Sharpe, drawdown, hit rate, and the crisis-window slicing every table is built on. Ported from a predecessor repo with `periods_per_year` made a required argument, since this project accrues daily rather than monthly. |
| `src/benchmarks.py` | The four unlevered comparators: equal weight, 60/40, naive inverse-volatility ("risk parity" as usually shipped, which ignores correlations) and a rolling long-only max-Sharpe tangency portfolio. All return the same `BacktestResult` the engine does. |
| `src/stress.py` | The crisis decomposition — every strategy across every crisis window, with the capital-weight-in-bonds and mean-leverage columns that make the ranking interpretable — plus the regime diagnostics: rolling SPY/TLT correlation and the diversification ratio `Σwᵢσᵢ / σₚ`. |
| `src/viz.py` | The four report charts: risk contributions through time, capital weight beside risk weight, the leverage multiplier, and correlation against the diversification ratio. Crisis windows are shaded on each. |
| `run_backtest.py` | The CLI. The only place the modules are wired together, and it owns no maths of its own — a wrong number here is an upstream bug. |

Supporting files: `tests/fixtures.py` builds a synthetic `BacktestResult` so the analysis layer can
be developed and tested without a live engine; `PLAN.md` holds the design rationale and the pinned
maths; `TASKS.md` holds the numbered build order.

---

## Method

Three ideas, each of which is a correction to something the obvious approach gets wrong.

### 1. Risk contribution, not capital weight

Split a portfolio's volatility into per-asset shares. For weights `w` and covariance matrix `C`,
portfolio volatility is

```
σ_p = √(wᵀ C w)
```

Asset *i*'s **marginal** contribution is `MRC_i = (Cw)_i / σ_p` — how much σ_p moves when you add a
little of asset *i* — and its **risk contribution** is that scaled by how much you hold:

```
RC_i = w_i · (Cw)_i / σ_p
```

These sum to `σ_p` exactly (Euler's theorem, since σ_p is homogeneous of degree 1 in `w`), so
`RC_i / σ_p` is a genuine share of portfolio risk. The **equal risk contribution** portfolio is the
`w` where all four shares equal 1/4.

Why this differs from equal capital, concretely: over this universe, 25% capital in SPY typically
accounts for well over half the portfolio's variance, while 25% in TIP accounts for a few percent.
Equalising risk instead means holding *more* of the low-volatility sleeves and less of the
high-volatility ones — the resulting book is heavily weighted toward bonds by capital. That is the
mechanism behind both risk parity's long-run record and its 2022 problem, which is why this repo
keeps `weights` and `risk_contributions` side by side everywhere rather than deriving one on
demand.

The solve itself is a constrained least-squares problem, minimising `Σ(RC_i − σ_p/N)²` subject to
weights summing to 1 and staying non-negative. That objective is **not convex**, and it has local
minima — they appear exactly when two sleeves are strongly negatively correlated, which is the
SPY/TLT pair. The solver restarts from multiple initial points to escape them, and an independent
convex log-barrier formulation was built to cross-check the answers (they agree to ~7e-08).

### 2. EWMA covariance

The covariance has to come from somewhere, and a plain sample covariance over a long window
responds to a regime change only slowly, while a short window is mostly noise. This engine uses an
**exponentially-weighted** covariance: observation *k* days ago carries weight `0.5^(k/halflife)`,
with a 60-day halflife. Recent history dominates, but nothing is discarded outright.

Two corrections follow the estimate. **Shrinkage** pulls the off-diagonal entries 10% toward zero,
because estimation error concentrates in the correlations rather than the variances. **PSD repair**
then guarantees the matrix is a valid covariance at all — an EWMA estimate can come back with a
tiny negative eigenvalue, which would make `√(wᵀCw)` meaningless. The repair clips those
eigenvalues in *correlation* space and renormalises, so the individual asset volatilities survive
untouched and the adjustment is absorbed by the correlation structure.

### 3. The volatility-targeting overlay

An ERC book over these four sleeves runs roughly 5–7% annualised volatility — safe, and far below
what an equity investor is comparing it against. The overlay scales the whole portfolio:

```
L_t = TARGET_VOL / σ̂_t,     clipped to [0.5, 3.0]
```

where `σ̂_t` is the trailing EWMA volatility of the current weights' portfolio returns, using only
data up to and including `t`. The vol halflife (42 days) is deliberately shorter than the
covariance halflife (60 days) — the overlay should react faster than the weights do.

Leverage is **financed**:

```
r_p = L_t · (r_assets − r_f) + r_f
```

The borrowed portion costs the 3-month bill rate. Dropping that term is the standard way a levered
backtest flatters itself; over this sample the financing leg is worth about a percentage point a
year. Transaction costs and a borrow spread over the bill rate are *not* modelled, so the levered
results here remain somewhat optimistic — stated here rather than left to be discovered.

### The invariant underneath all of it

At every rebalance date `t`, only data timestamped `≤ t` may be used — covariance, volatility
estimate and risk-free rate alike — and a decision taken at the close of `t` earns returns from
`t+1` onward, never `t` itself. `data.asof_slice()` is the single chokepoint for the first rule, and every estimator is routed
through it rather than re-deriving the guard at each call site. A dedicated suite that feeds the
engine a deliberate future spike and proves it stays invisible at `t` — `tests/test_lookahead.py` —
lands in a later step. This is the invariant every result in the next section rests on; without it
a backtest can produce a beautiful equity curve that means nothing.

---

## Results

*Coming in Phase 5 — populated from the live run.*

---

## The 2022 Failure Mode

*Coming in Phase 5 — populated from the live run.*

---

## Data and limitations

- **Universe:** SPY, TLT, DBC, TIP. Daily total-return closes from yfinance.
- **Start date:** 2006-02-03, DBC's inception. The brief asked for 2005; the data cannot deliver it
  and no proxy is back-filled in its place.
- **Risk-free rate:** FRED `DTB3`, the 3-month T-bill, de-compounded to a per-trading-day decimal.
- **Rebalancing:** monthly, at month end.
- **Not modelled:** transaction costs, bid-ask spread, and any borrow spread over the bill rate.
  Monthly rebalancing of four liquid ETFs keeps turnover low, so these would not change the shape
  of the conclusions — but they would move the levels.

## Repository layout

```
run_backtest.py     the CLI — the only place the pipeline is wired together
src/                one module per pipeline stage (table above)
tests/              standalone test scripts + the synthetic fixture
reports/            generated CSV and charts (gitignored until Phase 5)
data/cache/         cached network pulls (gitignored; delete a file to force a re-pull)
PLAN.md             design rationale and the pinned maths
TASKS.md            numbered build order, one owner per step
```
