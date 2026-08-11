# Project 16 — Risk Parity & Volatility-Targeting Engine (2-person plan)

## Context

Project 16 builds a systematic risk-parity portfolio engine: equal *risk* contribution
(not equal capital) across four asset classes, a volatility-targeting overlay that scales
leverage to a fixed annualized vol target, and stress tests through 2008, 2020 and 2022.
The intellectual payload is the 2022 regime — when stock-bond correlation flipped positive,
the diversification assumption underpinning risk parity broke and the strategy failed. The
write-up must show *why* it broke, not hide it.

Two people build it in parallel on separate machines. The hard part of a 2-person split on a
quant pipeline is that the analysis half normally can't start until the engine half produces
output. This plan removes that dependency with a **frozen result contract + synthetic fixture**
on day one, so both people are unblocked from hour 2 onward.

**Home:** the existing empty private repo `abhyudayb-1/portfolio-allocation-capstone-project`,
already wired to `C:\Users\abhyu\Desktop\Project 15` (local `main` tracks `origin/main`).

**Conventions are inherited from `abhyudayb-1/sp500-factor-backtester`,** which is mature and
sets the house style: `src/` module-per-stage pipeline, `config.py` as the single source of
constants, standalone `python tests/x.py` suites with no framework, extensive authoritative
README, keyless data sources only, `reports/` for generated PNG/CSV.

---

## Ownership split

| | **Person A — Engine** | **Person B — Analysis** |
|---|---|---|
| Owns | `data.py`, `covariance.py`, `riskparity.py`, `overlay.py`, `backtest.py`, `cache.py` | `metrics.py`, `benchmarks.py`, `stress.py`, `viz.py`, `tests/`, `run_backtest.py`, `README.md` |
| Theme | producing weights and returns | judging and explaining them |

`src/config.py` is shared and **append-only** — add constants, never reformat or reorder
existing ones, and say in the PR body which block you touched. This is the only file both
people write, and append-only keeps it conflict-free in practice.

---

## Target layout

```
src/config.py       universe, crisis windows, EWMA halflife, vol target, leverage caps, paths
src/cache.py        PORTED VERBATIM from sp500-factor-backtester — do not rewrite
src/data.py         yfinance ETF closes + keyless FRED series (risk-free, VIX context)
src/covariance.py   EWMA covariance, shrinkage, PSD repair, annualization
src/riskparity.py   risk_contributions(), solve_erc(), naive_inv_vol()
src/overlay.py      trailing vol estimate -> leverage -> caps -> financing cost
src/backtest.py     monthly rebalance loop, daily accrual, weight/leverage/RC history
src/metrics.py      PORTED from sp500-factor-backtester + crisis-window helpers
src/benchmarks.py   equal-weight, 60/40, max-Sharpe (cvxpy), naive 1/vol risk parity
src/stress.py       crisis decomposition, drawdown table, correlation-regime diagnostics
src/viz.py          the five charts
run_backtest.py     entry point wiring it end to end
tests/fixtures.py   synthetic BacktestResult generator — B's unblocking mechanism
tests/test_riskparity.py   ERC math + B's independent convex oracle
tests/test_lookahead.py    proves no rebalance reads future data
```

### Reuse, don't rewrite

- **`src/cache.py`** — copy verbatim. Atomic `.tmp`-then-`replace` writes, `is_fresh()`,
  human-inspectable CSV/JSON. Already exactly what this project needs.
- **`src/metrics.py`** — copy `cagr`, `sharpe_ratio`, `max_drawdown`, `max_drawdown_dates`,
  `_drawdown_series`, `summarize`. One change: this project accrues **daily**, so
  `PERIODS_PER_YEAR` becomes a passed argument (252 for daily, 12 for monthly) rather than a
  module default. Keep the `_ZERO_VARIANCE_EPS` guard and the peak-floored-at-1.0 drawdown fix
  — both are subtle and correct.
- **`src/viz.py`** — reuse `_style_axes`, `_new_figure`, `_title`, `_save` and the palette.
- **`src/prices.py`** — take `month_end_trading_days()` and the `price_asof()` shape; the rest
  is S&P-500-specific and not needed for a 5-ETF universe.

---

## Universe and the 2006 start date

| Sleeve | ETF | Inception |
|---|---|---|
| Equities | SPY | 1993-01 |
| Nominal bonds | TLT | 2002-07 |
| Commodities | DBC | **2006-02** |
| Inflation-linked | TIP | 2003-12 |
| Gold (5th, configurable) | GLD | 2004-11 |

**The plan targets 2005 but the data cannot deliver it.** DBC is the earliest liquid broad
commodity ETF at Feb 2006; GSG and DJP are later still. Start the backtest at the first date
all four sleeves trade (~2006-02-03) and state this plainly in the README rather than
back-filling a proxy silently. All three crisis regimes are still fully covered. Splicing a
pre-2006 commodity index (e.g. FRED/S&P GSCI) is a documented enhancement, not v1.

---

## The math, pinned

**Risk contribution.** For weights `w` and covariance `Σ`, portfolio vol `σ_p = sqrt(wᵀΣw)`,
marginal contribution `MRC = Σw / σ_p`, and `RC_i = w_i · MRC_i`, with `Σ RC_i = σ_p` exactly.
That identity is the first unit test.

**ERC solve** (`scipy.optimize.minimize`, SLSQP, long-only, `Σw = 1`): minimize
`Σ_i (RC_i − σ_p/N)²`. Supply the analytic gradient — SLSQP is materially more reliable with
it, and finite differences on a near-flat objective is where these solvers quietly return
garbage. **Explicitly not** the `1/σ` heuristic; that ships as `naive_inv_vol()` purely as a
comparator, and the write-up shows where it diverges from true ERC (it only coincides when all
correlations are equal).

**The independent oracle (Person B).** The convex reformulation — minimize
`½ wᵀΣw − (1/N) Σ ln(w_i)` then renormalize to sum 1 — has a unique global solution that *is*
the ERC portfolio. B implements this in `tests/test_riskparity.py` and asserts A's SLSQP output
matches to ~1e-6. Two independent derivations of the same quantity, written by two people,
cross-checked automatically. This is the highest-value test in the project.

**Vol targeting.** `L_t = target_vol / σ̂_t`, where `σ̂_t` is the trailing EWMA vol of the
*current* weights' portfolio returns using only data ≤ t. Cap `L_t` to `[0.5, 3.0]`.
**Leverage is financed, not free:** `r_p = L_t·(r_assets − r_f) + r_f`, with `r_f` from FRED
`DTB3`. Omitting the financing leg is the single most common way this strategy is
oversold — including it is a differentiator.

**Benchmarks:** equal-weight (25% each), 60/40 (SPY/TLT), rolling long-only max-Sharpe via
cvxpy, and naive 1/vol risk parity.

**Crisis windows** (in `config.py`): GFC `2007-10-09 → 2009-03-09`, COVID
`2020-02-19 → 2020-03-23`, rate shock `2022-01-03 → 2022-10-12`.

**Inherited invariant, carried over from the sp500 project:** at rebalance date `t`, only data
with timestamp ≤ `t` may be used — covariance, vol estimate, and risk-free rate alike.
`tests/test_lookahead.py` exists to prove it.

---

## Phases

**Phase 0 — joint, ~2h, do this together before splitting.** Invite Person B as a collaborator
on the private repo. Create `.venv` (system Python has no numpy on this machine — the sp500
project's CLAUDE.md records this trap). Scaffold `src/`, port `cache.py` and `metrics.py`, and
**freeze two contracts in one merged PR**: the `BacktestResult` dataclass (dated frames for
`weights`, `risk_contributions`, `leverage`, `returns`, plus a `meta` dict) and the
`solve_erc(cov, **kw) -> np.ndarray` signature. Land `tests/fixtures.py`, which emits a
synthetic `BacktestResult` with plausible random weights and returns. **Nothing else starts
until this PR is on `main`** — it is what makes the next three phases genuinely parallel.

| Phase | Person A | Person B |
|---|---|---|
| 1 (~5h) | `data.py`: yfinance ETF pull through `cache.py`; keyless FRED CSV endpoint (`fredgraph.csv?id=DTB3`) — no API key, consistent with the house rule | `benchmarks.py` + crisis-window helpers in `metrics.py`, built against `fixtures.py` |
| 2 (~6h) | `covariance.py` (EWMA + shrinkage + PSD repair) and `riskparity.py` SLSQP solver with analytic gradient | `tests/test_riskparity.py`: RC identity, convex oracle, degenerate cases (perfect correlation, near-singular Σ, single asset) |
| 3 (~6h) | `overlay.py` vol targeting with financing cost; `backtest.py` monthly loop with daily accrual | `stress.py` crisis decomposition + drawdown table; `viz.py` all five charts against fixtures |
| 4 (~5h) | `tests/test_lookahead.py`; solver diagnostics (convergence failures must be logged and surfaced, never silently swallowed) | `run_backtest.py` CLI; README with the 2022 failure-mode write-up |

**Phase 5 — joint, ~3h.** First real end-to-end run, swap `fixtures.py` for live data in B's
charts, reconcile, then the 2022 analysis together.

---

## The 2022 write-up (the thing that makes this portfolio-grade)

Four artifacts, produced by `stress.py` and `viz.py`, that together tell the story:

1. Rolling 63-day SPY/TLT correlation crossing from negative to **positive** in 2022.
2. ERC weights over the same window: bonds' low trailing vol earns them a *large* weight
   exactly as they become correlated with equities.
3. The leverage series staying elevated into the drawdown — trailing vol is a lagging
   estimator, so the overlay de-levers *after* the damage.
4. Crisis drawdown table: risk parity beats 60/40 in 2008 and 2020, and **loses** in 2022.

State that last result plainly. The conclusion is that risk parity is a bet on the
stock-bond correlation staying negative, and 2022 is what collecting on the wrong side of that
bet looks like.

---

## Git workflow

- Branches `feat/engine-*` (A) and `feat/analysis-*` (B); PR reviewed by the other person.
- File ownership above means the only shared file is `config.py`, kept append-only.
- Merge order matters once per phase: A's module lands before B's test of it, except in
  Phase 2 where B's oracle test can land first and fail until A's solver arrives.
- `.gitignore` (currently uncommitted in the working tree) gets extended in Phase 0 with
  `.venv/`, `__pycache__/`, `data/cache/`, alongside the existing
  `.claude/settings.local.json` line.

---

## Verification

```powershell
.\.venv\Scripts\python.exe tests\test_riskparity.py   # ERC identity + convex oracle agreement
.\.venv\Scripts\python.exe tests\test_lookahead.py    # no rebalance reads future data
.\.venv\Scripts\python.exe run_backtest.py            # full run -> reports/
.\.venv\Scripts\python.exe run_backtest.py --start 2021-01-01 --no-cache-refresh   # fast smoke
```

Done means: `Σ RC_i = σ_p` holds to 1e-10; SLSQP and convex solutions agree to 1e-6; every
rebalance provably uses only past data; the crisis table reproduces the known signs (risk
parity wins 2008/2020, loses 2022); and five charts land in `reports/` — RC stacked area,
capital-vs-risk weight comparison, leverage time series, rolling correlation, and the crisis
drawdown table.
