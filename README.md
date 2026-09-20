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
| `--start YYYY-MM-DD` | Earliest date price history is requested from. Defaults to `2006-02-03`, DBC's inception; the first close the sleeves share is `2006-02-06` and the first daily return is `2006-02-07`. |
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
.venv/bin/python tests/test_lookahead.py       # the <= t guard, including removing it to prove the check has teeth
.venv/bin/python tests/test_incremental.py     # Phase 6: perturbation scan, compensation test, 5-asset limit, realized risk
.venv/bin/python tests/test_compare.py         # head-to-head vs a ticker: relative metrics, tail block, crisis windows
```

No suite touches the network: all nine pass offline, checked by blocking `socket.connect` and
re-running each one, and yfinance is monkeypatched wherever a fetch is
exercised. Most engine modules also carry a `__main__` self-check — `.venv/bin/python -m src.riskparity`
verifies the analytic gradient against central differences and solves 500 random covariances,
`.venv/bin/python -m src.covariance` checks that an infinite halflife reproduces the sample
covariance and that PSD repair fires on an indefinite input. `.venv/bin/python -m src.incremental`
is the one exception to "self-checks run offline": it demonstrates the live-ticker path
`analyze_position.py` needs, so it reaches the network on purpose, once.
`.venv/bin/python -m src.compare` prints a full head-to-head report from fixtures, offline, and
writes nothing to disk.

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

An ERC book over these four sleeves runs at whatever volatility the sleeves happen to offer, which
for a bond-heavy allocation is well below what an equity investor is comparing it against. The
overlay scales the whole portfolio to a chosen level instead:

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
through it rather than re-deriving the guard at each call site. A dedicated suite feeds the engine a
deliberate future spike and proves it stays invisible at `t` — `tests/test_lookahead.py`, which also
removes the guard and asserts that the spike then *does* leak, so the test cannot pass vacuously.
This is the invariant every result in the next section rests on; without it a backtest can produce
a beautiful equity curve that means nothing.

---

## Results

*Every figure below comes from `python run_backtest.py` on live data; the four charts referenced
are committed under `reports/` so they can be read without running anything.*

Live run, 2007-02-28 → 2026-07-31: **234 monthly rebalances, 4897 return days, 0 solver failures,
0 covariances repaired**, worst risk-contribution deviation 1.6e-07. The engine hits its own
specification — every rebalance lands within 2e-07 of equal risk contribution. That is what the
dead-flat 25% bands in `reports/risk_contributions.png` are showing, and
`reports/capital_vs_risk_weights.png` puts the thesis in one frame: the capital weights swing
between roughly 10% and 55% while the risk weights never move off a quarter each. TIP averages
**42% of the money** to earn its 25% of the risk; SPY averages **20% of the money** for the same
25%. That gap is the entire argument for allocating risk rather than capital.

Whether that is *worth doing* is a different question, and the table answers it unkindly.

| Strategy | Total return | CAGR | Volatility | Return/vol | Max drawdown | Mean leverage |
|---|---|---|---|---|---|---|
| **Risk parity + vol target** | 280.6% | **7.12%** | 11.26% | **0.667** | −29.10% | 1.76x |
| Equal weight | 195.4% | 5.73% | 8.39% | 0.707 | −28.09% | 1.00x |
| 60/40 | 407.3% | **8.72%** | 11.47% | **0.786** | −29.92% | 1.00x |
| Naive 1/vol | 151.6% | 4.86% | 6.66% | 0.746 | **−16.69%** | 1.00x |
| Max Sharpe | 252.8% | 6.70% | 10.52% | 0.670 | −26.59% | 1.00x |

*Ratios are computed against a zero risk-free rate — a return-to-volatility ratio, applied
identically to all five rows, so the ranking is fair even though the level is not a true Sharpe.*

**The headline result is that the engine loses.** Over this sample the risk-parity book has the
**lowest risk-adjusted return of the five strategies**, and plain 60/40 beats it on return
(8.72% vs 7.12%) *and* on return-per-unit-risk (0.786 vs 0.667) while running the same volatility
and needing no leverage, no solver and no financing. If the question is "should you have run this
instead of 60/40 since 2007", the measured answer is no.

Two things drive that, and they are worth separating because only one of them is the strategy's
fault.

**Leverage is what costs it, once you charge for it.** The same book at three settings:

| | CAGR | Volatility | Return/vol | Max drawdown |
|---|---|---|---|---|
| Unlevered ERC | 5.10% | 6.92% | **0.754** | −17.15% |
| Levered to 10% target, financing charged | 7.12% | 11.26% | **0.667** | −29.10% |
| Levered to 10% target, financing *ignored* | 8.26% | 11.26% | **0.761** | −29.05% |

Read the third row against the first: with free money, leverage leaves risk-adjusted return
essentially untouched (0.754 → 0.761), which is exactly the claim the strategy is usually sold on.
Charge the T-bill and it drops to 0.667. The financing leg is worth **1.06%/yr** here, and it is
the whole difference between "leverage is free scaling" and "leverage is a cost". A backtest that
omits that term reports the third row and calls it the second.

Also note the unlevered book has by far the best drawdown of the three (−17.15%). Most of what the
overlay bought was volatility, not return.

**The worst drawdown is not 2022 — it is COVID, and the strategy caused it.**

| Crisis window | Risk parity | Equal weight | 60/40 | Naive 1/vol | Max Sharpe |
|---|---|---|---|---|---|
| GFC (2008) | −25.43% | −28.09% | −29.92% | −16.11% | −20.08% |
| **COVID (2020)** | **−29.10%** | −16.53% | −17.76% | −13.52% | −12.55% |
| Rate shock (2022) | −17.72% | −16.33% | −25.83% | −16.34% | −23.19% |

The −29.10% COVID drawdown (peak 2020-02-20, trough 2020-03-18) is the deepest in the whole sample,
and risk parity is the **worst** of the five strategies in that window by a margin of 11 percentage
points. The unlevered ERC book fell far less. The gap is leverage: the strategy entered COVID at
**2.33x, its highest pre-crisis leverage of the entire run** (GFC 1.83x, 2022 1.74x), because 2019
was calm and a trailing volatility estimator has nothing to say about a shock that has not happened
yet. The vol target levers hardest into quiet regimes, and the fastest crash in the sample followed
the quietest year in the sample.

That is not a bug in the overlay — it is the overlay working as specified. No causal filter can
de-lever before the volatility it is measuring has been realised. Vol targeting is a risk
*control*, not a risk *forecast*, and this is what that distinction costs.
`reports/leverage.png` shows the shape plainly: the line peaks just before each crisis band and
bottoms out inside or after it. In COVID it falls from 2.33x to 0.64x — the de-levering lands at
the March 2020 low, selling risk at the bottom rather than before the fall.

Risk parity does win the two slow crises. In 2008 and 2022 the drawdown arrives gradually enough
that the trailing estimator can act, and the book beats 60/40 by 4.5 and 8.1 points respectively.
The strategy is a good defence against a grinding decline and a poor one against a cliff.

---

## The 2022 Failure Mode

2022 is the more interesting failure, because it is the one where *diversification itself* broke
rather than where the overlay mistimed. Risk parity still beat 60/40 that year (−17.72% vs
−25.83%), so the interest is in the mechanism, not the ranking.

**The obvious explanation is wrong, and this repo checked.** The story usually told is that the
stock/bond correlation flipped positive and destroyed the hedge. Measured over the drawdown window
`2022-01-03 → 2022-10-12`, the rolling 63-day SPY/TLT correlation **averaged −0.09 and was positive
on only 31% of days**. It was still negative. The persistent flip is a 2023-onward phenomenon
(2023 mean **+0.14**, positive on 66% of days) — it shows up *after* the damage, which makes it
useless as a warning and wrong as an explanation.

**What actually happened is that the hedge lost more than the thing it hedged.** Peak-to-trough
over that window:

| SPY | TLT | DBC | TIP |
|---|---|---|---|
| −24.5% | **−29.3%** | +21.4% | −12.7% |

TLT is the sleeve that exists to rally when SPY falls. It fell *harder than SPY did*. Correlation
never had to turn positive to break the portfolio — it only had to rise toward zero, which was
enough to remove the offset while both legs fell together under a common driver, the repricing of
real rates. A hedge that merely stops helping is survivable; a hedge that loses 29% while the asset
it protects loses 24% is a second position in the same trade.

This is what makes 2022 a genuine failure of risk parity rather than of leverage. ERC leans on
bonds *by construction* — TLT and TIP are the low-volatility sleeves, so equal risk contribution
means an outsized share of capital in duration. At the 2022 rebalances the book held **23.0% of
capital in TLT** carrying its mandated 25.0% of risk, plus a large TIP position; the risk model,
looking at trailing covariance, still scored duration as a diversifier right up to and through the
event. And TIP fell 12.7% too — inflation-linked bonds are still bonds when the shock *is* the
discount rate.

**The diagnostic that did fire in time is the diversification ratio**, `Σwᵢσᵢ / σₚ` — how much the
portfolio's volatility is reduced relative to holding its sleeves standalone. 1.0 means no
diversification at all. Calendar-year means:

| 2008 | 2012 | 2020 | 2021 | **2022** | 2023 |
|---|---|---|---|---|---|
| 2.111 | 2.179 | 1.790 | 1.678 | **1.586** | 1.521 |

It falls for three consecutive years and is already at its lowest level of the sample **during**
2022, while the correlation-sign test is still reading negative and reassuring. That is the
practical lesson: correlation is the intuitive diagnostic and it is a lagging one, because a sign
test discards magnitude. The ratio prices in the fact that both legs got more volatile and stopped
offsetting, and it degrades while the damage is being done rather than the year after. Both panels
are plotted together in `reports/correlation_diversification.png` for exactly this comparison.

**What the project concludes.** Risk parity is not broken, but it is oversold in two specific ways
this backtest can measure. It is sold on a risk-adjusted return that mostly disappears once the
leverage it requires is financed at a real rate. And it is sold as diversification, which is a
claim about a covariance structure that is estimated from the past and is not stable — 2022 is what
it looks like when the estimate is still describing the previous regime. The engine did its job at
every rebalance to seven decimal places. That was never the part at risk.

---

## Data and limitations

- **Universe:** SPY, TLT, DBC, TIP. Daily total-return closes from yfinance.
- **Start date:** history is requested from 2006-02-03, DBC's inception. The first close the four
  sleeves share is 2006-02-06, so the first daily *return* is 2006-02-07 — the opening close is
  consumed computing it. The brief asked for 2005; the data cannot deliver it and no proxy is
  back-filled in its place. The first *rebalance* is a further ~252 trading days on, once the
  covariance has its warm-up.
- **Risk-free rate:** FRED `DTB3`, the 3-month T-bill, de-compounded to a per-trading-day decimal.
- **Rebalancing:** monthly, at month end.
- **Not modelled:** transaction costs, bid-ask spread, and any borrow spread over the bill rate.
  Monthly rebalancing of four liquid ETFs keeps turnover low, so these would not change the shape
  of the conclusions — but they would move the levels.

---

## Incremental Position Analysis

The engine above answers "how would this strategy have performed since 2006." A different,
narrower question comes up constantly in practice: **given the risk-parity book we already hold,
what happens if we add one more stock to it?** A single stock is not a risk-parity sleeve — it
carries idiosyncratic and sector risk with no independent structural risk premium the way an asset
class does, so it cannot compete against SPY/TLT/DBC/TIP for a slot in the equal-risk-contribution
book. But "what does adding this position do to the book we have" is exactly the incremental risk
analysis a trading desk runs whenever someone proposes a new position, and it is well posed for any
ticker. `analyze_position.py` answers it.

This is new, additive scope layered on top of the engine above — **nothing in `src/types.py`,
`backtest.py`, `overlay.py`, `benchmarks.py`, `stress.py`, `viz.py` or `run_backtest.py` changes for
it.** `src/incremental.py` is a second, independent consumer of `covariance.py` and `riskparity.py`,
the same way `stress.py` is an independent consumer of `BacktestResult`. See `PLAN.md`'s "Phase 6"
section for the full derivation.

### Running it

```bash
.venv/bin/python analyze_position.py --ticker AAPL
```

| Flag | What it does |
|---|---|
| `--ticker TICKER` | The candidate stock, e.g. `AAPL`. Required. Any ticker yfinance covers. |
| `--lookback-days N` | Target trailing window in trading days, ending today. Default `756` (~3 years). |
| `--weights W1,W2,...` | Comma-separated perturbation weights, each strictly between 0 and 1. Default `0.01,0.05,0.10`. |
| `--refresh-prices` | Bypass the on-disk price cache and re-download. |
| `--verbose` | Show INFO logging — downloads, the minimum-history retry, PSD repairs. |
| `--help` | The above, from the CLI. |

### Why a snapshot, not a second backtest

Replaying the four-sleeve approach — a 2006-to-present monthly rebalance with a fifth sleeve —
would fail for most tickers by construction: plenty of names investors actually ask about did not
trade in 2006, and splicing a stock's return history onto years before it existed is not a real
portfolio. Instead this estimates on a **trailing window ending today**
(`INCREMENTAL_LOOKBACK_DAYS`, ~3 years) through the same `covariance.estimate_covariance` the core
engine already uses — no new estimator, no new statistical assumption. The trade this makes
explicit: any ticker with enough recent history works, at the cost of not being a 20-year backtest.
A candidate younger than the full lookback but with at least `INCREMENTAL_MIN_HISTORY_DAYS`
(~1 year) of history is still analysed, on a shorter window rather than being rejected outright; a
name with less than that raises rather than silently estimating a covariance from too few points.

### What it computes

Three pieces of model-based math, all built from primitives `covariance.py` and `riskparity.py`
already validate — no new solver, no new risk model — plus a fourth that replays the window.

1. **Perturbation scan.** Scale the existing four-sleeve ERC weights by `(1 − w)` and add the
   candidate at `w`, for `w` in `0.01`, `0.05`, `0.10`. Read the candidate's own risk contribution
   and the portfolio volatility at each. A perturbation, not a re-optimisation: "what if I just add
   this much," which is how a position actually gets sized.
2. **The unconstrained 5-asset re-solve.** Re-run the ERC solver with the candidate free to compete
   for an equal share of risk, and diff the result against the four-sleeve baseline. A different
   question from (1): not "what if I add 5%" but "what would equal risk contribution actually
   assign this stock, and how much does it reshuffle the other four."
3. **The compensation test.** A known result (Treynor–Black / Elton–Gruber active-portfolio
   theory), not invented here: adding asset *i* at an infinitesimal weight improves portfolio *p*'s
   Sharpe ratio if and only if

   ```
   SR_i > SR_p · ρ(i, p)
   ```

   where `SR_i` is the candidate's own Sharpe ratio, `SR_p` is the baseline portfolio's, and
   `ρ(i, p)` is the candidate's correlation to the *portfolio's* returns — not to any one sleeve.
   This is a **marginal** test: it says whether increasing the candidate's weight from zero helps,
   not what the optimal weight is, which is why it ships alongside the perturbation scan rather
   than instead of it. A stock with a mediocre or even negative own Sharpe can still pass if its
   correlation to the book is low or negative; a stock with an excellent own Sharpe can still fail
   if it is highly correlated with the book. The candidate's correlation to each of the four
   sleeves individually is reported too, since that decomposes *why* the portfolio-level
   correlation came out where it did.
4. **Realized risk, before vs after.** The three pieces above read risk off a covariance matrix.
   This one replays the same trailing window: the four-sleeve baseline book, the book with the
   candidate mixed in at each perturbation weight, and the candidate alone, each as a
   constant-weight daily return series. Each gets the realized risk metrics the covariance view
   cannot give — max drawdown (with its peak and trough dates) and the longest time spent below a
   prior peak, historical VaR and CVaR at 95%, worst day, downside deviation, Sortino, Calmar, skew
   and excess kurtosis — plus CAGR and Sharpe for context, and a second table of each mix's change
   against the baseline. It is a historical simulation of *today's* weights on the window's own
   history: in-sample by construction, unlevered, cost-free. It describes the risk this mix has
   realized, not what a strategy running the rule through time would have earned.

### Worked example

`.venv/bin/python analyze_position.py --ticker AAPL`, run against live data:

```
window: 2023-09-14 -> 2026-09-18 (756 obs)

Baseline (existing book)          AAPL added, 5-asset re-solve
  SPY  0.1660                       AAPL  0.0736
  TLT  0.2259                       SPY   0.1419   (-0.0241 vs baseline)
  DBC  0.1549                       TLT   0.2051   (-0.0207 vs baseline)
  TIP  0.4532                       DBC   0.1522   (-0.0027 vs baseline)
                                     TIP   0.4272   (-0.0260 vs baseline)

Perturbation scan
  candidate weight   portfolio vol   AAPL risk-contribution fraction
  1%                 4.59%           0.9%
  5%                 4.70%           10.9%
  10%                5.17%           31.8%

Compensation test:  SR_i = +0.97   SR_p = +1.14   ρ(i,p) = +0.35   threshold = +0.40
  PASSES -- 0.97 > 0.40

Realized risk, before vs after (replay of this window, unlevered)
                       baseline     +1%     +5%    +10%  stock alone
  annual volatility       6.34%   6.37%   6.61%   7.11%       26.78%
  max drawdown           -5.17%  -5.43%  -6.47%  -7.75%      -33.36%
  VaR 95% (daily)        -0.62%  -0.61%  -0.61%  -0.63%       -2.50%
  CVaR 95% (daily)       -0.94%  -0.94%  -0.96%  -1.00%       -3.82%
  worst day              -2.08%  -2.10%  -2.19%  -2.30%       -9.25%
  Sortino                 1.654   1.692   1.815   1.907        1.451
  Calmar                  1.407   1.376   1.277   1.192        0.751
```

Read together: the solver would assign AAPL 7.4% of the book at full equilibrium — a real, not
token, allocation — reshuffling every existing sleeve down to make room, most of it out of TIP and
SPY. The compensation test passes: AAPL's own risk-adjusted return (0.97) is comfortably above the
0.40 bar its ~0.35 correlation to the book sets, so adding a small position would improve the
portfolio's risk-adjusted return at current estimates. Note what "comfortably passes" is not
saying — see the caveats immediately below.

The replay adds what the covariance view cannot say. A 10% AAPL position deepens the realized max
drawdown from −5.2% to −7.8% and lifts volatility from 6.3% to 7.1%, yet leaves the daily 95% VaR
almost untouched (−0.62% → −0.63%): the stock's cost shows up in how far the book falls along the
path, not in how bad an ordinary bad day is. Sortino improves (1.65 → 1.91) because return rose
faster than downside deviation, while Calmar falls (1.41 → 1.19) because the drawdown deepened
faster than CAGR. The replay's volatility also runs above the perturbation scan's (6.4% against
4.6% at a 1% position): the scan uses the EWMA covariance, which weights recent, calmer months
heavily, while the replay weights the whole three years equally — two different estimates of risk,
not one of them wrong.

### What this is, and is not

Printed at the end of every report, not just here, because a CLI user should see it every run, not
once in a document:

1. **Point-in-time, not historical.** This is a snapshot on the trailing window above, as of the
   run date. It is not a claim about how the position would have performed since 2006, and a
   re-run next month will read differently as the window moves — this is the deliberate trade
   against a full historical backtest (see "Why a snapshot" above).
2. **A marginal test, not a sizing recommendation.** The compensation test says whether the
   *direction* of adding the candidate helps, not what weight to add it at — read it alongside the
   perturbation scan, which is for that question.
3. **Still a single stock.** Passing the compensation test does not manufacture a structural risk
   premium the way holding an asset class does. It answers whether this improves *this*
   portfolio's risk-adjusted return given the correlation structure right now — a narrower and
   more honest question than "should I own this stock."
4. **The realized-risk table is an in-sample replay.** Today's weights, held constant and
   rebalanced daily, applied to the trailing window's own history — unlevered and cost-free, so
   its levels sit below the 10%-vol-target strategy's. A window this short rarely contains a
   crisis, and the VaR and CVaR rest on few observations (a 5% tail is 5% of the observation
   count printed above the table).

---

## Comparing Against a Ticker

`analyze_position.py` asks what *adding* a stock does to the book. A simpler question comes up
first: **how did the risk-parity strategy do against just buying and holding this ticker, and what
was the difference in risk?** `compare_ticker.py` answers it, for any ticker yfinance covers —
including the four sleeves themselves, so `--ticker SPY` is the classic "risk parity versus the
S&P 500."

```bash
.venv/bin/python compare_ticker.py --ticker AAPL
```

It runs the project's real engine over its full history — exactly as `run_backtest.py` does — buys
and holds the ticker (total return, dividends reinvested), cuts both to the dates the ticker exists,
and prints three columns side by side:

| Column | What it is |
|---|---|
| `Risk parity + vol target` | The engine's daily returns: levered to the 10% vol target and **net of financing** at `DTB3`. The headline strategy. |
| `Risk parity (unlevered)` | The same equal-risk book with no leverage. The ticker is unlevered, so this is the like-for-like risk comparison. |
| the ticker | Buy-and-hold total return. |

The strategy is run over its whole history *first* and cut to the overlap window afterwards, so its
weights and leverage on any date never depend on when the ticker started trading. The report has
five parts: the risk metrics for all three columns (volatility, max drawdown with its dates and the
longest time underwater, VaR and CVaR at 95%, worst day, downside deviation, Sortino, Calmar, skew,
excess kurtosis, plus CAGR and Sharpe); the strategy's **relative** behaviour (correlation, beta,
annualised tracking error, up and down capture); a **tail block** — what the strategy did on the
ticker's worst 5% of days, and on how many of them it beat the ticker; the three crisis windows,
measured from the peak close and marked `n/a` where the ticker had not yet listed; and the caveats.

Flags: `--ticker` (required), `--refresh-prices` (bypass the caches), `--verbose`, `--help`. It uses
the project's configured vol target and leverage cap, and says so in the report. With prices
cached, a run takes a few seconds — it re-runs the whole engine each time; nothing caches the
strategy between invocations.

### Why the ticker has its own fetch

`data.fetch_prices` cannot serve this. Its late-start guard rejects any ticker that listed more than
ten days after 2006 — correct for the four sleeves, wrong for a stock that IPO'd in 2019 — and it
writes the shared price cache as a single frame, so passing a candidate alongside the universe would
rewrite the cached SPY/TLT/DBC/TIP history too. `data.fetch_ticker_prices` fetches the ticker alone
into **its own** cache file (`data/cache/ticker_<TICKER>.csv`) and validates it against its own first
date, so an empty pull, a non-positive price or a long interior gap still raises. The shared
`prices_close.csv` is never touched — unlike `analyze_position.py`, which can shrink it (the
documented, self-healing trade-off).

### Worked example

`.venv/bin/python compare_ticker.py --ticker AAPL`, run against live data, overlap window
2007-03-01 → 2026-09-18 (4,920 observations):

```
                        Risk parity + vol target   Risk parity (unlevered)     AAPL
annual volatility                        11.25%                     6.91%   31.17%
max drawdown                            -29.10%                   -17.15%  -60.87%
VaR 95% (daily)                          -1.04%                    -0.65%   -2.92%
CVaR 95% (daily)                         -1.65%                    -1.02%   -4.51%
worst day                                -6.62%                    -3.04%  -17.92%
Sortino                                   0.925                     1.079    1.405
Calmar                                    0.244                     0.297    0.468
CAGR                                      7.10%                     5.10%   28.46%
Sharpe                                    0.666                     0.754    0.960

Relative to AAPL:  correlation 0.309   beta 0.111   tracking error 29.70%   up capture 13%   down capture 11%

On AAPL's worst 5% of days (246 days, AAPL averaged -4.51%): the strategy averaged -0.50%
and beat AAPL on 100% of them.

Crisis windows        strategy return   AAPL return
GFC (2008)                     -7.15%       -50.49%
COVID (2020)                  -19.41%       -30.67%
Rate shock (2022)             -15.00%       -23.68%
```

Read honestly, this is a risk story and not a return story. Risk parity was about a third as volatile,
drew down half as far (−29% against −61%), lost less in every crisis window, and beat AAPL on every
one of AAPL's worst days — and earned 7.1% a year against AAPL's 28.5%, with a *lower* Sharpe
(0.67 against 0.96). That last comparison is the hindsight caveat below at work: AAPL is a stock
someone asks about *because* it did well. Against `--ticker SPY` the picture is closer and less
flattering to the ticker: the strategy earned 7.1% against 11.1%, but with a max drawdown of −29%
against −55%, a Sharpe of 0.67 against 0.63 and a Calmar of 0.24 against 0.20.

### What this is, and is not

Printed at the end of every report:

1. **Overlap window only.** Both sides are measured on the dates the ticker exists, not over the
   strategy's full 2006-to-today history, so the numbers will not match `run_backtest.py`'s headline
   results.
2. **Not like-for-like on leverage.** The strategy is levered and charged financing; the ticker is
   not levered at all. The unlevered column and the scale-free ratios (Sharpe, Sortino, Calmar) are
   the fair read.
3. **Chosen with hindsight.** A ticker someone asks about today is, more often than not, one that
   did well or is well known, so it flatters the ticker against a strategy that never got to pick it.
4. **No costs.** No transaction costs, bid-ask spread or borrow spread over the bill rate, on either
   side; levered results in particular are still optimistic.
5. **Short windows are noisy.** A 5% tail is 5% of the observation count, and a max drawdown is a
   single episode. Below `COMPARE_MIN_OVERLAP_DAYS` (252) the comparison refuses to print at all.

**Which tickers work.** Any ticker yfinance has daily prices for, with at least 252 overlapping
trading days. US-listed stocks, ETFs, indices (`^GSPC`) and US-traded ADRs of foreign companies
(`TM` for Toyota) all match the strategy's calendar exactly. Two tiers guard the calendar, both
measured on live tickers rather than assumed:

| Calendar match | What happens | Measured examples |
|---|---|---|
| 99% or better | Runs cleanly | US listings and ADRs: 100% |
| 95% to 99% | **Runs, with a printed warning** | Toronto 98.1–98.2%, London 97.8%, Zurich 97.5%, Hong Kong 97.5%, Frankfurt 97.4%, Sydney 97.4%, Tokyo 96.6% |
| under 95% | Refused (`DataUnavailableError`) | Bitcoin (trades weekends): 69% |

The warning exists because a foreign weekday exchange is usable but not clean. On the ~2–3% of days
one market was shut and the other open, a multi-day return is paired with a one-day one; and a
market that closes at a different time of day reacts to US news on a different date, which pulls
correlation, beta, capture and the tail block toward zero (the non-synchronous-trading effect).
Those numbers are printed with a `! calendar warning` and should be read as indicative, not exact.
The volatility, drawdown and return figures are unaffected by the pairing.

---

## Repository layout

```
run_backtest.py       the core-engine CLI — the only place that pipeline is wired together
analyze_position.py   the incremental-analysis CLI (Phase 6) — see above
compare_ticker.py     the head-to-head CLI (Phase 7): strategy vs buy-and-hold of any ticker
src/                  one module per pipeline stage (table above), plus incremental.py (Phase 6)
                      and compare.py (Phase 7)
tests/                standalone test scripts + the synthetic fixture
reports/              generated CSV and charts (the committed set is from a live run)
data/cache/           cached network pulls (gitignored; delete a file to force a re-pull)
PLAN.md               design rationale and the pinned maths
TASKS.md              numbered build order, one owner per step
```
