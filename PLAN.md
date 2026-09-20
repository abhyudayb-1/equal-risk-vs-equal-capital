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
src/viz.py          the report charts: four PNGs (the crisis table is the fifth deliverable, a CSV from stress.py)
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
| 3 (~6h) | `overlay.py` vol targeting with financing cost; `backtest.py` monthly loop with daily accrual | `stress.py` crisis decomposition + drawdown table; `viz.py` the report charts against fixtures |
| 4 (~5h) | `tests/test_lookahead.py`; solver diagnostics (convergence failures must be logged and surfaced, never silently swallowed) | `run_backtest.py` CLI; README with the 2022 failure-mode write-up |

**Phase 5 — joint, ~3h.** First real end-to-end run, swap `fixtures.py` for live data in B's
charts, reconcile, then the 2022 analysis together.

---

## The 2022 write-up (the thing that makes this portfolio-grade)

**This section was rewritten after step 10, when live data refuted the original thesis.**
The plan said risk parity broke in 2022 "because stock-bond correlation turned positive."
It did not. Measured on real data across `2022-01-03 → 2022-10-12`, the rolling 63-day
SPY/TLT correlation **averaged −0.09 and was positive on only 31% of days**. The persistent
flip is a *2023-onward* phenomenon — 66%, 65% and 70% of days positive in 2023, 2024, 2025.

The true mechanism is subtler and a better finding, because it means risk parity does not
need positive correlation to fail:

Sleeve returns are **peak-to-trough**: `price(end)/price(start) − 1`, i.e. the returns stamped
*after* the window's opening close, since the start date is the peak itself. The correlation
column is the **pairwise correlation of daily returns inside the window** — not the same
statistic as the mean of the rolling 63-day series quoted below, and the two differ most in
exactly the row that matters (2022: **+0.03** pairwise, **−0.09** as a rolling mean). Both are
reported because the gap between them *is* the subtlety.

| Window | SPY/TLT corr (in-window pairwise) | SPY | TLT | DBC | TIP |
|---|---|---|---|---|---|
| Calm 2013–19 | −0.34 | | | | |
| GFC 2008 | −0.47 | −55.2% | **+25.1%** | −27.4% | +3.7% |
| COVID 2020 | −0.50 | −33.7% | **+14.2%** | −25.7% | −0.6% |
| **Rate shock 2022** | **+0.03** | −24.5% | **−29.3%** | **+21.4%** | −12.7% |

Three things in that last row, and the write-up should hit all three:

1. **The hedge lost more than the thing it hedged.** TLT −29.3% against SPY −24.5%, while the
   rolling 63-day correlation was still mildly negative (mean −0.09, positive on only 31% of
   days; the in-window pairwise figure is a shade positive at +0.03, which is why the write-up
   must name which one it means). Negative correlation only protects you if bonds actually
   *rise*; in 2008 and 2020 they did, in 2022 they fell hardest.
2. **Correlation rising toward zero was enough.** For two equal-vol legs at 50/50, moving
   from −0.34 to ~0 lifts portfolio volatility from 0.57σ to 0.71σ — a third of the
   diversification benefit gone without a sign change. The measured **diversification ratio**
   `Σwᵢσᵢ / σₚ` fell from 2.18 (2012 peak) and 2.11 (2008) to 1.68 (2021), 1.59 (2022) and
   1.52 (2023, the lowest in the sample).
3. **The one sleeve that worked is the one risk parity under-weights.** Commodities returned
   **+21.4%**, and because DBC is the highest-volatility sleeve, equal *risk* contribution
   gives it the *smallest* capital weight. The diversifier that paid was the one held least.

Four artifacts from `stress.py` and `viz.py`:

1. Rolling 63-day SPY/TLT correlation **alongside the diversification ratio** — the
   correlation is the intuitive picture, the ratio is the diagnostic that actually degrades
   while the damage is being done rather than after it.
2. ERC weights over the same window: bonds' low trailing volatility earns them a large
   capital weight exactly as they stop diversifying.
3. The leverage series staying elevated into the drawdown — trailing volatility is a lagging
   estimator, so the overlay de-levers *after* the damage.
4. Crisis drawdown table across all strategies, with a **capital-weight-in-bonds** column.
   That column is what explains the ranking rather than merely reporting it.

State the loss plainly. The honest conclusion is stronger than the original one: risk parity
is not a bet that stocks and bonds are *negatively* correlated, it is a bet that bonds rally
when equities fall. Those are different bets, and 2022 is what it costs to discover the
difference. The `REGIME_SHIFT_WINDOW` in `config.py` (2021→present) is where the open
question lives — whether 2022 was an episode or a regime change.

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
parity wins 2008/2020, loses 2022); and the report artifacts land in `reports/` — four PNGs (RC
stacked area, capital-vs-risk weight comparison, leverage time series, and the rolling correlation
drawn with the diversification ratio) plus the crisis drawdown table as a CSV. (Planned as five
charts; the correlation and diversification ratio share one figure, and the table is a CSV.)

---

## Phase 6 — Incremental position analysis (post-v1.0 extension)

**Added after v1.0 is tagged (step 35). Nothing in Phases 0–5 changes for this — it is a pure
addition, not a rewrite.** The prompt for it: a single stock is not a risk-parity sleeve. It
has idiosyncratic/sector risk and no independent structural risk premium the way an asset
class does, so it cannot "compete" against SPY/TLT/DBC/TIP as a candidate fifth member of the
equal-risk-contribution book. The question that *is* well posed, and that risk desks answer
routinely, is narrower: **if we already hold the four-sleeve risk-parity book, what happens if
we add this one stock to it?** That is incremental risk analysis, and it is the shape of this
feature.

### Point-in-time, not a second backtest

Phase 0–5 answer "how would this strategy have performed since 2006." Phase 6 answers a
different question — "what does adding this position look like *right now*" — and answering it
the same way (a 2006-to-present monthly rebalance with a fifth sleeve) would fail for most
candidate stocks by construction: plenty of names investors actually ask about didn't trade in
2006, and splicing a stock's return history onto years before it existed is not a real
portfolio. So Phase 6 estimates on a **trailing window ending today** (`INCREMENTAL_LOOKBACK_DAYS`,
default ~756 trading days / 3 years) using the same EWMA-covariance machinery `covariance.py`
already provides — no new estimator, no new statistical assumptions, just a shorter window fed
through `estimate_covariance`. The tradeoff this buys: **any ticker with enough recent trading
history works**, at the cost of not being a 20-year backtest. The tool says exactly which one
it is every time it prints a report, so it is never mistaken for the other.

`data.fetch_prices` already accepts an arbitrary ticker list and already fails loudly
(`DataUnavailableError`) when a ticker returns nothing — Phase 6 needs no change to `data.py`
to support "any stock." The one new guard is `INCREMENTAL_MIN_HISTORY_DAYS`: if the candidate
has less trading history than that inside the lookback window (a recent IPO, a thinly-traded
name), the estimate is unreliable and the tool raises rather than silently returning a
covariance estimated from too few points — the same "fail loudly instead of degrading" policy
`data.py` already uses for the core universe.

### What gets computed, and why it's three things, not one

The pasted framing this phase is built from asked for four outputs; they collapse into three
pieces of math, all built from primitives `covariance.py` and `riskparity.py` already have and
already test — Phase 6 adds no new solver, no new risk model. (Step 43 later added a fourth,
different kind of output — a replay of the window rather than more math; see item 4.)

1. **Marginal risk contribution at fixed candidate weights (1%, 5%, 10%).** Scale the current
   4-asset ERC weights by `(1 − w)` and add the candidate at `w`, for each
   `w in INCREMENTAL_CANDIDATE_WEIGHTS`. Feed the resulting 5-weight vector and the 5×5
   covariance through `riskparity.risk_contributions` — the same identity `Σ RC_i = σ_p` used
   everywhere else in this project — and report the candidate's RC in isolation alongside the
   portfolio vol at each weight. This is a perturbation scan, not a re-optimization: it answers
   "what if I just add this much," which is how a position actually gets sized in practice.

2. **The full 5-asset ERC re-solve.** Separately, call `riskparity.solve_erc_with_status` on the
   5×5 covariance with no weight pinned, and diff the result against the baseline 4-asset ERC
   weights. This answers a different question than (1) — not "what if I add 5%" but "what would
   equal risk contribution actually assign this stock, and how much does it reshuffle the other
   four" — and it is why both analyses ship rather than either alone. A stock the solver assigns
   3% at equilibrium and a stock it assigns 25% are very different findings even before looking
   at return.

3. **The compensation test — is the added risk paid for.** This is the piece that turns "it adds
   risk" into a yes/no answer, and it is a known result rather than something invented for this
   project (Treynor–Black / Elton–Gruber active-portfolio theory): adding an asset `i` at an
   infinitesimal weight improves portfolio `p`'s Sharpe ratio if and only if

   ```
   SR_i > SR_p · ρ(i, p)
   ```

   where `SR_i = (μ_i − r_f)/σ_i` is the candidate's own Sharpe ratio, `SR_p` is the baseline
   4-asset portfolio's Sharpe ratio (same trailing window), and `ρ(i, p)` is the candidate's
   correlation to the *portfolio's* returns (not to any one sleeve). Derivation: write the
   blended return as `(1−ε)r_p + ε r_i`, differentiate `Sharpe(ε)` at `ε = 0`, and the sign of
   the derivative reduces to exactly this inequality. It is why correlation is "the real driver
   of whether it helps or hurts," stated precisely instead of asserted: a stock with a mediocre
   or even negative own Sharpe can still pass this test if `ρ(i, p)` is low or negative, and a
   stock with an excellent own Sharpe can still fail it if it is highly correlated with the book.
   This is a **marginal** test — it says whether increasing the weight from 0 helps, not what the
   optimal weight is — which is exactly complementary to (1) and (2), and the report says so.

   `ρ(i, p)` in the formula above is one number (candidate vs. portfolio); the report also shows
   the candidate's correlation to each of the four sleeves individually, since that decomposes
   *why* `ρ(i, p)` came out where it did (e.g. a stock uncorrelated with SPY but correlated with
   DBC is not diversifying the equity risk it looks like it's diversifying).

4. **Realized before/after risk (step 43, added after `v1.2`).** (1)–(3) all read risk off a
   covariance matrix, so the report could not say what adding a stock had actually done to a
   book's drawdown or tail losses. This piece replays the same trailing window for the baseline
   book, the book with the candidate at each `INCREMENTAL_CANDIDATE_WEIGHTS` entry, and the
   candidate alone, and reports realized risk for each: max drawdown with its peak/trough dates,
   the longest time spent below a prior peak (flagged open if unrecovered at window end),
   historical VaR and CVaR at `VAR_CONFIDENCE` (95%), worst day, downside deviation, Sortino,
   Calmar, skew and excess kurtosis, plus CAGR / total return / Sharpe for context, and a
   change-vs-baseline table. Design choices:

   - **In-sample, on purpose.** Weights are held constant and rebalanced daily on the window's own
     history — a historical simulation of *current holdings*, the standard risk-desk tool. It
     describes the risk this mix has realized, not what a strategy running the rule through time
     would have earned; it is unlevered and cost-free. `compensation_test` already holds its
     baseline weights constant the same way.
   - **Not a walk-forward engine run.** A 252-day covariance warm-up would consume a third of a
     756-day window and all of a recent IPO's data, and a full-history run would hit
     `data._validate`'s late-start guard for any ticker listed after 2006.
   - **95%, not 99%.** Over ~756 daily returns a 5% tail is ~38 observations (~14 for a 285-day
     IPO); a 1% tail would be ~8 (~3), too few for CVaR to mean anything. Undefined metrics
     (a Sortino with no losing period, a Calmar with no drawdown) print `--`, never a huge number.
   - **Its volatility differs from (1)'s, and neither is wrong.** The perturbation scan uses the
     EWMA covariance, which weights recent months heavily; the replay weights the whole window
     equally.
   - The "scale the baseline by `(1 − w)`, insert the candidate at `w`" construction is one shared
     helper (`_mixed_weights`), so (1) and (4) cannot describe different portfolios under the same
     label.

### File layout added by Phase 6

```
src/incremental.py     candidate data alignment, the perturbation scan, the 5-asset ERC
                        re-solve, the compensation test, per-sleeve correlations; and (step 43)
                        the mixed-book replay and the realized-risk metrics
analyze_position.py    CLI entry point: analyze_position.py --ticker <TICKER> [--lookback-days N]
                        [--weights 0.01,0.05,0.10] [--refresh-prices]
tests/test_incremental.py   perturbation weights sum to 1; the compensation-test formula against
                        a hand-derivable synthetic case; the 5-asset re-solve's limiting case
                        (a candidate identical to an existing sleeve must split that sleeve's RC);
                        correlation-table correctness; and (step 43) hand-derived VaR / CVaR /
                        downside deviation / Sortino / drawdown duration / Calmar, the mixed-book
                        identities with a positive control, and a mutation check that removing
                        the (1 - w) scaling makes them fail
```

`src/config.py` gains one new append-only block (`INCREMENTAL_LOOKBACK_DAYS`,
`INCREMENTAL_CANDIDATE_WEIGHTS`, `INCREMENTAL_MIN_HISTORY_DAYS`), and step 43 later appended one
more (`VAR_CONFIDENCE`). No existing file in
`src/` is modified — `types.py`, `backtest.py`, `overlay.py`, `benchmarks.py`, `stress.py`,
`viz.py` and `run_backtest.py` are all untouched by this phase, because none of them need to
know this feature exists: `incremental.py` is a second, independent consumer of the same
covariance/risk-parity primitives the core engine already validated, the same way `stress.py`
and `viz.py` are independent consumers of `BacktestResult`.

### Explicit caveats the tool must state every time it runs

- **Point-in-time, not historical.** The report is a snapshot as of the run date on a trailing
  window; it is not a claim about how the position would have performed since 2006, and it will
  change on a re-run next month as the trailing window moves.
- **A marginal test, not a sizing recommendation.** The compensation test says whether the
  *direction* of adding the position helps; it says nothing about whether 1%, 5%, or 10% is the
  right amount — that is what the perturbation scan in (1) is for, read alongside it.
- **Still a single stock.** Passing the compensation test does not manufacture a structural risk
  premium the way holding an asset class does. It answers "does this improve this portfolio's
  risk-adjusted return, given the correlation structure right now" — a narrower and more honest
  question than "should I own this stock."
- **The realized-risk table is an in-sample replay** (step 43). Today's weights, held constant and
  rebalanced daily, applied to the trailing window's own history — unlevered and cost-free, so it
  describes the risk this mix has realized, not what the strategy would have earned, and its
  levels sit below the 10%-vol-target strategy's. A window this short rarely contains a crisis,
  and the VaR / CVaR rest on few observations.

---

## Phase 7 — Head-to-head against any ticker (post-v1.3 extension)

**Added after `v1.3` (step 44). Nothing in Phases 0–6 changes for this except one additive function
in `data.py` and a small refactor in `incremental.py`.** Phase 6 asks what *adding* a stock does to the
book. The question a reader asks first is plainer: **how did the risk-parity strategy do against
simply buying and holding this ticker, and what was the difference in risk?** `run_backtest.py`
could not answer it, since it compares the strategy only against four fixed, unlevered benchmarks.

### What is compared, and on what dates

Three columns, on the dates both series exist (the *overlap window*):

1. **The strategy** — the engine's daily returns, levered to the vol target and **net of financing**;
   exactly what `run_backtest.py` reports.
2. **The same equal-risk book unlevered** (`overlay.unlevered_returns(returns, result.weights)`).
   The ticker is unlevered, so this is the like-for-like risk comparison; the levered column is not.
3. **The ticker**, buy-and-hold, total return.

The strategy is run over its **whole 2006+ history first and cut to the overlap window afterwards**.
Running it only on the overlap would make its covariance warm-up, and therefore its weights, depend on
when the ticker happened to list — a leak of the ticker's history into the strategy's decisions. Cutting
afterwards keeps every weight and leverage decision exactly what the headline run made.

### The ticker's data path

`data.fetch_prices([*UNIVERSE, ticker], start=DATA_START_DATE)` cannot be used, for two independent
reasons: `data._validate`'s late-start guard rejects any ticker that listed more than
`MAX_LATE_START_DAYS` after 2006 (correct for the sleeves, where a late start is a configuration error;
wrong for a 2019 IPO), and `cache.write_frame` replaces the whole shared price cache, so a candidate
passed alongside the universe rewrites the cached SPY/TLT/DBC/TIP history too. `data.fetch_ticker_prices`
therefore fetches the ticker alone into **its own** cache file (`ticker_<SAFE>.csv`) and validates it
against its own first date, so an empty pull, a non-positive price or an interior gap longer than
`MAX_INTERIOR_GAP_DAYS` still raise `DataUnavailableError`. "Short history" is deliberately not one of
them — how much overlap is enough is the comparison's question (`COMPARE_MIN_OVERLAP_DAYS`, 252, the
project's existing "a year of daily data is the least we trust" floor), not the fetch's.

### What is computed

- **Risk and return** for all three columns via `incremental.risk_summary` (volatility, max drawdown and
  dates, longest drawdown, VaR/CVaR at `VAR_CONFIDENCE`, worst day, downside deviation, Sortino, Calmar,
  skew, kurtosis, CAGR, Sharpe), formatted by `incremental.format_metric_table` so this report and
  Phase 6's print identical rows.
- **Relative metrics** for each risk-parity column against the ticker: correlation; beta
  `cov(s, t) / var(t)`; tracking error (annualised std of the daily difference); and up/down capture
  (the strategy's mean return on the days the ticker rose / fell, as a multiple of the ticker's own mean
  on those days).
- **A tail block** — the ticker's worst `1 − VAR_CONFIDENCE` of days, the ticker's and the strategy's mean
  on them, and the share of them on which the strategy beat the ticker. Correlation and beta average over
  the whole sample and can hide exactly this, which is the diversification claim being tested.
- **Crisis windows**, per `config.CRISIS_WINDOWS`, measured from the peak close (returns stamped strictly
  after the window's start, as `metrics.slice_to_window` does for a `BacktestResult`). A window counts
  as covered only if the overlap reaches both its ends; otherwise the row is `n/a`, never a partial window
  that reads like a full one.

### Guards

Two failures are properties of the data, not of the call, and raise `DataUnavailableError`: an overlap
under `COMPARE_MIN_OVERLAP_DAYS`, and a trading calendar that cannot be aligned
(`COMPARE_MIN_CALENDAR_MATCH`, 95% of the ticker's days must also be strategy days). Non-finite, unsorted
or duplicate-dated inputs raise `ValueError`.

**The calendar has two tiers, and the second was added after measuring rather than assumed.** The 95%
refusal catches a series that cannot be aligned at all (crypto, which trades weekends: measured 69%). It
does not catch a foreign *weekday* exchange, and measured on live tickers every one sits in a narrow band
just under a US listing's 100%: Toronto 98.1–98.2%, London 97.8%, Zurich and Hong Kong 97.5%, Frankfurt and
Sydney 97.4%, Tokyo 96.6%. Those pass the refusal and are still not clean — on the ~2–3% of days one market
was shut, a multi-day return is paired with a one-day one, and a market that closes at a different time of
day reacts to US news on a different date, which pulls correlation, beta, capture and the tail block toward
zero (non-synchronous trading). So below `COMPARE_CALENDAR_WARN_BELOW` (99%, between the two populations:
every US listing and US-traded ADR measured 100%, every foreign exchange under 98.3%) the comparison runs
but `TickerComparison.calendar_warning` is set and the report prints it, marked `!`. It warns rather than
refuses because comparing the strategy against a foreign stock is a legitimate thing to want; the failure
to avoid is silence, not the comparison. A US-traded ADR of the same company is the clean route.

### File layout added by Phase 7

```
src/compare.py         alignment and guards, relative metrics, the tail block, the crisis table,
                        the comparison dataclass and the report (pure functions of return series)
compare_ticker.py      CLI entry point: compare_ticker.py --ticker <TICKER> [--refresh-prices]
                        — fetches, runs the engine, prints; owns no maths
tests/test_compare.py  relative metrics on hand-derived pairs (s = 0.5 t, s = -t, a four-day example);
                        the tail block on a hundred returns with a known worst five; crisis windows
                        (day-count compounding, peak-day exclusion, n/a coverage); alignment and both
                        guards; a check that the identities FAIL when the ticker is one day out of step
tests/test_data.py     fetch_ticker_prices against a monkeypatched yfinance: a late-listed ticker is
                        accepted where fetch_prices' guard rejects it, the shared cache is untouched
```

`src/config.py` gains one append-only block (`COMPARE_MIN_OVERLAP_DAYS`, `COMPARE_MIN_CALENDAR_MATCH`, `COMPARE_CALENDAR_WARN_BELOW`).
`incremental.py` gains a public `format_metric_table` extracted from its own report code, so the two
reports cannot drift apart; nothing else in `src/` changes.

### Explicit caveats the tool must state every time it runs

- **Overlap window only.** The numbers will not match `run_backtest.py`'s full-history headline.
- **Not like-for-like on leverage.** The strategy is levered and financed; the ticker is not. Read the
  unlevered column and the scale-free ratios.
- **Chosen with hindsight.** A ticker someone asks about today is more often a winner than not, which
  flatters it against a strategy that never got to pick it.
- **No costs.** No transaction costs, bid-ask spread or borrow spread over the bill rate, on either side.
- **Short windows are noisy.** A 5% tail is 5% of the observation count and a max drawdown is one episode.
