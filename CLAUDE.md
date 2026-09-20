# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build ledger — what is done, at a glance

**`git log --oneline` is the truth; this table is a convenience that goes stale.** Commit subjects
are `Step NN [A|B]: ...`, so the log reads as the same ledger. Step numbers, owners and the full
"done when" for each row live in `TASKS.md` — that text is the spec, not this summary.

**Done: steps 1–44, all on `main`, tagged `v1.0` (steps 1–35), `v1.1` (steps 36–42), `v1.2`
(a post-tag correctness audit, no numbered steps), `v1.3` (step 43), `v1.3.1` (a small
cleanup and one tooling fix, no numbered steps) and `v1.4` (step 44).** The engine, the analysis layer, the hardening, the CLI, the live run, the real charts,
the write-up, the final review, the incremental position analysis layered on top as new additive
scope, in `v1.2` five bugs and one dead-code removal found by auditing the whole repo after
`v1.1` shipped, in `v1.3` — step 43 — realized before/after risk metrics added to that
incremental analysis, in `v1.3.1` two dead items removed plus a fix so the `src.viz` /
`src.stress` self-checks stop overwriting the tracked `reports/` files, and in step 44 a
head-to-head comparison of the strategy against buy-and-hold of any ticker, tagged `v1.4`.
Nothing is left in any of the six; see "Incremental position analysis" and "Head-to-head
comparison" below for what `v1.1`, step 43 and step 44 added and found, and "What the v1.2 audit found" below for the fifth
commit table.

| Step | Owner | Landed as | What it produced |
| --- | --- | --- | --- |
| 1–3 | A | `2eac618` `2237b70` `466ab1f` | scaffold, `config.py`, **`types.py` — the contract** |
| 4–6 | B | `0cd02e7` `728a852` `1c3e089` | `cache.py` (verbatim port), `metrics.py` (daily port), `tests/fixtures.py` |
| 7 | Both | `54446cf` | Phase 0 gate: `tests/test_contract.py`. **B unblocked from here** |
| 8–10 | A | `1670e80` `c7f6271` `8683fba` | `data.py`: yfinance prices, keyless FRED `DTB3`, returns + rebalance calendar |
| 11–13 | B | `43cef53` `1fea96f` `4f6b332` | `benchmarks.py` (equal-weight, 60/40, max-Sharpe), crisis-window slices in `metrics.py` |
| 14–15 | A | `d51fce7` | `covariance.py`: EWMA covariance, shrinkage, PSD repair in correlation space |
| 16–17 | A | `f05e03f` | `riskparity.py`: risk contributions, `naive_inv_vol`, the SLSQP ERC solver |
| 18–20 | B | `5ffd1fd` `6c26f98` | `tests/test_riskparity.py`: RC identity, **the independent convex oracle**, degenerate cases |
| 21–22 | A | `da0c70c` `46acb51` | `overlay.py`: vol targeting, the leverage cap, financed leverage |
| 23 | A | `725e120` | `backtest.py`: the rebalance loop → a real validated `BacktestResult` |
| 24–27 | B | `f0b2758` | `stress.py` (crisis table, diversification ratio) and `viz.py` (the four charts) |
| 28 | A | `46906f4` `fea684c` | `tests/test_lookahead.py`: the `<= t` guard, **attacked by removing it** |
| 29 | A | `9659e8e` | `meta` diagnostics + `backtest.format_run_summary`, marking failures with `!` |
| 30 | B | `160b8e8` | `run_backtest.py`: the end-to-end CLI, every flag documented |
| 31 | B | `864a5f6` | `README.md`: install, run, and the method — results sections left for Phase 5 |
| 32–34 | A/B | `61d7c1f` | The live run, real charts in `reports/`, and the README write-up — see "Where the build is" |
| 35 | Both | `1dcd185` | Final review: retired both stand-ins in `benchmarks.py`, wired `format_run_summary` into `run_backtest.py`, lowered `MAX_CONDITION_NUMBER` to a value that can actually fire. Tagged `v1.0`. |

Step 32 (the live run itself) has no dedicated commit of its own — it was folded into `61d7c1f`
rather than landed as a separate A-owned PR first, so the ledger records it as done but the
process TASKS.md describes for it did not run exactly as written. Worth knowing if a future
step here looks "skipped": check the log before assuming it wasn't done.

Follow-up fixes that landed on top of those steps and are worth knowing about: `361c995` (reject a
non-PSD covariance in `solve_erc` rather than solving it), `d61e7bd` (reject out-of-order rows in
both EWMA estimators), `bc39670` (measure crisis windows from the peak close), `b9916a8` /
`36a275c` (correct the 2022 thesis against live data), `bc7c36f` / `1491e6b` / `c5f9c5f`
(audit passes on Phase 0, Phase 2 and B's step 24–27 branch), and `dfd0f84` (retire
`tests/test_riskparity.py`'s dead Phase-2 fallback solver and its seven permanently-unreachable
`if riskparity is None` guards, found in a post-v1.0 consistency audit — the suite still prints
61 checks passed).

**`v1.1` is tagged. Phase 6 (36–42) is done**, a later, purely additive pivot (see "Incremental
position analysis" below and `PLAN.md`/`TASKS.md`) that changes or deletes nothing in Phases 0–5:

| Step | Owner | Landed as | What it produced |
| --- | --- | --- | --- |
| 36 | A | `0fda448` | `src/config.py`: `INCREMENTAL_LOOKBACK_DAYS` (756), `INCREMENTAL_CANDIDATE_WEIGHTS` (1%/5%/10%), `INCREMENTAL_MIN_HISTORY_DAYS` (= `COV_WARMUP_TRADING_DAYS`) |
| 37 | A | `6754098` | `src/incremental.py`: the trailing-window fetch (candidate + universe together, so the shared price cache keeps all five columns rather than being overwritten down to one), the baseline ERC snapshot, the perturbation scan, the unconstrained 5-asset re-solve |
| 38 | A | `fb726b8` | `src/incremental.py`: the compensation test `SR_i > SR_p · ρ(i, p)` and per-sleeve correlations |
| 39 | B | `3db9c3a` | `tests/test_incremental.py`: 40 checks, offline — the duplicate-sleeve symmetry, the formula regression, correlation-table correctness |
| 40 | B | `02b95a1` | `analyze_position.py`: the CLI, plus `format_position_report` in `src/incremental.py` |
| 41 | Both | `e0bc297` | `README.md`: the "Incremental Position Analysis" section, worked example against a real ticker |
| 42 | Both | `9ea54e1` | Final review: an independent multi-agent code review over the whole Phase 6 diff, 9 of its 10 findings fixed in `src/incremental.py`/`analyze_position.py` (see "What the final review found" below). Tagged `v1.1`. |

See "Incremental position analysis" below for what this answers, why it is a snapshot rather than
a second historical backtest, and what building it found; `PLAN.md`'s Phase 6 section has the full
derivation.

Four loose ends flagged during Phase 4 as "Phase 5 work" were resolved as part of step 35's final
review, rather than left for whoever noticed next: both stand-ins in `src/benchmarks.py` are
retired (`_ewma_cov` and `_nearest_psd` are gone; the module now imports
`covariance.estimate_covariance` / `covariance.nearest_psd` directly, so the real estimator is
the default even for a caller that omits `cov_estimator`), `format_run_summary` is wired into
`run_backtest.py` (printed right after the engine runs, alongside the generic `_print_meta`
dump), and `MAX_CONDITION_NUMBER` came down from `1e8` to `10_000` — the old value measured at
three orders of magnitude above anything this project's data can produce, so the diagnostic
could never fire; the new one sits ~8.5x above the worst legitimate case measured (1,183) and
would still catch a genuinely pathological covariance. See "Two stand-ins" below for what
changed and why, and the `MAX_CONDITION_NUMBER` comment in `config.py` for the measurement
behind the new value.

**`v1.2` is tagged.** Not a phase and not on `TASKS.md` — a correctness audit run after `v1.1`
shipped, the same instinct as step 35's and step 42's final reviews but scoped to the whole repo
rather than one phase's diff. Four parallel passes (config/cache/types/data; the math core;
metrics/benchmarks/stress/viz; Phase 6 plus both CLIs and the full test suite) read every source
file for bugs the existing test suite didn't reach. Five real issues and one piece of dead code,
each landed as its own PR and verified against a constructed failure scenario, not just re-read:

| What it produced | Landed as |
| --- | --- |
| `src/types.py`: `BacktestResult.validate()`'s look-ahead check used `<` instead of `<=`, silently permitting `returns.index[0] == weights.index[0]` — a decision earning its own rebalance day's return. `tests/test_contract.py` already asserted the strict inequality on a fixture; `validate()` itself, the one function every result is required to pass through, didn't enforce it. | `9b21a38` |
| `run_backtest.py`: added the `ERCSolverError` handling `analyze_position.py` already had, so a genuinely infeasible ERC problem on a live run reports cleanly instead of crashing with a raw traceback. | `6ac2526` |
| `src/covariance.py` / `src/viz.py` / `src/metrics.py`: `condition_number()` took `abs()` of the eigenvalues before finding the smallest, letting an indefinite matrix report a finite number instead of `inf`; the correlation/diversification chart's crisis shading read a transient axis range instead of the final one; mixing a crisis-window name with an explicit end date produced a confusing downstream parse error instead of a clear one. | `bd51290` |
| `src/incremental.py` / `analyze_position.py`: five edge-case fixes — a misleading NaN-vs-missing-label message in `compensation_test`, an empty-`candidate_weights` guard in `perturbation_scan`, duplicate-`--weights` rejection, a NaN Sharpe reporting as a silent "FAILS" instead of "UNDEFINED", and an empty-`--ticker` check. | `33e093e` |
| `src/types.py`: removed the dead `ERCSolver` Protocol and `solve_erc` stub — Phase 0-1 scaffolding unreferenced since the real solver landed in `src/riskparity.py` at step 17. | `f6868af` |

None of these changed a design decision, a documented formula, or a number in `README.md` — they
close gaps an already-passing test suite didn't reach, which is why `README.md` and `PLAN.md`
needed no update alongside them. All 8 offline test suites and all 7 module self-checks stayed
green throughout.

**`v1.3` is tagged (step 43).** New scope, not an audit: `analyze_position.py`
reported only *model-based* numbers (covariance-derived vol and risk contribution, the 5-asset
re-solve, the compensation test), so it could not say what adding a stock had actually done to a
book's drawdown or tail losses. It now also replays the trailing window and reports realized risk
for each book. Like Phase 6 it is not on `TASKS.md`, so this row is its only ledger entry:

| Step | Owner | Landed as | What it produced |
| --- | --- | --- | --- |
| 43 | A | `de61034` (PR #52, merged `4a03e34`) | Realized before/after risk in `src/incremental.py`: a constant-weight replay of the baseline book, the book with the stock at each candidate weight, and the stock alone, with max drawdown and its dates, longest drawdown (flagged open), historical VaR/CVaR at 95%, worst day, downside deviation, Sortino, Calmar, skew, excess kurtosis, and a change-vs-baseline table. `config.py` gains `VAR_CONFIDENCE` only. `tests/test_incremental.py` goes from 40 to **76** checks. |

See "Incremental position analysis" below for the design choices in it that are easy to undo by
accident.

**`v1.3.1` is tagged.** Not a phase and not on `TASKS.md` — a cleanup pass after `v1.3`, each item
its own PR, with no change to any number, formula or design decision:

| What it produced | Landed as |
| --- | --- |
| `src/data.py` / `src/config.py`: removed an unused `import numpy as np` and the unreferenced `BENCHMARK_TICKER` constant, found by scanning every module for imports and config constants with no reference in code or docs. Deliberately kept: the unreferenced helpers in `src/cache.py` and `metrics.cumulative_curve`, because both files are documented verbatim ports that must not be rewritten. | `b226867` (PR #54) |
| `src/viz.py` / `src/stress.py`: the `python -m` self-checks wrote fixture output into `reports/`, silently replacing the tracked real charts and `crisis_drawdowns.csv`; they now render into a temporary directory. `CLAUDE.md`'s claim that `reports/` is gitignored was corrected (see "Architecture"). | `1f93dfc` (PR #55) |

**`v1.4` is tagged (step 44).** New scope: `run_backtest.py` compares the
strategy only against four fixed, unlevered benchmarks and `analyze_position.py` asks what
*adding* a stock does to the book, so there was no way to ask how the strategy did against simply
buying and holding a ticker of your choice. `compare_ticker.py --ticker XYZ` does. Like Phase 6 and
step 43 it is not on `TASKS.md`, so this row is its only ledger entry:

| Step | Owner | Landed as | What it produced |
| --- | --- | --- | --- |
| 44 | A | `bc1209d` and `e4fcdc6` (PR #58, merged `f1cf95d`) | `src/compare.py` + `compare_ticker.py`: the levered strategy, the same book unlevered, and the ticker on their overlap window, with the full risk-metric set, relative metrics (correlation, beta, tracking error, up/down capture), a tail block (the strategy on the ticker's worst 5% of days) and crisis windows. New `data.fetch_ticker_prices`. `config.py` gains `COMPARE_MIN_OVERLAP_DAYS`, `COMPARE_MIN_CALENDAR_MATCH`, `COMPARE_CALENDAR_WARN_BELOW`. New `tests/test_compare.py` (**61** checks); `tests/test_data.py` 35 → **49**. |

See "Head-to-head comparison" below for the design choices in it that are easy to undo by accident.

Read "Where the build is" next — the live-run numbers and the two findings that constrain the
write-up are there, and they are the part a summary cannot carry.

## What this is

Project 16 — a risk-parity + volatility-targeting portfolio engine over four ETF sleeves
(SPY / TLT / DBC / TIP), backtested 2006→present through the GFC, COVID and the 2022 rate shock.
The deliverable is not just the engine: the analytical payload is **why risk parity failed in 2022**.
Results that make the strategy look bad are stated plainly, not hidden.

Note the mechanism carefully, because the obvious version is wrong and this repo already corrected
it once against live data: the stock/bond correlation did **not** flip positive during the 2022
drawdown. Over `2022-01-03 → 2022-10-12` the rolling 63-day SPY/TLT correlation averaged **−0.09**
and was positive on only 31% of days; the persistent flip is 2023 onward. What happened is that
**TLT fell 29.3% against SPY's 24.5%** — the hedge lost more than the thing it hedged — while
correlation merely rose toward zero, which was enough to remove the offset. See PLAN.md's 2022
section and the "Data reality check" in TASKS.md before writing anything about 2022.

Those sleeve returns are **peak-to-trough** — `price(end)/price(start) − 1`, the returns stamped
*after* the window's opening close, because the start date is the peak itself. Quoting a crisis
window by compounding the returns stamped *on* `[start, end]` instead measures from the close
before the peak, which is a different and larger number: it reads TLT 2022 as −31.2% against
SPY's −24.1%, inflating the gap from 4.8pp to 7.1pp. Earlier drafts of this file carried those
figures. `metrics.slice_to_window` uses the exclusive bound and
`tests/test_crisis_slices.py::test_window_excludes_the_peak_days_return` pins it.

`PLAN.md` holds the design rationale and the pinned math. `TASKS.md` holds the numbered build
order — every step names one owner, one branch, and a "done when". Read the relevant step before
starting work; the step text is the spec.

## Where the build is

Which steps are done is in the ledger above; this section is what the finished steps *found*.
**Check `git log --oneline` before trusting either — they go stale, the log does not.**

**The engine runs end to end on live data, and the numbers are now final, in `README.md`'s
Results section.** 234 monthly rebalances, 2007-02-28 → 2026-07-31, 4897 return days: **CAGR
7.12%, vol 11.26%, return/vol 0.667, max drawdown −29.10%**, mean leverage 1.76x, **0 solver
failures, 0 covariances repaired, worst risk-contribution deviation 1.6e-07**. (Ratios are
return-to-volatility at a zero risk-free rate, not a true Sharpe — applied identically across
strategies so the ranking is fair.)

**The headline result the write-up leads with: risk parity loses.** Plain 60/40 beats it over
the full sample on both return (8.72% vs 7.12% CAGR) and risk-adjusted return (0.786 vs 0.667),
running the same volatility with no leverage, no solver and no financing. Charging real
financing costs 1.06%/yr of risk-adjusted return versus free leverage (0.667 vs 0.761), and the
unlevered ERC book alone — 5.10% CAGR, 0.754 return/vol, −17.15% max drawdown — is the best
risk-adjusted result of any row in the table. The overlay bought volatility, not return. See
`README.md`'s Results section for the full breakdown.

**A live-run finding the write-up has to address, because the project is framed around 2022 and
this does not fit that frame.** The worst drawdown in the sample is not 2022 — it is **COVID, at
−29.10%** (peak 2020-02-20, trough 2020-03-18), against −25.43% for the GFC and **−17.72% for
2022**. And the strategy walked into it at **2.33x, its highest pre-crisis leverage of the whole
run** (GFC 1.83x, 2022 1.74x), because 2019 was calm and the trailing estimator had nothing else
to say. That is the lagging-overlay mechanism from step 21 at its most expensive: the vol target
levers hardest into quiet regimes, and the fastest crash in the sample followed the quietest year.
2022 is still the more interesting *mechanism* — it is where diversification itself failed rather
than where the overlay mistimed — but "risk parity's worst month was caused by its own vol target"
is a finding, and this repo does not hide the ones that look bad.

**Phase 2 is closed.** The cross-check it existed for passed: Person B's independent convex
log-barrier oracle agrees with Person A's SLSQP solver to **7.3e-08** on 50 random covariances
(bar: `ERC_AGREEMENT_TOLERANCE`, 1e-6), and both were checked against the equicorrelation closed
form so a shared sign error would still have been caught. Its highest-value use was the ρ = −0.93
case: the restart ladder can say it left a local minimum, but only an independent derivation can
say the escaped answer is the *global* optimum rather than a second, prettier local one. It is, to
2.9e-09, after 8 restarts.

Two things the solver work established that stay true downstream:

- The least-squares ERC objective `Σ(RC_i − σ_p/N)²` is **not convex**. It has local minima that pin
  an asset at `MIN_WEIGHT` with a *negative* risk contribution, and they show up whenever two
  sleeves are strongly negatively correlated — i.e. exactly the SPY/TLT pair. SLSQP reports
  `"Optimization terminated successfully"` on every one of them, so the optimiser's own status is
  not evidence of anything. `solve_erc_with_status` escapes them by restarting.
- That escape does **not** reach zero. About 0.05% of adversarial random 4×4 covariances (|ρ| up to
  0.95) still miss the bar; on covariances drawn around the real four-sleeve structure it is 0 in
  2000. Anything asserting across many random covariances should **assert agreement where both
  solvers converge**, rather than assuming every draw does — over 50 draws a per-draw rate of 0.05%
  is a ~2.5% flaky test. Numbers and the diminishing-returns curve are in the `ERC_MAX_RESTARTS`
  comment in `config.py`.

To resume in a new session: every numbered step through 44 is built — `v1.1` closed Phase 6,
`v1.2` adds a post-tag audit's bug fixes, `v1.3` is step 43 (realized before/after risk), `v1.3.1`
a small cleanup, and `v1.4` is step 44 (the head-to-head comparison) — there is no unfinished step
in `TASKS.md` to pick up, and `main` already has everything `v1.4` does. New work here is a fresh
scope, not a continuation.
The fastest way back into the core engine is
`.\.venv\Scripts\python.exe run_backtest.py --no-charts`, which exercises the whole pipeline on
live data in one command; `.\.venv\Scripts\python.exe analyze_position.py --ticker AAPL` does the
same for the Phase 6 incremental analysis, and
`.\.venv\Scripts\python.exe compare_ticker.py --ticker AAPL` for the head-to-head comparison
(a few seconds with prices cached: it re-runs the whole engine each time).

Two stand-ins that used to live in `src/benchmarks.py` are **retired as of step 35**, not just
superseded at the call site. Worth knowing why both had to go, since the reasoning is still what
would stop someone from reintroducing either as a "simplification":

- `_ewma_cov` → `src.covariance.estimate_covariance`. Already done at the call site that matters
  before step 35 — `run_backtest.py` passes `cov_estimator=estimate_covariance` to all four
  benchmarks, so every number in a real run already came from the engine's own estimator. What
  step 35 removed was the module *default* (`estimate_cov = cov_estimator or _ewma_cov`), which
  had still applied to any caller that omitted the argument, including the `__main__` self-check
  and the fixtures. `_ewma_cov` is deleted outright now that nothing references it.
- `_nearest_psd` → `src.covariance.nearest_psd`. This one mattered more to retire, because it was
  **still in the path even under injection**: `max_sharpe` and `naive_risk_parity` both called
  `_nearest_psd(estimate_cov(...))`, so the stand-in ran on top of whatever estimator was
  injected. It was harmless as long as `estimate_cov` was `estimate_covariance` — it early-returned
  the symmetrised input untouched, since `estimate_covariance`'s output is always above
  `MIN_EIGENVALUE` already — but the two are **not** equivalent in general: `_nearest_psd` clips
  eigenvalues of the covariance directly, which shifts the diagonal and quietly alters asset
  volatilities, while `covariance.nearest_psd` repairs in correlation space and leaves them
  intact. A tangency portfolio is about as sensitive to those volatilities as a portfolio gets, so
  a future `cov_estimator` that actually needed repair would have gotten the wrong one silently.
  `src/benchmarks.py` now imports `nearest_psd` from `src.covariance` directly; there is no
  private copy left to diverge.

## Incremental position analysis (Phase 6, v1.1; extended by step 43 — done)

A single stock is not a risk-parity sleeve — it has idiosyncratic/sector risk and no
independent structural risk premium the way an asset class does, so it was wrong to frame it as
a candidate fifth member competing with SPY/TLT/DBC/TIP. The question that **is** well posed,
and that this phase answers, is narrower: given the existing four-sleeve risk-parity book, what
happens if you add one more stock — any stock — to it? That is standard incremental risk
analysis (what a risk desk does whenever someone proposes adding a position to a book), not a
second backtest, and it shipped as new, additive files only:

- **Point-in-time, not historical.** (Step 43 adds an in-sample *replay* of the same window —
  see its bullet below — which is still not a strategy backtest.) It estimates on a trailing
  window ending today
  (`INCREMENTAL_LOOKBACK_DAYS`, ~3 years) through the same `covariance.estimate_covariance` the
  core engine already uses — no new estimator. This is the deliberate trade against a full
  2006-to-present backtest: most candidate stocks people actually ask about don't have 20 years
  of history, and splicing pre-IPO years is not a real portfolio. The report always says which
  kind of analysis it is. Verified against a real recent IPO, not just reasoned about: RDDT
  (listed 2024-03-21) correctly falls back to a shrunk window (285 observations, above the
  252-day `INCREMENTAL_MIN_HISTORY_DAYS` floor, below the 756-day target) rather than being
  rejected outright.
- **`data.fetch_prices` already accepts any ticker** and already fails loudly
  (`DataUnavailableError`) on one with no data, so "any stock the user chooses" needed zero
  *structural* changes to `data.py` — only a new minimum-history guard
  (`INCREMENTAL_MIN_HISTORY_DAYS`) for a candidate too thinly traded for a trailing-window
  covariance to be trustworthy. What this does **not** avoid, and the final review had to
  document rather than silently fix: `data.py`'s cache (`cache.write_frame`) replaces its whole
  file rather than merging into it, so fetching an uncached candidate alongside the universe on
  a short analysis window can temporarily shrink the *universe's* cached history too, not only
  add the candidate's column. It is bounded and self-healing — `run_backtest.py`'s own fetch
  always requests `DATA_START_DATE` and checks coverage before trusting the cache — and the
  alternative (requesting `DATA_START_DATE` here as well) would force most real candidates into
  the short-history retry unnecessarily. See the "final review" note below and
  `src/incremental.py`'s `_fetch_window` docstring for the full trade-off.
- **Three pieces of model-based math (plus step 43's replay, next bullet), all built from
  primitives `riskparity.py` already validates:** (1) a
  perturbation scan — scale the 4-asset ERC weights by `(1−w)` and add the candidate at
  `w ∈ {1%, 5%, 10%}`, read its risk contribution and the portfolio vol at each; (2) an
  unconstrained 5-asset `solve_erc_with_status` re-solve, diffed against the baseline weights,
  answering the different question of what equal risk contribution would actually assign the
  stock; (3) the compensation test, `SR_i > SR_p · ρ(i, p)` — a known result (Treynor–Black /
  Elton–Gruber), not invented here: adding asset `i` at an infinitesimal weight improves
  portfolio `p`'s Sharpe ratio iff the candidate's own Sharpe exceeds the portfolio's Sharpe
  times its correlation to the portfolio. It is a **marginal** test (does the direction help),
  answered alongside — not instead of — the perturbation scan (does this amount help). Verified
  against five hand-derivable two-asset synthetic cases (ρ = +1, 0, −1) built via an *exact*
  Gram-Schmidt construction rather than a random draw, so the check has zero sampling noise. The
  full derivation is in `PLAN.md`'s Phase 6 section.
- **Step 43 — a fourth piece: realized before/after risk.** The three above read risk off a
  covariance matrix; this replays the same trailing window (`simulate_mixed_books`,
  `realized_risk_table`, all in `src/incremental.py`) so the report can say what a mix has
  actually been through. Choices that look like details and are not:
  - **It is in-sample on purpose.** Today's weights, held constant and rebalanced daily, applied to
    the window's own history — the standard risk-desk historical simulation of *current holdings*.
    It describes realized risk, not what a strategy running the rule through time would have
    earned, and it is unlevered and cost-free. The report's fourth caveat says so every run. A
    walk-forward engine run was rejected: a 252-day covariance warm-up eats a third of a
    756-day window and all of a recent IPO's data, and a full-history fetch trips
    `data._validate`'s late-start guard for anything listed after 2006. `compensation_test`
    already holds its baseline weights constant the same way, so this is consistent with it.
  - **The risk functions live in `src/incremental.py`, not `src/metrics.py`.** `metrics.py` is a
    deliberately verbatim port (see "Conventions inherited"); `historical_var`, `historical_cvar`,
    `downside_deviation`, `sortino_ratio`, `calmar_ratio`, `drawdown_duration` and `risk_summary`
    follow its conventions but sit beside their only consumer. Do not move them into `metrics.py`
    as a tidy-up without deciding that deliberately.
  - **`_mixed_weights` is shared with `perturbation_scan`.** The "scale the baseline by `(1−w)`,
    insert the candidate at `w`" construction exists once, so the model-based scan and the replay
    cannot describe different portfolios under the same `+5%` label.
  - **VaR/CVaR are 95%, not 99%** (`config.VAR_CONFIDENCE`): over ~756 daily returns a 5% tail is
    ~38 observations (~14 for a 285-day IPO), a 1% tail ~8 (~3), too few for CVaR to mean anything.
    Undefined metrics (a Sortino with no losing period, a Calmar with no drawdown) print `--`,
    never a huge finite number.
  - **The replay's vol runs above the perturbation scan's, and neither is wrong.** Live AAPL:
    6.4% realized against 4.6% modelled at a 1% position, because the scan uses the EWMA
    covariance (recent, calmer months weigh heavily) and the replay weights the whole three years
    equally. Do not "reconcile" them.
  - **Verified against live data, not just fixtures:** VaR/CVaR hand-recomputed from the cached
    prices matched the table to six decimals (tail of 38 of 756), and RDDT's short 285-day window
    renders with the open-drawdown flag. A first version of the test rebuilt the window from the
    fixture returns instead of the prices → `daily_returns` → last-756 path `analyze_position`
    actually takes and failed on that mismatch — a test bug, not a code bug, but the same lesson as
    the rest of this repo: a check has to reconstruct the real path, not a convenient one.
- **Nothing in `src/types.py`, `backtest.py`, `overlay.py`, `benchmarks.py`, `stress.py`,
  `viz.py` or `run_backtest.py` changed for this** (nor did step 43 touch `metrics.py`, and
  `config.py` gained only `VAR_CONFIDENCE`). `src/incremental.py` is a new, independent
  consumer of `covariance.py`/`riskparity.py`, the same way `stress.py` is an independent
  consumer of `BacktestResult` — and it ships behind its own CLI, `analyze_position.py`, not a
  flag on `run_backtest.py`.

**What the final review found (step 42).** An independent multi-agent code review of the whole
Phase 6 diff — the same "derive it a second way" instinct the project already applies to the ERC
solver (Person B's oracle) and the look-ahead guard (attacked by removal) — surfaced 10 real
issues, 9 fixed in `9ea54e1`. Worth carrying forward:

- Two were **documentation overclaims caught against actual behaviour**, not just reworded: the
  cache-sharing note above, and `_fetch_window`'s min-history retry, which is sized to the
  *floor* (`INCREMENTAL_MIN_HISTORY_DAYS`) rather than to however much history a candidate
  between the floor and the target actually has — a real gap between the docstring's claim and
  the code, not a phrasing nit.
- One was a **live-data-verified bug in a fix made during the same review**: replacing a
  hand-tuned calendar-day ratio with `pd.bdate_range` (cleaner, reuses the same primitive
  `data.py` already does its own calendar work with) also needed its holiday buffer
  re-measured — the first value chosen (30 business days) returned 754 observations against a
  756-day target on a real fetch, not 756. Caught by testing the fix against live data rather
  than trusting the reasoning; raised to 45 after measuring the actual shortfall.
- One was an **exception-hierarchy bug with a real behavioural consequence**:
  `analyze_position.py` caught `ValueError` before `DataUnavailableError`, and
  `riskparity.ERCSolverError` subclasses `ValueError`, so a genuine solver/economic failure
  (a non-PSD covariance, a near-zero-vol asset) was silently reported as a CLI misconfiguration
  (exit 2) instead of a data problem (exit 1) — the same "two failure modes must stay
  distinguishable" principle the core engine's `backtest.py` already enforces, missed here on
  the new code path.
- One was a **latent contract violation**: `compensation_test`'s type hint promised a `pd.Series`
  input would be handled safely, but the implementation read it positionally rather than by
  label — no call site in the shipped code triggers it, but a future caller passing a
  differently-ordered Series would have gotten silently mismatched weights, exactly the failure
  mode this project's whole `types.py` contract exists to prevent elsewhere.

The one finding *not* fixed on the code side: `_fetch_window`'s min-history retry always using
the floor rather than the candidate's true available history was judged a documented trade-off,
not a bug — computing "however much is actually available" needs a search, not a single retry,
and the current behavior (analysed on a smaller window than theoretically possible, clearly
flagged as such) is honestly labelled rather than silently wrong. See `_fetch_window`'s
docstring for the reasoning.

## Head-to-head comparison against a ticker (step 44 — done)

`compare_ticker.py --ticker XYZ` runs the real strategy and buys-and-holds any ticker (total return)
on the dates both exist, in three columns: the strategy (levered to the vol target, **net of
financing**), **the same equal-risk book unlevered** (the ticker is unlevered, so this is the
like-for-like risk read) and the ticker. `src/compare.py` is pure functions of return series and
tests offline; `compare_ticker.py` only fetches, runs the engine and prints, importing `load_data`
from `run_backtest.py` so the data path is identical to the headline run. Choices that look like
details and are not:

- **The strategy is run over its whole 2006+ history and cut to the overlap window afterwards.**
  Running it only on the overlap would make its covariance warm-up, and so its weights, depend on
  when the ticker happened to list — the ticker's history leaking into the strategy's decisions.
  The check that it is right: the levered column reproduces `run_backtest.py`'s numbers exactly
  (−29.10% max drawdown; the unlevered book's 5.10% CAGR and −17.15%).
- **`data.fetch_ticker_prices` exists because `fetch_prices` cannot do this, and must not be
  "simplified" back into it.** `fetch_prices([*UNIVERSE, ticker], start=DATA_START_DATE)` fails
  twice over: `_validate`'s late-start guard rejects any ticker listed after 2006 (right for the
  sleeves, wrong for a 2019 IPO), and `cache.write_frame` replaces the whole shared price cache, so
  the candidate rewrites the cached SPY/TLT/DBC/TIP history. The new function fetches one ticker
  into **its own** file (`data/cache/ticker_<SAFE>.csv`), validates against the ticker's *own* first
  date, and never reads or writes `prices_close.csv` (a test checks the shared file is byte-for-byte
  unchanged, and a positive control checks `fetch_prices`' guard rejects the same late-listed series).
  This is the one place a candidate is fetched without the shared-cache shrink that
  `analyze_position.py` has and documents.
- **The calendar guard has two tiers, and the second exists because the first missed a real case.**
  A first version refused only under 95% calendar match (crypto measured 69%). Running Toyota on the
  Tokyo exchange then passed silently at 96.6%. Measured across live tickers, every foreign weekday
  exchange sits in a narrow band under a US listing's 100% (Toronto 98.1–98.2%, London 97.8%, Zurich
  and Hong Kong 97.5%, Frankfurt and Sydney 97.4%, Tokyo 96.6%) while US listings and US-traded ADRs
  are 100%. Those runs are usable but biased: ~2–3% of days pair a multi-day return with a one-day
  one, and a market closing at a different time of day reacts to US news on a different date, which
  pulls correlation, beta, capture and the tail block toward zero. So below
  `COMPARE_CALENDAR_WARN_BELOW` (0.99, between the two measured populations) it runs and the report
  prints a `! calendar warning`. It warns rather than refuses because comparing against a foreign
  stock is legitimate; the failure to avoid is silence. The clean route to a foreign company is its
  US-traded ADR. Do not raise the 0.95 refusal to 0.99 as a "tightening": that would silently drop
  every non-US listing.
- **Crisis windows are measured from the peak close and are `n/a` unless the overlap reaches both
  ends of the window** — the same peak-exclusive rule `metrics.slice_to_window` applies to a
  `BacktestResult`, repeated in `compare.crisis_table` because these are bare series. The two series
  share one index, so the later starter decides; a ticker that listed after a window's peak gets
  `NaN`, never a partial window that reads like a full one.
- **The metric rows are shared with Phase 6, not re-derived.** `compare.py` imports
  `incremental.risk_summary` / `realized_risk_table` and `incremental.format_metric_table`, which was
  extracted from `_format_realized_section` for this, so both reports print identical rows in an
  identical format. The dependency runs `compare` → `incremental`; keep it that way.
- **The relative metrics have exact definitions worth keeping:** beta is `cov(s, t) / var(t)`;
  up/down capture are the strategy's mean return on the days the ticker rose/fell as a multiple of
  the ticker's own mean on them (arithmetic daily means); the tail block is the ticker's worst
  `1 − VAR_CONFIDENCE` of days, with the share of them on which the strategy beat the ticker —
  which correlation and beta average over the whole sample and can hide.
- **What it found, and how to read it.** Against AAPL (2007-03-01 → 2026-09-18) the strategy was
  about a third as volatile, drew down −29% against −61%, lost less in every crisis window and beat
  AAPL on all 246 of its worst days — and earned 7.1% a year against 28.5% with a *lower* Sharpe
  (0.67 against 0.96). Against SPY it earned 7.1% against 11.1% but with a −29% against −55% max
  drawdown and a higher Sharpe (0.67 against 0.63) and Calmar (0.24 against 0.20). It is a risk
  story, not a return story, and the "chosen with hindsight" caveat matters: a ticker someone asks
  about is usually one that did well.
- **Verified against live data, not just fixtures:** the printed beta, correlation, total return and
  tail block matched an independent recomputation to six decimals (AAPL); SPY, GOOGL, MSFT and
  `^GSPC` run; RDDT (625-day overlap) marks every crisis window `n/a`; a nonexistent ticker exits 1
  with a clean message. Building it produced three *test-side* mistakes — a "starts after every
  window" series that began in 2021 and so still covered 2022, a strategy label that defaulted
  differently from the fixture's, and a collision test that the code's own upper-casing made
  unreachable — each caught because the expectation was derived by hand instead of assumed.
- **Not built:** charts, strategy flags (`--target-vol`, `--max-leverage`), caching the strategy run
  between invocations (a run is a few seconds with prices cached), and multiple tickers per run.

## Setup and commands

System Python has no numpy on this machine — always work inside a venv.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python tests/test_contract.py       # Phase 0 gate: contract + metrics port + fixtures
.venv/bin/python tests/test_data.py           # data layer: windowing, calendar, look-ahead guard
.venv/bin/python tests/test_benchmarks.py     # equal weight / 60-40 / max Sharpe / naive 1-vol
.venv/bin/python tests/test_crisis_slices.py  # crisis-window slicing and summary metrics
.venv/bin/python tests/test_stress.py         # crisis table, diversification ratio, the four charts
.venv/bin/python tests/test_riskparity.py     # RC identity, the ERC oracle, degenerate cases
.venv/bin/python tests/test_lookahead.py      # the <= t guard, incl. removing it to prove the teeth
.venv/bin/python tests/test_incremental.py    # Phase 6: perturbation scan, compensation test, 5-asset limit,
                                              # realized risk (step 43)
.venv/bin/python tests/test_compare.py        # step 44: relative metrics, tail block, crisis windows, guards
.venv/bin/python tests/fixtures.py            # print a synthetic BacktestResult summary
.venv/bin/python run_backtest.py              # full run -> reports/
.venv/bin/python analyze_position.py --ticker AAPL   # Phase 6: incremental analysis against a live ticker
.venv/bin/python compare_ticker.py --ticker AAPL     # step 44: strategy vs buy-and-hold of any ticker
```

Engine modules also carry a `__main__` self-check, which is where Person A's steps assert their
"done when" until Person B's matching suite exists. Run them the same way:

```bash
.venv/bin/python -m src.covariance    # halflife -> inf == sample cov; PSD repair on indefinite input
.venv/bin/python -m src.riskparity    # analytic gradient vs central differences; ERC on 500 randoms;
                                      # degenerate inputs and the non-PSD rejection regression
.venv/bin/python -m src.overlay       # vol -> leverage across regimes; the cap; the de-levering lag;
                                      # financing at L=1..3 and the one-day decision lag
.venv/bin/python -m src.backtest      # full synthetic run: contract, alignment, look-ahead, meta
.venv/bin/python -m src.benchmarks    # capital weight vs risk weight, side by side
.venv/bin/python -m src.stress        # crisis table + both regime diagnostics, from fixtures
.venv/bin/python -m src.viz           # the four charts from fixtures, into a temp dir (never reports/)
.venv/bin/python -m src.incremental   # Phase 6: hand-derivable compensation-test cases, the
                                      # MIN_HISTORY_DAYS guard, and a live-ticker demonstration
.venv/bin/python -m src.compare       # step 44: a full report from fixtures; offline, writes nothing
```

Run every suite after any change to `src/`. **All nine test suites run offline, and so do eight
of the nine module self-checks** — verified by blocking `socket.connect` / `getaddrinfo` and
re-running each one, not by assumption. `src/data.py` imports `requests`, so the import graph
touches the network stack, but nothing under `tests/` opens a connection. `src.incremental` is
the one self-check exception: its live-ticker demonstration reaches the network on purpose (see
"Incremental position analysis" above) — there is no way to demonstrate `analyze_position.py`'s
actual behaviour without it. The MIN_HISTORY_DAYS guard and the hand-derivable compensation-test
cases inside that same self-check are still deterministic and offline; only the final
"live ticker" block touches yfinance. Keep every other suite and self-check offline: live-data
checks belong in a live run (Phase 5's `run_backtest.py`, Phase 6's `analyze_position.py`), not in
a test, because a suite that needs the network fails for reasons that have nothing to do with the
code.

**Every suite exits 0.** `test_riskparity.py` used to exit 1 by design, on a known bug filed
against `solve_erc`; that fix has landed and the suite is green. If one exits non-zero now, it is a
real failure — do not silence it.

Tests are **standalone scripts with no framework** — no pytest, no test runner. Each file defines
`test_*()` functions plus a `check(label, condition)` helper, and a `__main__` block calls them in
order and prints `N checks passed`. Run a single test file by running that file; run a single test
by calling its function in the `__main__` block. New test files must match this shape, including
the `sys.path.insert(0, ...)` preamble that makes `from src import ...` work when run from the
repo root.

This file is checked in and read on both machines, so "this machine" would be ambiguous here: the
interpreter path depends on the OS, not on whose repo it is. **Windows** (Person A, and what PLAN.md
and TASKS.md quote): `.\.venv\Scripts\python.exe`. **macOS/Linux** (Person B): `.venv/bin/python`.
The commands above are written in the POSIX form; substitute as needed.

## Architecture

One module per pipeline stage under `src/`, wired end to end by `run_backtest.py`:

```
config.py → data.py → covariance.py → riskparity.py → overlay.py → backtest.py → BacktestResult
                                                                                       ↓
                                                          metrics.py / benchmarks.py / stress.py / viz.py
```

`src/__init__.py` carries the authoritative stage list, and **every module in it now exists**.
`backtest.run_backtest(returns, risk_free=...)` is the entry point that walks the calendar and
returns a validated `BacktestResult`; it composes `apply_overlay(returns, weights, leverage,
risk_free)`, whose `unlevered` / `leverage` / `risk_free` / `strategy` frame is where the
financing leg is visible side by side with the return it was charged against.

`stress.py` and `viz.py` consume `BacktestResult` and a daily asset-return frame and nothing else,
so they already run end to end against `tests/fixtures.py`. Phase 5 swaps the fixture for the real
result and re-renders; no call site changes. `reports/` is gitignored by default, because a
committed fixture chart is indistinguishable from a real one to anyone browsing the repo — but the
four charts and `crisis_drawdowns.csv` **are tracked** (force-added in Phase 5 from a live
`run_backtest.py`; see `.gitignore`), so anything that writes to `reports/` overwrites the real
results. Nothing but `run_backtest.py` may: the `python -m src.viz` and `python -m src.stress`
self-checks write to a temporary directory for exactly this reason (they used to write into
`reports/` and silently replaced the tracked files with fixture output — found after `v1.3`,
fixed in `v1.3.1`). If `git status` ever shows `reports/` modified after running a self-check, that is this
bug back; restore with `git checkout -- reports/`, and do not commit the fixture versions.

Two notes on the Phase 2 modules, because both encode a decision that looks like a detail:

- `covariance.py` repairs a non-PSD matrix **in correlation space** — factor out the volatilities,
  clip the correlation matrix's eigenvalues, renormalise its diagonal to 1, multiply the
  volatilities back. The renormalisation is a congruence transform by a positive diagonal, so unlike
  overwriting the diagonal in place it cannot reintroduce a negative eigenvalue. Volatilities come
  back unchanged and the correlation structure absorbs the adjustment, which is the right place to
  put it: the variances were the well-estimated part.
- `riskparity.py` keeps `optimiser_succeeded`, `converged` and `restarts_used` apart on
  `ERCSolution`, and they disagree in both directions. It also **raises** rather than returning
  `converged=False` when no equal-risk portfolio exists above the weight floor at all — a sleeve
  whose volatility is a millionth of its peers' is a broken price series, not a solver event.
- `solve_erc_with_status` **rejects a non-PSD covariance rather than repairing it**, on a bar
  relative to the largest variance (`ERC_MIN_EIGENVALUE_RATIO`). Both halves are deliberate.
  Repair belongs upstream in `covariance.py` where the run summary can see it happen; a solver that
  quietly fixed its own input would hide a broken estimator. And an *absolute* bar would mean
  something different at every scale, since these are annualised ETF variances running ~1e-2 —
  note in particular that a relative-looking `max(1.0, max|diag|)` silently pins itself to the
  absolute bar for every matrix below unit scale, which is all of them here. The bar is set from
  the gap between two measured populations, not picked: legitimately singular matrices (ρ = 1
  exactly, which must keep solving — it is the TLT/TIP-in-a-rate-shock limit) bottom out at −1e-15
  of the largest variance, pure `eigvalsh` rounding noise, while a genuine violation sits at −0.42.
  This guard exists because without it `solve_erc` returned weights for a matrix implying a
  correlation of 1.67 while every field on `ERCSolution` read healthy.

And one on `overlay.py`, which is the module most likely to be "fixed" by someone who reads its
behaviour as a bug:

- **The overlay de-levers *after* the damage, and that is a finding, not a defect.** A trailing
  estimator cannot see volatility until it has been realised, so leverage is still elevated well
  into every drawdown. Measured on live data with equal weights, the fraction of each crisis window
  the book spent still levered *above 1.0x*: **GFC 69%, COVID 54%, 2022 26%** — and COVID reached
  its minimum leverage on 2020-03-23, the exact market bottom. Shortening `VOL_EWMA_HALFLIFE_DAYS`
  moves the lag; nothing removes it, because no causal filter can. Vol targeting is a risk
  *control*, not a risk *forecast*. This is PLAN.md's third artifact — do not tune it away.
- `trailing_portfolio_vol` estimates on a **counterfactual** series: the weights *about to be held*
  applied to past returns, not the portfolio returns actually realised (which used older weights).
  The quantity being levered is the risk of the next book, and the realised series lags hardest
  right after the weights move — i.e. after a regime change, when the estimate matters most.
- It demeans with the **weighted** mean and uses the same `1 - Σw²` correction as
  `ewma_covariance`, which makes `trailing_portfolio_vol(r, w, h)²` equal `wᵀ ewma_covariance(r, h) w`
  *exactly*. That is a free second derivation of the same number and the self-check asserts it, so
  do not "simplify" either convention independently. The overlay still needs its own halflife —
  sharing the matrix would tie its reaction speed to the allocator's, which is the one thing
  `VOL_EWMA_HALFLIFE_DAYS` exists to prevent.
- **Two alignment rules live in `overlay.py` and they differ by one day. Step 23 depends on
  getting this right.** Sizing the book you are *about to hold* uses the decision **at or before**
  the date (`leverage_path`); attributing the return *earned on* a date uses the decision
  **strictly before** it (`_in_force_before`, used by `unlevered_returns` / `financed_returns`) —
  a choice made at the close of day `d` cannot have earned day `d`'s move. `reindex(...).ffill()`
  implements the first; reaching for it to do the second is the bug this module is shaped to
  prevent, and it is worth **+0.053%/yr of CAGR and +0.006 of Sharpe** of unearned hindsight on
  the real run. Note the rule is chosen by **kind, not shape**: a *daily* leverage series is still
  a decision series and is still deferred a day. An earlier draft switched on whether the series
  already covered the return index, which silently mis-classified `leverage_path`'s own default
  output — the most likely input of all.
- **Financing is charged, and it is what the honest version of this strategy costs.** On the real
  four-sleeve run, unlevered ERC is CAGR 5.11% / Sharpe 0.75; levered to the 10% target *with*
  financing it is 7.13% / **0.67**; with the financing leg omitted it is 8.27% / **0.76**. So
  free-money leverage appears to leave Sharpe intact while real leverage visibly *reduces* it —
  which is exactly the oversell PLAN.md warns about, worth 1.14%/yr here. `financed_returns`
  computes `L·r_a − (L−1)·r_f` rather than the equivalent `L·(r_a − r_f) + r_f` because only the
  first is bit-exact at `L = 1` (the second misses on 7.3% of draws — `(a−b)+b ≠ a`), and step 22's
  "done when" asks for exactness.

And on `backtest.py`, which owns no maths of its own and is therefore wrong only in ways that look
like tidying:

- **The daily accrual is one call to `apply_overlay`, deliberately.** Hand-rolling the alignment
  is how the one-day rule gets undone, and the result still produces a plausible equity curve.
  The self-check pins `overlay`'s `_in_force_before` against `benchmarks._unlevered_result`'s
  `.ffill().shift(1)` — they agree to **0.0 over 1241 days**, and they have to, or the engine and
  its own benchmarks are not comparable.
- **The two solver failures are different events and are handled differently.**
  `converged=False` is numerical: use the best iterate, log, count it in `meta["solver_failures"]`,
  keep going — one awkward covariance in 234 is not a reason to discard nineteen years.
  `ERCSolverError` is economic: no equal-risk portfolio exists above the weight floor, which means
  a sleeve's volatility is a millionth of its peers' and the price series is broken. It propagates
  with the date attached and kills the run. Collapsing these two would hide whichever one mattered.
- **`risk_contributions` are stored as fractions of `σ_p`** (rows sum to 1, per the contract), and
  `backtest.py` composes them from `riskparity`'s primitives rather than importing
  `benchmarks.risk_contribution_fractions` — `benchmarks` sits *downstream* in the pipeline and
  importing it would invert the layering. The self-check asserts the two agree (5.6e-17), which is
  what the duplication warning in `risk_contributions`' docstring is actually protecting.
- **Not modelled, and say so when asked:** no transaction costs, no bid-ask, no borrow spread over
  the bill rate. Financing is charged at `DTB3` flat, so a real book pays more and these levered
  results are still optimistic — less so than the free-leverage version, but optimistic. Monthly
  turnover on four liquid ETFs is low enough that costs would not change the *shape* of any
  conclusion here, which is why the omission is defensible rather than why it is invisible. The
  run summary prints this line too, because it is the block someone screenshots.
- **A failure has to be visible three ways (step 29), and `meta["run_health"]` is the one to read
  first.** It is `"clean"` or a phrase naming what went wrong; `format_run_summary(result)` renders
  the whole picture and marks a bad line with a leading `!`; and the run logs a warning both when a
  rebalance fails and once at the end. Silence from the last of those means a clean run, which is
  the only thing silence is allowed to mean. The date fields in `meta` are **short strings, not
  lists of `Timestamp`** — a list prints as its length in a generic dump, so `solver_failure_dates`
  as a list would read `2 entries` and the date would be invisible exactly where someone reads it.
  `meta["diagnostics"]` stays the lossless per-rebalance frame; filter that when you want data.
- **`0 ill-conditioned` means "nothing pathological", not "well conditioned".** Measured while
  building step 29: shrinkage maps `ρ → 0.9ρ`, which caps the shrunk *correlation* matrix's
  condition number at `(1 + (N−1)(1−d))/d` = **37** for four sleeves — 37.00 at `ρ = 1` exactly,
  where the unshrunk matrix is 3.2e32. So `MAX_CONDITION_NUMBER = 1e8` needs a ~1,600:1 volatility
  ratio between two sleeves to fire at all; over 20,000 random covariances at realistic vols the
  worst was 1,183. The informative fields are the condition numbers themselves, which is why the
  median is reported beside the max. Whether to lower that constant is a Phase 5 decision, not a
  silent one.
- **`format_run_summary` is not wired into `run_backtest.py` yet.** The CLI's generic `_print_meta`
  dump surfaces every new field including `run_health`, so nothing is hidden — but printing the
  formatted block is a one-line addition in B's file, and Phase 5 is when to make it.

### The contract is the load-bearing design decision

`src/types.py` defines `BacktestResult` (frozen dataclass: `weights`, `risk_contributions`,
`leverage`, `returns`, `meta`). It exists so two people on separate machines can build the engine
and the analysis layer *at the same time*: Person B builds charts, benchmarks and stress tests
against `tests/fixtures.make_fake_result()` long before a real engine runs, and Phase 5 swaps the
fixture for the real thing without touching anything downstream.

Consequences that constrain how you work:

- **Never modify `src/types.py` on a feature branch.** Changing the contract means announcing it
  first, then one PR that changes the contract and every call site together.
- **Every function that builds a `BacktestResult` calls `.validate()` before returning it.**
  `validate()` raises `ContractViolation` and is deliberately loud — a silently malformed result is
  the one failure mode that lets the two halves diverge unnoticed.
- Anything a run wants to surface but no chart depends on goes in `meta` (`strategy` is required).
- `weights` and `risk_contributions` are kept side by side rather than derived on demand, because
  their divergence *is* the project's thesis.

### Look-ahead is the invariant everything else rests on

At rebalance date `t`, only data timestamped `<= t` may be used — covariance, trailing vol estimate
and risk-free rate alike. `data.asof_slice()` is the single chokepoint for this; route new
estimators through it rather than re-deriving the guard at each call site.

`tests/test_lookahead.py` (step 28, 49 checks) attacks that one function three ways, and the third
is the one to preserve if the file is ever refactored:

1. a 12× spike in the last 200 days — every earlier rebalance must be **bit-identical**, and later
   ones must move, since "the future changed nothing" is also true of an engine ignoring its input;
2. **truncation equivalence** — deleting everything after a cutoff cannot change a rebalance before
   it. Stronger than the spike, because it does not depend on the perturbation moving the leaked
   quantity;
3. **removing the guard** — `asof_slice` is monkeypatched to a leaky identity and 1–2 are re-run
   *expecting failure*. Note it patches every module that imported the function, found by scanning
   `sys.modules`: `backtest` and `overlay` both do `from .data import asof_slice`, so patching
   `src.data` alone would change nothing and the mutation test would pass while proving nothing.

The same file covers the one-day rule with its own mutation, and `data.align_risk_free`'s
forward-fill-only direction. What it deliberately does **not** assert is that the hindsight version
earns more: that is true on live data (+0.053%/yr) but across 12 seeds the synthetic sign flips
(−0.148%/yr to +0.135%/yr, positive on 4), so it pins the mechanism instead — the leak must land on
exactly the rebalance days and no others.

### Conventions inherited from `abhyudayb-1/sp500-factor-backtester`

That repo is the house style and several files here are deliberate ports — **do not rewrite them**:

- `src/cache.py` is verbatim. Atomic `.tmp`-then-`replace` writes; CSV/JSON so the cache stays
  human-inspectable. `read_frame`/`write_frame` for date-indexed frames, `read_table`/`write_table`
  for tidy panels with a RangeIndex.
- `src/metrics.py` is ported with exactly one change: `periods_per_year` is a required argument,
  not a module constant, because this project accrues **daily** (252) while the original was
  monthly (12). Two subtleties in it look like noise and are not — `_ZERO_VARIANCE_EPS` (a bare
  `sd == 0` guard lets a Sharpe of ~1e16 through) and the drawdown peak floored at 1.0 (without it
  a window starting at a market peak reports 0% drawdown). Tests assert both.
- `src/viz.py` (step 26) reuses `_style_axes` / `_new_figure` / `_title` / `_save` and the palette.

### config.py is shared and append-only

Every magic number lives in `src/config.py`, never inline at a call site. It is the only file both
people write, so: add constants at the bottom of the relevant block, never reorder or reformat what
is there, and name the block you touched in the PR body.

Notable policy encoded there: `DATA_START_DATE = 2006-02-03` (DBC's inception — the brief asked for
2005 and the data cannot deliver it; state this rather than back-filling a proxy), `TARGET_VOL`,
`LEVERAGE_CAP = (0.5, 3.0)`, the three `CRISIS_WINDOWS`, and the EWMA halflives (vol halflife is
shorter than covariance halflife on purpose — the overlay should react faster than the weights).

One trap in the ERC block, since the names look interchangeable and are not. `ERC_TOLERANCE` is
**not** what SLSQP receives; `ERC_SLSQP_FTOL` is. `ftol` is an *absolute* threshold on an objective
that is already ~1e-12 at the 1e-6 risk-fraction bar, so passing `ERC_TOLERANCE = 1e-12` stops the
optimiser exactly *at* the bar — measured, it missed on 35 of 200 random 4×4 covariances. Every
constant in that block carries the measurement that set it; read the comments before changing one.

### Data policy

Keyless sources only, so anyone can clone and run: yfinance for prices, the FRED CSV endpoint
(`fredgraph.csv?id=DTB3`) for the risk-free rate. No API keys anywhere in this project.

- Prices are pulled `auto_adjust=True` — the close series is **total return**. Not cosmetic: TLT and
  TIP pay most of their return as coupons, and a price-only series would badly understate the sleeve
  risk parity leans on hardest.
- `data.py` raises `DataUnavailableError` rather than degrading. A silently dropped sleeve produces
  a plausible equity curve and an entirely wrong conclusion.
- Leverage is **financed, not free**: `r_p = L_t·(r_assets − r_f) + r_f`. Omitting the financing leg
  is the most common way this strategy gets oversold.

## Working style for this repo

- Failures are surfaced, never swallowed. Solver non-convergence and covariance condition numbers
  go into `meta` and the run summary; bad data raises. Two failure modes must stay distinguishable
  in `meta` even when they share a fallback path — a solver that did not converge is not the same
  event as a genuine economic condition, and collapsing them hides the one you needed to see.
- **Never assert on a single simulated path.** Over a 0.78-year window at ~24% annualised
  volatility, one path's total return carries a ~21% standard deviation, so sleeve orderings flip
  seed to seed. A realised-path check written this way failed on 3 of 15 seeds before it was caught.
  Assert on the *parameters* instead (`_CRISIS_PROFILES`, the correlation tables) — equally
  informative, never flaky. Ensemble means across seeds are the fallback when a realised property
  genuinely must be tested. It bit a second time in step 28: "the look-ahead version earns more" is
  true on live data (+0.053%/yr) and *failed on the first seed tried*, because across 12 seeds the
  synthetic drift runs −0.148%/yr to +0.135%/yr and is positive on 4 of them. The fix was to assert
  the mechanism (which days moved) rather than the outcome (which direction), and to sweep seeds
  before believing any new check.
- **A test that cannot fail proves nothing — so break the thing and watch it fail.** The house
  technique now, used three times: `asof_slice` monkeypatched to a leaky identity, `_in_force_before`
  put back to the naive `ffill`, and a solver failure injected at the seam in `backtest.py`'s
  self-check. All three assert that the *reporting or the guard* is what makes the green result
  green. Every perturbation check also needs its positive control — the poisoned run has to differ
  *somewhere*, or an engine ignoring its input passes.
- Bugs found by review get a regression check in the matching suite, not just a fix. Every bug
  fixed here so far was silent — passing tests, plausible output, wrong behaviour — so "it runs and
  the number looks reasonable" is not evidence of anything.
- Comments here explain *why* a subtle choice is correct (the ones that would otherwise get
  "simplified" away). Match that density — this is a capstone that gets read and defended, not
  just run.
- Ownership split: Person A (`abhyudayb-1`) owns the engine modules, Person B (`shivakasula`) owns
  metrics/benchmarks/stress/viz/tests/CLI/README. Only touch the files listed for the step you are
  on. Branches are `feat/a<NN>-slug` / `feat/b<NN>-slug`, one step per branch, merged before the
  next starts. Two standing exceptions, both already exercised: a step may assign a file outside its
  owner's usual area (step 28 gave `tests/test_lookahead.py` to A), and `CLAUDE.md` refreshes go on
  their own `docs/`-prefixed branch rather than riding along with a feature.
- **Phase 5 is joint, so the split above stops applying at step 32.** Its branches are named in
  `TASKS.md` (`feat/a32-live-run`, `feat/b33-real-charts`, `feat/analysis-2022`) and steps 34–35
  are both people. Wiring `format_run_summary` into `run_backtest.py` — A's function, B's file — is
  the first thing that would have needed a negotiation under the old rule and does not now.
