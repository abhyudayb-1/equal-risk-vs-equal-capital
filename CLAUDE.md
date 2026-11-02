# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build ledger — what is done, at a glance

> **Read this first: the repo was cut back after `v1.4`, so parts of this ledger are history
> rather than current state.** The project's bottom line is one question — what allocating equal
> *risk* buys over allocating equal *capital* across a multi-asset book, and what that costs in
> volatility and return (see "What this is"). Work that answered a different question was removed
> from `main`, each piece in its own PR, with the reusable parts pulled out first.
>
> | | Status |
> |---|---|
> | The four-sleeve ERC engine, overlay, benchmarks, stress, charts (steps 1–35, `v1.0`/`v1.2`) | **kept — this is the project** |
> | `src/riskmetrics.py`, `src/headtohead.py` | **kept** — salvaged for Phase 8, and `src/comparison.py` consumes both now |
> | `src/comparison.py` (steps 53–54) | **the study itself** — reproduces `PLAN.md`'s Phase 8 table and the risk-allocation block |
> | Incremental position analysis, "add one stock to the book" (steps 36–43, `v1.1`/`v1.3`) | **REMOVED** — see its section below |
> | Head-to-head against a ticker (step 44, `v1.4`) | **REMOVED** — see its section below |
> | `web/` — the research site | **on `main` now**, with four non-research features removed; see "The website" below |
> | `TASKS.md` Phase 7 (website, steps 45–52) | **re-scoped** — steps 46 and 51 struck; that file carries the note |
> | `TASKS.md` Phase 8 (steps 53–60) | **in progress** — steps 53 and 54 done; 55–60 remain |
>
> Rows below are left as they landed, because `git log` records the same thing and the code stays
> reachable at its tags. Where a row describes removed work, a note says so.

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
Both of the post-`v1.0` products — the incremental position analysis and the head-to-head against
a ticker — have since been removed from `main`; their sections below say what they were, what was
kept out of them and which tag holds them. "What the v1.2 audit found" below is the fifth commit
table.

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

**Steps 36–42 and 43 below are history, not current state: that work has been REMOVED from
`main`.** The rows stay because `git log` does, and the code stays reachable at tags `v1.1` and
`v1.3`. See the "Incremental position analysis — REMOVED" section below for why it went and what
was kept out of it. `PLAN.md`'s and `TASKS.md`'s Phase 6 sections were deleted with it, so there
is nothing left to look up there.

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

Removed along with the rest of Phase 6 — see the note above and the section below. The realized
risk metrics it introduced are the part that survived, now in `src/riskmetrics.py`.

**`v1.3.1` is tagged.** Not a phase and not on `TASKS.md` — a cleanup pass after `v1.3`, each item
its own PR, with no change to any number, formula or design decision:

| What it produced | Landed as |
| --- | --- |
| `src/data.py` / `src/config.py`: removed an unused `import numpy as np` and the unreferenced `BENCHMARK_TICKER` constant, found by scanning every module for imports and config constants with no reference in code or docs. Deliberately kept: the unreferenced helpers in `src/cache.py` and `metrics.cumulative_curve`, because both files are documented verbatim ports that must not be rewritten. | `b226867` (PR #54) |
| `src/viz.py` / `src/stress.py`: the `python -m` self-checks wrote fixture output into `reports/`, silently replacing the tracked real charts and `crisis_drawdowns.csv`; they now render into a temporary directory. `CLAUDE.md`'s claim that `reports/` is gitignored was corrected (see "Architecture"). | `1f93dfc` (PR #55) |

**`v1.4` is tagged (step 44).** New scope at the time: `run_backtest.py` compares the strategy
only against four fixed, unlevered benchmarks, so there was no way to ask how the strategy did
against simply buying and holding a ticker of your choice. `compare_ticker.py --ticker XYZ` did.
Like Phase 6 and step 43 it is not on `TASKS.md`, so this row is its only ledger entry — and
**like Phase 6 it has since been removed from `main`** (see its section below):

| Step | Owner | Landed as | What it produced |
| --- | --- | --- | --- |
| 44 | A | `bc1209d` and `e4fcdc6` (PR #58, merged `f1cf95d`) | `src/compare.py` + `compare_ticker.py`: the levered strategy, the same book unlevered, and the ticker on their overlap window, with the full risk-metric set, relative metrics (correlation, beta, tracking error, up/down capture), a tail block (the strategy on the ticker's worst 5% of days) and crisis windows. New `data.fetch_ticker_prices`. `config.py` gains `COMPARE_MIN_OVERLAP_DAYS`, `COMPARE_MIN_CALENDAR_MATCH`, `COMPARE_CALENDAR_WARN_BELOW`. New `tests/test_compare.py` (**61** checks); `tests/test_data.py` 35 → **49**. |

Removed — see its section below for what was kept out of it and the three rules from it that
still apply to surviving code.

Read "Where the build is" next — the live-run numbers and the two findings that constrain the
write-up are there, and they are the part a summary cannot carry.

## What this is

**A multi-asset portfolio-construction study with one question: what does allocating equal *risk*
buy you over allocating equal *capital*, and what does it cost in volatility and return?**

Four ETF sleeves (SPY / TLT / DBC / TIP), daily, 2006→present, through the GFC, COVID and the 2022
rate shock. Two construction rules are the protagonists — **equal risk contribution (ERC)** and
**equal capital (1/N)** — with naive 1/σ, max-Sharpe and 60/40 as reference points. Results that
make risk parity look bad are stated plainly, not hidden; that is the whole reason the repo exists
in this shape.

The study has three parts, and each has a measured answer. **`src/comparison.py`'s
`compare_rules()` reproduces every one of these numbers** (step 53; data through 2026-10-30, 237
rebalances, 4,950 return days) — what it does not yet have is a CLI, which is step 56.
`README.md`'s Results section is a separate, older snapshot (through 2026-07-31); do not mix the
two in one table.

1. **Risk allocation — what risk does each rule actually take?** ERC hits equal risk to 1.6e-07
   at every rebalance while its *capital* swings from 16% to 42% a sleeve. 1/N holds capital flat
   at 25% and lets risk go where it likes: SPY's risk share runs **8.9% to 60.3%** (mean 32.9%),
   DBC's averages **41.8%**, TIP's **9.0%**, and TLT's goes **negative** (−6.3%) in some months.
   The worst single-sleeve risk concentration 1/N reaches in any month is **66.5%**. In effective
   bets (1/HHI of risk shares) that is **2.92 of a possible 4.00, against ERC's 4.00** — the
   compactest statement of the thesis in the repo. (An earlier draft said 2.83. That was
   `1 / mean(Σ RC²)`; the average number of bets is `mean(1 / Σ RC²)`, and because `1/x` is convex
   the first understates it. Step 54 computes the second and `comparison.concentration_table`'s
   docstring records the trap.)
2. **The unlevered comparison.** ERC 5.12% CAGR / 6.90% vol / **0.759** return-to-vol / −17.2% max
   drawdown, against 1/N's 5.79% / 8.35% / **0.716** / −28.1%. So equalizing risk buys a real
   risk-adjusted improvement and nearly halves the drawdown — and gives up **67 bp/yr of return**
   to do it. Naive 1/σ gets **0.747** with no optimizer at all, so the correlation structure the
   ERC solver exists for is worth about a point of that improvement, not all of it. Say so.
3. **The volatility/return trade-off, which is what the overlay is for now.** Two books at
   different volatilities are not comparable on return, so the vol-targeting overlay exists to put
   ERC at 1/N's own volatility. Levered there with **real T-bill financing charged**, ERC scores
   **0.696 — below 1/N's 0.716**. With the financing leg omitted it scores **0.767**, above it. The
   financing cost eats the entire risk-allocation gain and a little more. That single pair of
   numbers is the project's sharpest result: **equal risk wins on risk-adjusted return until you
   pay for the leverage that makes the comparison fair.**

**The overlay is a comparison instrument, not the strategy.** It used to be framed as the second
half of the product ("risk parity + volatility targeting"); it is now the thing that makes a
like-for-like comparison possible, and its own costs — financing, and the lag documented under
`overlay.py` in "Architecture" — are findings in their own right rather than defects.

The crisis chapters stay, as **evidence for part 2 and 3 rather than as the thesis**: they are where
the covariance-estimated diversification ERC depends on visibly stops holding, and where the
overlay's lag is most expensive. Note the 2022 mechanism carefully, because the obvious version is
wrong and this repo already corrected it once against live data.

The stock/bond correlation did **not** flip positive during the 2022 drawdown.
Over `2022-01-03 → 2022-10-12` the rolling 63-day SPY/TLT correlation averaged **−0.09**
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

`PLAN.md` holds the design rationale and the pinned math, including the **Phase 8** section that
specifies the comparison above and records the scan that measured it. `TASKS.md` holds the numbered
build order — **Phase 8 (steps 53–60) is the work to pick up**, and step 53 is the one that makes
those numbers reproducible. Read the relevant step before starting work; the step text is the spec.

## Where the build is

Which steps are done is in the ledger above; this section is what the finished steps *found*.
**Check `git log --oneline` before trusting either — they go stale, the log does not.**

**Two data dates are in play and must not be mixed.** `README.md`'s Results section is a live run
through **2026-07-31** (234 rebalances, 4897 return days) and is the published write-up. The
comparison numbers in "What this is" come from a **scan through 2026-10-30** (237 rebalances, 4950
days) run while re-scoping the project, and **no CLI reproduces them yet** — Phase 8 step 53 is
what closes that. Either set is fine to quote; quoting both in one table is not. Refreshing
`README.md` onto one date is Phase 8 step 59.

**The engine itself is sound on both dates:** 0 solver failures, 0 covariances repaired, worst
risk-contribution deviation 1.6e-07, `run_health clean`. Ratios throughout are
return-to-volatility at a zero risk-free rate, not a true Sharpe — applied identically to every
row, so the ranking is fair even though the level is not.

**The headline the write-up now leads with, and it is not flattering to risk parity.** Equalizing
risk does buy something real: unlevered ERC takes **4.00 effective bets against 1/N's 2.92**, earns
**0.759 return-to-vol against 0.716**, and nearly halves the drawdown (−17.2% against −28.1%). Then
two things take it away. **Lever ERC to 1/N's own volatility so the returns are comparable, charge
the real T-bill, and it scores 0.696 — below 1/N.** With the financing leg omitted it scores 0.767,
above it; that gap *is* the cost of leverage, and a backtest that omits the term reports the second
number and calls it the first. And **naive 1/σ reaches 0.747 with no optimizer at all**, so the
correlation structure the ERC solver exists for is worth roughly a point of the improvement rather
than all of it. Plain 60/40 still beats everything here on risk-adjusted return (0.778).

So the honest summary is: **equal risk is a better *risk* allocation than equal capital and a worse
*return* proposition once the leverage that makes it comparable is paid for.** Both halves get
stated.

**What the overlay costs in a crisis, and it is the most expensive single finding in the repo.**
The worst drawdown in the sample is not 2022 — it is **COVID, at
−29.10%** (peak 2020-02-20, trough 2020-03-18), against −25.43% for the GFC and **−17.72% for
2022**. And the strategy walked into it at **2.33x, its highest pre-crisis leverage of the whole
run** (GFC 1.83x, 2022 1.74x), because 2019 was calm and the trailing estimator had nothing else
to say. That is the lagging-overlay mechanism from step 21 at its most expensive: the vol target
levers hardest into quiet regimes, and the fastest crash in the sample followed the quietest year.
2022 is still the more interesting *mechanism* — it is where diversification itself failed rather
than where the overlay mistimed — but "the deepest drawdown in the sample was caused by the
comparison instrument, not the allocation rule" is a finding, and this repo does not hide the ones
that look bad. It is also the reason part 3 of "What this is" reports the unlevered comparison
alongside the matched-risk one rather than only the latter.

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

To resume in a new session: steps 1–44 were built, then the repo was **cut back to one question**
(see "What this is"). Phase 6's incremental position analysis and the head-to-head against a
ticker are both removed; `PLAN.md` is now purely the Phases 0–5 design plus the Phase 8 spec,
`TASKS.md` Phases 0–5 are untouched, its Phase 7 is re-scoped, and **`TASKS.md` Phase 8
(steps 53–60) is the work to pick up, and steps 53 and 54 are done** — `src/comparison.py`
reproduces every row of `PLAN.md`'s Phase 8 table *and* the risk-allocation block, so the figures
in "What this is" are no longer unreproducible. **Step 55 (the standalone suite) is next**, then
56's CLI. Step 58's sensitivities are the ones most likely to be uncomfortable; run them
before the write-up, not after.
The fastest way back into the engine is
`.\.venv\Scripts\python.exe run_backtest.py --no-charts`, which exercises the whole pipeline on
live data in one command.

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

## Incremental position analysis (Phase 6, v1.1; step 43) — REMOVED

`src/incremental.py`, `analyze_position.py` and `tests/test_incremental.py` are gone from `main`.
They answered "given the four-sleeve risk-parity book we already hold, what does adding one more
*stock* to it do?" — a perturbation scan, a 5-asset ERC re-solve, the compensation test
`SR_i > SR_p · ρ(i, p)`, and an in-sample replay of the trailing window. It worked and it was
tested; it is simply not this project's question. Single-name idiosyncratic risk is not
multi-asset portfolio construction, and the project's bottom line is equal *risk* against equal
*capital* across a multi-asset book.

**Reachable, not destroyed:** tag `v1.1` has Phase 6 as shipped, `v1.3` adds step 43's realized
risk, and both PRs' diffs stay on GitHub. `git show v1.3:src/incremental.py` brings it back.

**What was kept, because it outlived the feature:** `src/riskmetrics.py` (historical VaR/CVaR,
downside deviation, Sortino, Calmar, drawdown duration, `risk_summary`, `realized_risk_table`,
`format_metric_table`) was carved out of it first, and `src/headtohead.py` was carved out of
`compare.py` the same way. `config.VAR_CONFIDENCE` stays for them; the three `INCREMENTAL_*`
constants went with the module.

**Two findings from it that still apply to code that is still here**, so they are not lost with
the prose that used to be in this section:

- **`cache.write_frame` replaces its whole file rather than merging into it**, so fetching a new
  ticker alongside the universe through `data.fetch_prices` on a short window can *shrink* the
  universe's cached history as well as add a column. It is bounded and self-healing —
  `run_backtest.py` always requests `DATA_START_DATE` and checks coverage before trusting the
  cache. `data.fetch_ticker_prices` existed to work around it and has itself been removed, so the
  hazard is now unguarded — see the "Head-to-head comparison" section below.
- **`riskparity.ERCSolverError` subclasses `ValueError`**, so a `except ValueError` placed before
  `except DataUnavailableError` silently reports a genuine solver/economic failure as a CLI
  misconfiguration. This bit the removed CLI once and `run_backtest.py` carries the same two
  handlers; keep them ordered most-specific first.

## Head-to-head comparison against a ticker (step 44, `v1.4`) — REMOVED

`src/compare.py`, `compare_ticker.py`, `tests/test_compare.py` and
`data.fetch_ticker_prices` are gone from `main`. The tool ran the real strategy against
buy-and-hold of any ticker on their overlap window, in three columns (levered and financed, the
same book unlevered, the ticker), with relative metrics, a tail block and crisis windows. Like
Phase 6 it worked; like Phase 6 it answers a different question than "equal risk against equal
capital across a multi-asset book."

**Reachable, not destroyed:** tag `v1.4`. `git show v1.4:src/compare.py` and
`git show v1.4:compare_ticker.py` bring it back, and `git show v1.4:src/data.py` has the ticker
fetch.

**What was kept:** `src/headtohead.py` (relative metrics, tail block, crisis table, both printed
forms) and `src/riskmetrics.py`. **Both now have no production caller** — only their own suites,
`tests/test_headtohead.py` (38 checks) and `tests/test_riskmetrics.py` (50). That is deliberate
and temporary: they are the building blocks for the matched-risk ERC-against-equal-weight
comparison this repo is being cut back to, and re-extracting them from a tag later would be
strictly worse than holding them with their tests green. Do not delete them as "unreferenced"
without reading this; equally, do not assume a caller exists when changing a signature.

**Three things from it that still matter, so they are not lost with the prose:**

- **The shared price cache can still be shrunk, and the guard against it went with the feature.**
  `cache.write_frame` replaces its whole file, so `data.fetch_prices([*UNIVERSE, extra])` on a
  short window rewrites the cached SPY/TLT/DBC/TIP history as a side effect of adding a column.
  `fetch_ticker_prices` existed precisely to avoid that (its own cache file, validated against the
  ticker's own first date) and is deleted, so **the hazard is now unguarded**. Anyone widening
  `UNIVERSE` or adding a second data source should read `data.fetch_prices` and `cache.write_frame`
  first. It is bounded and self-healing — `run_backtest.py` always requests `DATA_START_DATE` and
  checks coverage before trusting the cache — but it is a real trap, not a theoretical one.
- **"Run the whole history, then cut to the comparison window" is the rule, not an optimisation.**
  The strategy was always run over its full 2006+ history and sliced to the overlap afterwards,
  because running it only on the overlap would make its covariance warm-up — and therefore its
  weights — depend on when the other series happened to start. That is the comparison window
  leaking into the strategy's decisions. **The same rule applies to any ERC-against-equal-weight
  comparison on a sub-window**, and it is the one that is easy to get wrong while looking right.
  It is also what `TASKS.md` step 49 relies on.
- **Crisis windows are measured from the peak close**, with an uncovered window reported as `NaN`
  and printed `n/a`, never a partial window that reads like a full one. That rule lives in
  `metrics.slice_to_window` for a `BacktestResult` and is repeated in `headtohead.crisis_table`
  for bare series; both survive.

**And one measured result worth remembering,** reproducible from `v1.4`: against SPY the strategy
earned 7.1%/yr against 11.1% but drew down −29% against −55%, with a higher Sharpe (0.67 vs 0.63)
and Calmar (0.24 vs 0.20). Against AAPL it was about a third as volatile and beat AAPL on every
one of its worst 5% of days, while earning 7.1% against 28.5% at a *lower* Sharpe. It is a risk
story, not a return story — and "chosen with hindsight" always applies: a ticker someone asks
about is usually one that did well.

## The website (`web/`)

`web/` is a Vite + React research site that reads **only** what `web/export_data.py` writes, so
every figure on it traces back to the same `src/` functions `run_backtest.py` uses. It arrived on
`main` with the cleanup; before that it lived on the unmerged `web-ui` branch (PR #62), which is
superseded. The live deploy is `risk-parity-lab.vercel.app`.

**Six tabs:** Overview, Findings, The data, Benchmarks, Hardening, Method. **Four features were
removed** with the rest of the cleanup, because none of them was about portfolio construction and
three had no subject left once `src/incremental.py` and `src/compare.py` went: the position-analysis
tab and its `api/analyze.py` / `api/search.py` Python functions, a Gemini-backed research chatbot,
a passcode-gated two-person timeline workspace, and the Build log tab. `api/live.js` is the only
handler left, so `web/bundle_engine.py` and `web/requirements.txt` are gone too — there is no
Python serverless function to ship `src/` into any more.

**Read this before touching anything under `web/`:**

- **Node and npm are not installed on Person A's machine**, so the PR that removed those features
  **could not build or test the JS**. What was done instead: a static check that every relative
  import still resolves and no identifier is left imported-but-unused, plus a full live run of
  `web/export_data.py`. **Run `npm ci && npm test && npm run build` before trusting the bundle**,
  and note `web/package-lock.json` is stale until `npm install` regenerates it.
- **`web/QA.md` and `web/PROJECT_NOTES.md` are dated evidence from before the cleanup** and carry
  banners saying so. They were deliberately *not* rewritten — re-measuring needs node.
- **`web/data/*.json` is generated and checked in.** It is refreshed by CI and was refreshed by
  hand in the cleanup from a verified live run. `history.json` now feeds only the version badge in
  the header; trimming it is `TASKS.md` step 45's job.
- **Two encoding bugs in `export_data.py` were found by running it and are fixed.** Both were
  Windows-only and both pre-dated the cleanup: `subprocess.check_output(..., text=True)` and
  `Path.read_text()` with no `encoding=` decode as cp1252, so `gh api` died on byte 0x81 of a real
  pull-request body (taking the whole export down, because `UnicodeDecodeError` was not in the
  `except` clause that exists to degrade gracefully) and every em-dash in `TASKS.md` reached the
  export as mojibake — which also corrupted the phase titles, since the title regex skips leading
  non-word characters and the first byte of the mojibake is a letter. **Always pass
  `encoding="utf-8"` in this file.**
- **The site must not publish a comparison the repo cannot reproduce.** `TASKS.md` Phase 7 waits
  on Phase 8 for exactly this reason.

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
.venv/bin/python tests/test_riskmetrics.py    # VaR/CVaR, Sortino, Calmar, drawdown duration, the shared table
.venv/bin/python tests/test_headtohead.py     # correlation/beta/capture, the tail block, the crisis table
.venv/bin/python tests/fixtures.py            # print a synthetic BacktestResult summary
.venv/bin/python run_backtest.py              # full run -> reports/
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
.venv/bin/python -m src.comparison    # steps 53-54: the whole comparison from fixtures, 25 checks; offline
```

Run every suite after any change to `src/`. **All nine test suites and all eight module
self-checks run offline, with no exceptions** — verified by patching `socket.connect` /
`create_connection` / `getaddrinfo` to raise and re-running each one, not by assumption.
`src/data.py` imports `requests`, so the import graph touches the network stack, but nothing under
`tests/` opens a connection. There used to be one exception, `src.incremental`'s live-ticker
demonstration, and it went with that module — so the rule is now absolute and should stay that
way: live-data checks belong in a live run (`run_backtest.py`), not in a test, because a suite
that needs the network fails for reasons that have nothing to do with the code.

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
  into every drawdown. **The engine is worse than the first measurement said.** That measurement
  used a daily, equal-weight overlay and found the book above 1.0x on GFC 69%, COVID 54%, 2022 26%
  of each crisis window, with COVID's minimum leverage on 2020-03-23, the exact market bottom. The
  real engine only re-levers at its monthly rebalances, and measured on its own leverage (data
  through 2026-10) it was above 1.0x on **GFC 76%, COVID 100%, 2022 52%** of days: it held 2.33x,
  then 2.18x, through the entire COVID crash and only cut to 0.64x at the 2020-03-31 rebalance,
  after the trough. Quote the engine's figures, not the equal-weight ones; Phase 7's step 47
  publishes them on the site. Shortening `VOL_EWMA_HALFLIFE_DAYS`
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
