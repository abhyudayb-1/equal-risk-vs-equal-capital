# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Project 16 — a risk-parity + volatility-targeting portfolio engine over four ETF sleeves
(SPY / TLT / DBC / TIP), backtested 2006→present through the GFC, COVID and the 2022 rate shock.
The deliverable is not just the engine: the analytical payload is **why risk parity failed in 2022**.
Results that make the strategy look bad are stated plainly, not hidden.

Note the mechanism carefully, because the obvious version is wrong and this repo already corrected
it once against live data: the stock/bond correlation did **not** flip positive during the 2022
drawdown. Over `2022-01-03 → 2022-10-12` the rolling 63-day SPY/TLT correlation averaged **−0.09**
and was positive on only 31% of days; the persistent flip is 2023 onward. What happened is that
**TLT fell 31.2% against SPY's 24.1%** — the hedge lost more than the thing it hedged — while
correlation merely rose toward zero, which was enough to remove the offset. See PLAN.md's 2022
section and the "Data reality check" in TASKS.md before writing anything about 2022.

`PLAN.md` holds the design rationale and the pinned math. `TASKS.md` holds the numbered build
order — every step names one owner, one branch, and a "done when". Read the relevant step before
starting work; the step text is the spec.

## Where the build is

**Check `git log --oneline` and `gh pr list` before trusting this section — it goes stale, they do
not.** As of the last edit, Phase 0 and steps 8–11 and 13 were on `main`, with step 12 (`max_sharpe`)
still in an open PR.

Next up: **steps 14–17** for Person A — EWMA covariance, shrinkage and PSD repair, risk
contributions, and the SLSQP ERC solver — then **18–20** for Person B.

Step 19 is the highest-value test in the project: Person B's independent convex log-barrier oracle,
asserting Person A's SLSQP solver to `ERC_AGREEMENT_TOLERANCE`. It must be written **from the
maths, not by reading step 17** — two independent derivations are the entire point, and peeking
collapses them into one.

To resume in a new session: read `TASKS.md`, find the lowest-numbered unfinished step for your
owner, and build it. Nothing else needs reconstructing — the contract in `src/types.py` and the
fixtures make each step independently buildable.

Two stand-ins exist that Phase 5 must replace, both in `src/benchmarks.py` and both deliberately
injectable rather than hard-wired: `_ewma_cov` (superseded by step 14's `src/covariance.py`) and
`_nearest_psd` (superseded by step 15). Pass the real estimator via `cov_estimator=` at the call
site; do not fork a second copy.

## Setup and commands

System Python has no numpy on this machine — always work inside a venv.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python tests/test_contract.py      # Phase 0 gate: contract + metrics port + fixtures
.venv/bin/python tests/test_data.py          # data layer: windowing, calendar, look-ahead guard
.venv/bin/python tests/test_benchmarks.py    # equal weight / 60-40 / max Sharpe
.venv/bin/python tests/fixtures.py           # print a synthetic BacktestResult summary
.venv/bin/python run_backtest.py             # full run -> reports/   (exists from step 30)
```

Run all three suites after any change to `src/`. Only `test_data.py` and `test_contract.py` are
network-free; the suites must keep passing offline, so live-data checks belong in the Phase 5 run
rather than in a test.

Tests are **standalone scripts with no framework** — no pytest, no test runner. Each file defines
`test_*()` functions plus a `check(label, condition)` helper, and a `__main__` block calls them in
order and prints `N checks passed`. Run a single test file by running that file; run a single test
by calling its function in the `__main__` block. New test files must match this shape, including
the `sys.path.insert(0, ...)` preamble that makes `from src import ...` work when run from the
repo root.

PLAN.md and TASKS.md quote PowerShell paths (`.\.venv\Scripts\python.exe`) because Person A is on
Windows. On this machine use `.venv/bin/python`.

## Architecture

One module per pipeline stage under `src/`, wired end to end by `run_backtest.py`:

```
config.py → data.py → covariance.py → riskparity.py → overlay.py → backtest.py → BacktestResult
                                                                                       ↓
                                                          metrics.py / benchmarks.py / stress.py / viz.py
```

`src/__init__.py` carries the authoritative stage list. Modules past `backtest.py` in that list do
not exist yet — see TASKS.md for which step creates each.

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
`tests/test_lookahead.py` (step 28) exists to attack that one function.

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
  genuinely must be tested.
- Bugs found by review get a regression check in the matching suite, not just a fix. Every bug
  fixed here so far was silent — passing tests, plausible output, wrong behaviour — so "it runs and
  the number looks reasonable" is not evidence of anything.
- Comments here explain *why* a subtle choice is correct (the ones that would otherwise get
  "simplified" away). Match that density — this is a capstone that gets read and defended, not
  just run.
- Ownership split: Person A (`abhyudayb-1`) owns the engine modules, Person B (`shivakasula`, this
  machine) owns metrics/benchmarks/stress/viz/tests/CLI/README. Only touch the files listed for the
  step you are on. Branches are `feat/a<NN>-slug` / `feat/b<NN>-slug`, one step per branch, merged
  before the next starts.
