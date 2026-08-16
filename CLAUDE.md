# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build ledger — what is done, at a glance

**`git log --oneline` is the truth; this table is a convenience that goes stale.** Commit subjects
are `Step NN [A|B]: ...`, so the log reads as the same ledger. Step numbers, owners and the full
"done when" for each row live in `TASKS.md` — that text is the spec, not this summary.

**Done: steps 1–27, all on `main`** — the whole engine and the whole analysis layer.

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

Follow-up fixes that landed on top of those steps and are worth knowing about: `361c995` (reject a
non-PSD covariance in `solve_erc` rather than solving it), `d61e7bd` (reject out-of-order rows in
both EWMA estimators), `bc39670` (measure crisis windows from the peak close), `b9916a8` /
`36a275c` (correct the 2022 thesis against live data), and `bc7c36f` / `1491e6b` / `c5f9c5f`
(audit passes on Phase 0, Phase 2 and B's step 24–27 branch).

**Remaining:** Person A has **28–29** (`tests/test_lookahead.py`; diagnostics into `meta` + the run
summary). Person B has **30–31** (`run_backtest.py` CLI, `README.md` — currently an empty stub).
Then **Phase 5, 32–35**: live run, real charts, the 2022 write-up, tag `v1.0`. A's and B's tracks
are parallel and neither blocks the other.

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

**The engine runs end to end on live data.** 234 monthly rebalances, 2007-02-28 → 2026-07-31,
4896 return days: **CAGR 7.13%, vol 11.26%, Sharpe 0.67, max drawdown −29.10%**, mean leverage
1.76x, **0 solver failures, 0 covariances repaired, worst risk-contribution deviation 1.6e-07**.
Those are the headline numbers until Phase 5 re-runs them; treat them as current, not final.

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

To resume in a new session: read `TASKS.md`, find the lowest-numbered unfinished step for your
owner, and build it. Nothing else needs reconstructing — the contract in `src/types.py` and the
fixtures make each step independently buildable.

Two stand-ins in `src/benchmarks.py` are now superseded and must be retired in Phase 5:

- `_ewma_cov` → `src.covariance.estimate_covariance`. Injectable rather than hard-wired: pass the
  real estimator via `cov_estimator=` at the call site — all four benchmarks accept it — and do not
  fork a second copy.
- `_nearest_psd` → `src.covariance.nearest_psd`. Note these two are **not** equivalent, so swapping
  is a behaviour change and not a rename. `_nearest_psd` clips eigenvalues of the covariance
  directly, which shifts the diagonal and therefore quietly alters asset volatilities;
  `covariance.nearest_psd` repairs in correlation space and leaves them intact. The real one is
  what `max_sharpe` should be judged on, since a tangency portfolio is sensitive to exactly those
  volatilities.

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
.venv/bin/python -m src.viz           # the four charts -> reports/, from fixtures
```

Run every suite after any change to `src/`. **All six suites and all seven module self-checks run
offline** — verified by blocking `socket.connect` / `getaddrinfo` and re-running each one, not by
assumption. `src/data.py` imports `requests`, so the import graph touches the network stack, but
nothing under `tests/` or any `__main__` opens a connection. Keep it that way: live-data checks
belong in the Phase 5 run, not in a test, because a suite that needs the network fails for reasons
that have nothing to do with the code.

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
result and re-renders; no call site changes. `reports/` is gitignored until then, because a
committed fixture chart is indistinguishable from a real one to anyone browsing the repo.

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
  conclusion here, which is why the omission is defensible rather than why it is invisible.

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
