# TASKS — Project 16 build order

Step-by-step build order for the risk parity engine described in [`PLAN.md`](PLAN.md).
Every step is owned by exactly one person, is one branch, and ends in one pushed PR.

**Person A — Engine** (`abhyudayb-1`) · **Person B — Analysis** (`shivakasula`)

Work top to bottom *within your own column*. A and B steps at the same phase run
concurrently — neither waits on the other except at the two joint phases (0 and 5).

---

## The loop, every step

```bash
git checkout main && git pull                 # always start from current main
git checkout -b feat/a01-config               # branch name is given per step

# ... do the work, run the tests ...

git add -A
git commit -m "Short description of the step"
git push -u origin feat/a01-config
gh pr create --fill                           # other person reviews, then merge
```

Rules that keep this conflict-free:

- **Only touch the files listed for your step.** File ownership is the whole design.
- **`src/config.py` is append-only.** Add constants at the bottom of the relevant block;
  never reorder or reformat what's there. Name the block you touched in the PR body.
- **Never change `src/types.py` on a feature branch.** It's the contract — see *Changing the
  contract* at the bottom.
- Merge your step before starting your next one. Don't stack four branches.

---

## Phase 0 — Contract (JOINT, ~2h, both present)

Do these in order, together. **Nothing in Phase 1 starts until step 7 is merged.**

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 1 | A | `feat/a01-scaffold` | `.venv`, `requirements.txt` (pandas, numpy, scipy, cvxpy, yfinance, requests, matplotlib, seaborn), extend `.gitignore` with `.venv/`, `__pycache__/`, `data/cache/`. Create empty `src/__init__.py`, `tests/`, `reports/`. | `pip install -r requirements.txt` succeeds in a fresh venv |
| 2 | A | `feat/a02-config` | `src/config.py`: ETF universe, `START_DATE = 2006-02-03`, `TARGET_VOL = 0.10`, `LEVERAGE_CAP = (0.5, 3.0)`, `EWMA_HALFLIFE_DAYS`, `TRADING_DAYS_PER_YEAR = 252`, the three crisis windows, `PROJECT_ROOT`/`DATA_DIR`/`CACHE_DIR`/`REPORTS_DIR`. | `python -c "import src.config"` clean; every magic number in the plan appears here |
| 3 | A | `feat/a03-contract` | **`src/types.py` — the contract.** `BacktestResult` frozen dataclass (`weights`, `risk_contributions`, `leverage`, `returns`, `meta`) and the `solve_erc(cov, **kw) -> np.ndarray` signature as a `NotImplementedError` stub. Docstring each field with its index, columns, and invariant. | Both people have read it and agree. This is the most important merge in the project. |
| 4 | B | `feat/b04-cache` | Port `src/cache.py` **verbatim** from `abhyudayb-1/sp500-factor-backtester`. Do not rewrite it. | Round-trips a DataFrame to `data/cache/` and back |
| 5 | B | `feat/b05-metrics` | Port `src/metrics.py` from the same repo. **One change:** `periods_per_year` becomes a required argument, not a module constant (this project accrues daily = 252, not monthly = 12). Keep `_ZERO_VARIANCE_EPS` and the peak-floored-at-1.0 drawdown fix — both are subtle and correct. | Sharpe/CAGR/max-drawdown reproduce the sp500 repo's values on the same input |
| 6 | B | `feat/b06-fixtures` | `tests/fixtures.py`: `make_fake_result(n_assets=4, n_days=4000) -> BacktestResult` with plausible random weights (sum to 1), risk contributions (sum to 1), leverage in `[0.5, 3.0]`, daily returns. | Returns a valid `BacktestResult`; `weights.sum(axis=1)` is 1.0 everywhere |
| 7 | Both | — | Review and merge 1–6, plus `tests/test_contract.py`: proves `validate()` *rejects* malformed results, the metrics port kept its two subtleties, and the fixtures carry real regime structure. | `python tests/test_contract.py` passes. **Gate: B is now unblocked for the whole project** |

---

## Phase 1 — Data (A) ‖ Benchmarks (B)

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 8 | A | `feat/a08-prices` | `src/data.py`: yfinance download of SPY/TLT/DBC/TIP (+GLD optional), adjusted closes, through `cache.py`. Fail loudly if a ticker returns nothing — never silently drop a sleeve. | Returns a dated close matrix, 2006→present, no NaN gaps mid-series |
| 9 | A | `feat/a09-riskfree` | `src/data.py`: FRED `DTB3` via the **keyless** CSV endpoint `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3`. Convert annualized % to a daily rate. Forward-fill weekends/holidays only. | Daily risk-free series aligned to the price index |
| 10 | A | `feat/a10-calendar` | `src/data.py`: daily simple returns matrix + `month_end_rebalance_dates()` (port the shape of `month_end_trading_days()` from the sp500 repo). | Rebalance dates are all real trading days |
| 11 | B | `feat/b11-benchmarks` | `src/benchmarks.py`: `equal_weight()` (25% each) and `sixty_forty()` (60 SPY / 40 TLT), both returning `BacktestResult`. Build against `fixtures.py`. | Both produce valid `BacktestResult`s with `leverage == 1.0` |
| 12 | B | `feat/b12-maxsharpe` | `src/benchmarks.py`: `max_sharpe()` — rolling long-only tangency portfolio via cvxpy, same covariance window the engine will use. | Solves without error on a random PSD covariance; weights are non-negative and sum to 1 |
| 13 | B | `feat/b13-crisis-slices` | `src/metrics.py`: `slice_to_window()` and `summarize_window()` for the three crisis periods in `config.py`. | Returns a per-window metric row for a fixture result |

---

## Phase 2 — Solver (A) ‖ Oracle (B)

This is the mathematical core. **Step 17 is what proves step 16 is correct.**

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 14 | A | `feat/a14-ewma-cov` | `src/covariance.py`: exponentially-weighted covariance over daily returns, annualized. Halflife from `config.py`. | Reduces to sample covariance as halflife → ∞ |
| 15 | A | `feat/a15-cov-repair` | `src/covariance.py`: shrinkage toward a diagonal target, PSD repair (clip negative eigenvalues), condition-number logging. | Always returns a PSD matrix, even on near-singular input |
| 16 | A | `feat/a16-risk-contrib` | `src/riskparity.py`: `risk_contributions(w, cov)` (`RC_i = w_i·(Σw)_i/σ_p`) and `naive_inv_vol(cov)` as the comparator. | `sum(RC) == portfolio_vol` to 1e-10 |
| 17 | A | `feat/a17-erc-solver` | `src/riskparity.py`: `solve_erc()` — SLSQP minimizing `Σ(RC_i − σ_p/N)²`, long-only, `Σw = 1`, **with the analytic gradient**. Return convergence status in the result; log a warning on failure, never swallow it. | Converges on random 4×4 PSD covariances; all RCs equal to 1e-6 |
| 18 | B | `feat/b18-rc-identity` | `tests/test_riskparity.py`: the RC identity, weights summing to 1, RC fractions summing to 1. Standalone-runnable, no framework — match the sp500 repo's test style. | `python tests/test_riskparity.py` prints N checks passed |
| 19 | B | `feat/b19-oracle` | **The cross-check.** Independently implement the convex form — minimize `½wᵀΣw − (1/N)Σln(w_i)`, then renormalize — and assert A's `solve_erc()` agrees to 1e-6. Write this from the math, *not* by reading A's solver. | Two independent derivations agree on 50 random covariances |
| 20 | B | `feat/b20-degenerate` | `tests/test_riskparity.py`: perfectly correlated assets, near-singular covariance, single asset, one near-zero-vol asset. Assert ERC ≠ `1/σ` when correlations are unequal — this is the claim the project rests on. | All degenerate cases either solve or raise a clear error; none return silent garbage |

---

## Phase 3 — Overlay & Backtest (A) ‖ Stress & Charts (B)

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 21 | A | `feat/a21-vol-target` | `src/overlay.py`: trailing EWMA vol of the *current* weights' portfolio returns using only data ≤ t; `L_t = TARGET_VOL / σ̂_t`; clip to `LEVERAGE_CAP`. | Leverage rises in calm regimes, falls in volatile ones, never leaves the cap |
| 22 | A | `feat/a22-financing` | `src/overlay.py`: `r_p = L_t·(r_assets − r_f) + r_f`. **Leverage is financed, not free** — omitting this is the most common way this strategy gets oversold. | With `L = 1`, output equals the unlevered return exactly |
| 23 | A | `feat/a23-backtest` | `src/backtest.py`: monthly rebalance loop → daily accrual → populated `BacktestResult`. Only data ≤ `t` at each rebalance. | Produces a real `BacktestResult` that satisfies every contract invariant |
| 24 | B | `feat/b24-stress` | `src/stress.py`: per-crisis decomposition and the cross-strategy drawdown table (risk parity vs equal-weight vs 60/40 vs max-Sharpe vs naive 1/vol). **Include a capital-weight-in-bonds column** — it is what explains the 2022 ranking rather than just reporting it. | Emits `reports/crisis_drawdowns.csv` from fixtures |
| 25 | B | `feat/b25-diversification` | **Respecified after step 10 — read the note below.** `src/stress.py`: rolling `CORRELATION_WINDOW_DAYS` SPY/TLT correlation **for the chart**, and the **diversification ratio** `Σwᵢσᵢ / σₚ` as the actual diagnostic. A correlation-sign alarm fires in 2023, after the damage; the ratio is already falling during 2022. | Ratio reproduces the known live path: ~2.11 (2008), 1.68 (2021), 1.59 (2022), 1.52 (2023) |
| 26 | B | `feat/b26-viz-core` | `src/viz.py`: port `_style_axes`/`_new_figure`/`_title`/`_save` and the palette from the sp500 repo. Then the **risk contribution stacked area** chart. | PNG lands in `reports/` from fixture data |
| 27 | B | `feat/b27-viz-rest` | `src/viz.py`: capital-weight vs risk-weight comparison, leverage time series, rolling correlation chart. | Four PNGs total in `reports/` |

---

## Phase 4 — Hardening (A) ‖ CLI & Docs (B)

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 28 | A | `feat/a28-lookahead` | `tests/test_lookahead.py`: prove no rebalance reads future data — covariance, vol estimate and risk-free rate alike. Use synthetic data with a deliberate future spike that must be invisible at `t`. | Spike is provably not used; test fails if the `≤ t` guard is removed |
| 29 | A | `feat/a29-diagnostics` | Surface solver convergence failures and covariance condition numbers into `BacktestResult.meta`, and print a run summary. | A failed rebalance is visible in the output, not silent |
| 30 | B | `feat/b30-cli` | `run_backtest.py`: wires it end to end. Flags `--start`, `--target-vol`, `--refresh-prices`, `--max-leverage`, `--help`. | `python run_backtest.py --help` documents every flag |
| 31 | B | `feat/b31-readme` | `README.md` skeleton: what it does, module table, setup, running it, the method sections. Leave the results/2022 sections empty for Phase 5. | Someone who has never seen the repo can install and run it |

---

## Phase 5 — Integration (JOINT, ~3h, both present)

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 32 | A | `feat/a32-live-run` | First real end-to-end run on live data. Fix whatever breaks. | `python run_backtest.py` completes 2006→present |
| 33 | B | `feat/b33-real-charts` | Swap `fixtures` for the real `BacktestResult` throughout `viz.py`/`stress.py`. Regenerate `reports/`. | All charts and the drawdown table render from real data |
| 34 | Both | `feat/analysis-2022` | The write-up: correlation flip, ERC over-weighting bonds as they stop diversifying, leverage staying high into the drawdown, and the crisis table showing risk parity **losing** in 2022. State it plainly. | README's results sections complete |
| 35 | Both | — | Final review, tag `v1.0`. | Both can explain any module in the repo, not just their own |

---

## Data reality check — read before steps 24, 25 and 34

Steps 8–10 are merged, so the engine's data layer now runs on **live** data, and it corrected
one of this project's founding assumptions. The original plan (and an earlier version of the
fixture) said the 2022 failure was a stock-bond correlation flipping positive. It was not:

- Across `2022-01-03 → 2022-10-12` the rolling 63-day SPY/TLT correlation **averaged −0.09**
  and was positive on only **31% of days**. The persistent flip is **2023 onward** (66% / 65%
  / 70% of days in 2023 / 2024 / 2025).
- What actually happened: **TLT fell 29.3% against SPY's 24.5%** — the hedge lost more than
  the thing it hedged — while **DBC returned +21.4%**, the only sleeve that worked and the one
  equal-risk-contribution holds least by capital. These are peak-to-trough, measured from the
  window's opening close; compounding the returns stamped *on* `[start, end]` measures from the
  day before the peak and reads −31.2% / −24.1% / +21.9% instead.
- The **diversification ratio** `Σwᵢσᵢ / σₚ` degrades right through the episode: 2.11 (2008),
  1.78 (2020), 1.68 (2021), 1.59 (2022), 1.52 (2023). That is why step 25 keys off it.

**Two consequences for Person B:**

1. Do not write a "correlation crossed zero" alarm. It fires a year late. See step 25.
2. **The fixtures are shape-accurate, not magnitude-accurate.** `tests/fixtures.py` now
   calibrates each crisis profile to the real windows above, but every number it produces is
   still synthetic and one random draw. Build and test against it freely; never quote its
   output as a finding. Anything that goes in the README comes from a live run in Phase 5.

A related trap worth internalising, because it cost two test rewrites already: **do not assert
on a single simulated path.** Over a 0.78-year window at ~24% annualized volatility, one
path's total return carries a ~21% standard deviation, so sleeve orderings flip seed to seed.
A realized-path diversification-ratio check failed on 3 of 15 seeds. Assert on the *parameters*
(`_CRISIS_PROFILES`, the correlation tables) instead — equally informative, never flaky.

## Ground rules

**Review each other's PRs.** Not a formality — it's a capstone, and both of you need to
explain the *whole* system in an interview, not half of it. If B has never read the
vol-targeting overlay, B can't answer why leverage stayed high into 2022, which is the most
likely question either of you will be asked about this project.

**Changing the contract.** `src/types.py` will need to change once or twice. When it does:
say so *before* writing code, then open **one** PR that changes the contract and both call
sites together. Never unilaterally on a feature branch. If it changes more than twice, it was
cut wrong — spend 30 minutes together re-cutting it rather than patching.

**If you finish your column early,** don't start the other person's steps. Write more tests,
or start reviewing. Two people editing one file is what this whole structure exists to avoid.

**Later:** UI and backend get their own plan after step 35. Don't build toward them now — the
contract in `src/types.py` is the seam a dashboard would attach to, and it's already in place.

---

## Phase 6 — Incremental position analysis (JOINT, starts only after v1.0 is tagged)

**Nothing in Phases 0–5 is superseded by this phase — there is nothing to delete.** Every step
1–35 still builds the deliverable those phases describe (the four-sleeve engine and its v1.0
write-up), and this phase does not touch `types.py`, `backtest.py`, `overlay.py`,
`benchmarks.py`, `stress.py`, `viz.py` or `run_backtest.py`. It is a new, independent consumer
of `covariance.py` and `riskparity.py`, the same way `stress.py` is a consumer of
`BacktestResult` — see PLAN.md's "Phase 6" section for the full design and the math.

**Do not start this phase until step 35 has tagged `v1.0`.** The pivot that motivated it — "let
someone add any stock to the risk-parity book and see whether it's compensated, not whether it
belongs as a fifth sleeve" — is additive scope, and the original capstone deliverable ships on
its own timeline first.

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 36 | A | `feat/a36-incremental-config` | `src/config.py`: append `INCREMENTAL_LOOKBACK_DAYS` (~756 trading days), `INCREMENTAL_CANDIDATE_WEIGHTS` (`(0.01, 0.05, 0.10)`), `INCREMENTAL_MIN_HISTORY_DAYS`. | Constants exist at the bottom of the file; nothing else in `config.py` reformatted |
| 37 | A | `feat/a37-incremental-core` | `src/incremental.py`: fetch the candidate + universe on the trailing window (any ticker — `data.fetch_prices` already supports this); baseline 4-asset ERC snapshot on that same window; the perturbation scan at `INCREMENTAL_CANDIDATE_WEIGHTS`; the unconstrained 5-asset `solve_erc_with_status` re-solve and its diff against baseline weights. Reuse `covariance.estimate_covariance` and `riskparity`'s primitives — no new statistical machinery. | Runs end to end against a real live ticker; raises `DataUnavailableError` (not a silent empty result) when the candidate has less than `INCREMENTAL_MIN_HISTORY_DAYS` of history |
| 38 | A | `feat/a38-compensation-test` | `src/incremental.py`: the compensation test `SR_i > SR_p · ρ(i, p)` (derivation in PLAN.md) plus the candidate's correlation to each of the four sleeves individually. | Verified against a hand-derivable two-asset synthetic case where the sign of the inequality is known in advance |
| 39 | B | `feat/b39-incremental-tests` | `tests/test_incremental.py`: perturbation weights sum to 1 at every candidate weight; the compensation-test formula regression; the 5-asset re-solve's limiting case (a candidate identical to an existing sleeve must split that sleeve's risk contribution); correlation-table correctness. Standalone script, no framework, matches the house style. | `python tests/test_incremental.py` prints `N checks passed` and exits 0 |
| 40 | B | `feat/b40-incremental-cli` | `analyze_position.py`: `--ticker` (required, any ticker), `--lookback-days`, `--weights`, `--refresh-prices`, `--help`. Formatted report output in the style of `backtest.format_run_summary`, naming plainly that this is a point-in-time snapshot, not a historical backtest. | `analyze_position.py --help` documents every flag; a run against a real live ticker prints a complete report including the caveats in PLAN.md |
| 41 | Both | `feat/incremental-writeup` | `README.md`: new section covering what the tool answers, a worked example against one real stock, and the explicit caveats (point-in-time, marginal test only, idiosyncratic risk is not a structural risk premium). | Someone who has never seen the tool can run it and correctly interpret what its output does and does not claim |
| 42 | Both | — | Final review, tag `v1.1`. | Both can explain the compensation-test derivation and every new module, not just their own |

---

## Phase 7 — Website feature parity (JOINT, starts only after PR #62 is merged)

This is the "UI and backend get their own plan" the ground rules above defer to. The live site
(`risk-parity-lab.vercel.app`) is built by PR #62's `web/` tree, and a gap audit against `main`
found that it shows less than the project computes: no unlevered-ERC or financing-omitted rows, no
overlay-lag numbers, the realized-risk and head-to-head metrics hidden or shown as raw JSON, and
none of the CLI's settings (`--start`, `--target-vol`, `--max-leverage`, `--lookback-days`,
`--weights`). Steps 43 and 44 are recorded in `CLAUDE.md`'s ledger rather than here, so this phase
numbers from 45.

**Nothing under `src/` changes in this phase.** All new numbers are produced by
`web/export_data.py`, which already calls the real engine (`load_data`, `run_backtest`,
`build_benchmarks`, `compare_to_ticker`), so the site has exactly one source for every figure.
`web/` is B's area; step 50's equivalence tests go to A, the way step 28 did, because they guard the
engine's own alignment rule.

Four measurements shape the design, and are the reasons it is not built the obvious way:

- **An engine run takes 0.5 s** (benchmarks 2.1 s, data load 1.3 s, measured on the full sample) —
  not the "minutes of compute" `export_data.py`'s comment claims. Precomputing many runs is cheap.
- **Leverage is `clip(target_vol / trailing_vol, cap)`, and `trailing_vol` depends only on the
  weights.** Neither the target nor the cap feeds back into the ERC weights, so those two settings
  can be re-applied in the browser with no engine run at all.
- **`results.json` is 2.7 MB and served `no-cache`.** About 2 MB is daily equity plus drawdown
  series, and drawdowns are derivable from equity. Fix the payload before adding to it.
- **Every missing realized-risk and comparison metric is already exported** (Calmar, skew, excess
  kurtosis, worst day, longest drawdown and its open flag, beta, tracking error, capture, the tail
  block, `calendar_warning`). Those steps are UI only.

**Checked before any step started.** Every claim the steps below rest on was measured against the
real engine and PR #62's export, and the steps were revised to match: a reference implementation of
the browser overlay reproduced `run_backtest` exactly (0.0 difference) at the defaults and at four
non-default settings, with the weights and trailing vol bit-identical across all of them; the
proposed `core.json` is 3.8 KB gzipped against 649 KB for today's `results.json`; a later start date
changes the weights only during its first year; and the engine's own crisis leverage differs sharply
from the equal-weight figures `CLAUDE.md` used to quote (step 47). The numbers are in each step.

| # | Who | Branch | What | Done when |
|---|---|---|---|---|
| 45 | B | `feat/b45-web-payload` | `web/export_data.py` + the data loader: split `results.json` into `core.json` (metrics, financing, crisis, mechanisms, run summary, config — everything above the fold) and lazily loaded `series.json` (via the existing `DeferredChart`); drop the drawdown series and derive it client-side from equity; store series columnar (one shared date array, values rounded to 7 significant figures); move the raw `diagnostics` / `benchmark_diagnostics` dumps into their own lazy file; content-hash every data file and serve it `public, max-age=31536000, immutable`, leaving only `manifest.json` as `no-cache`. Also correct the "minutes of compute" comment. Measured: `core.json` 3.8 KB gzipped (today's `results.json` is 649 KB); `series.json` ~122 KB gzipped at 7 significant figures, which is fine because it is lazy. **Table figures (max drawdown and the like) come from `core.json`, computed in Python — never re-derived from rounded chart series.** | First-load data under 100 KB gzipped; a test proves every chart value equals the old export to 1e-6; the client-side drawdown matches the exported one to 1e-6 (it is exact from full-precision equity, and 9e-7 off from 7-significant-figure equity, so 1e-12 is the wrong bar for chart data); Lighthouse in `web/screenshots/` shows no LCP/TBT regression |
| 46 | B | `feat/b46-web-metrics` | Show what is already exported. Realized-risk tab: the full metric set (Calmar, skew, excess kurtosis, worst day, downside deviation, longest drawdown with an open badge, drawdown peak/trough dates) and a change-vs-baseline table **exported from Python** (same rows the CLI prints), not recomputed in JS. Comparison tab: replace the raw-JSON `<details>` with relative-metrics, tail-block ("beat the ticker on X% of its worst 5% of days") and crisis tables; uncovered windows render `n/a`, never 0. A visible banner whenever `calendar_warning` is set, naming the match and the US-ADR route. Undefined metrics render `--`. | A Node test renders both tabs from `candidates.json` and asserts every exported field appears; a foreign-listing fixture (e.g. `7203.T`, 96.6% match) shows the banner; no `Infinity`/`NaN` reaches the DOM |
| 47 | B | `feat/b47-web-headline` | The README's headline findings. Benchmark table gains **Unlevered ERC** and **Financing omitted** rows (already in `financing`) and a **Total return** column (already in `summarize`), the three financing rows visibly grouped. `export_data.py` computes, per crisis window, the share of days held above 1.0x leverage and the minimum leverage with its date, shown on the crisis cards. **Compute these from the engine's own leverage, which changes only at the monthly rebalances.** Measured on the engine (data through 2026-10): GFC above 1.0x on 76% of days, minimum 0.79x from 2009-01-02; **COVID on 100%** — it held 2.33x, then 2.18x, through the whole crash and only cut to 0.64x at the 2020-03-31 rebalance, after the 03-23 trough; 2022 on 52%, minimum 0.82x. The 69% / 54% / 26% that `CLAUDE.md` used to quote came from a daily, equal-weight overlay and must not be reused. | A test recomputes the levered-share figures with pandas directly from `engine.leverage` and matches; on the same data they reproduce the figures above; the new rows match `README.md`'s Results table at its printed precision, or README is refreshed to the same data date in the same PR |
| 48 | B | `feat/b48-web-evidence` | The verification on the site, from a real run. CI already runs the suites: have the export record each suite's `N checks passed` line and the solver-oracle agreement figure into the manifest, and render them in the guardrails section, each linked to its test file. | The figures shown are read from that CI run's output, never typed in; a failing suite blocks the export rather than publishing a stale pass |
| 49 | B | `feat/b49-web-explorer` | Strategy settings explorer. **Target vol and leverage cap in the browser:** export daily unlevered returns, the daily risk-free rate and each rebalance's `trailing_vol` (measured 53 KB gzipped at 10 significant figures, lazy); JS computes `L = clip(t/σ̂, [lo, hi])` per rebalance, applies it **strictly before** each return date (the one-day rule), then `L·r_a − (L−1)·r_f` and the metrics, on every slider input. **Start date by slicing, not re-running:** measured at starts in 2010, 2015 and 2019, a later start changes the weights only in its first year (at most 1.2 pp), they converge to ~1e-5 within two years, and CAGR over the overlap matches the full-history run to four decimals. So the explorer cuts the full-history run to the chosen date range in the browser and says so — the same "run the whole history, cut afterwards" rule step 44 uses — with no per-start files. **Slider bounds mirror the engine's own validation:** target vol > 0, leverage floor > 0 (`overlay` raises on a floor of 0: "the overlay may hold no position at all, which is a different strategy"), floor ≤ ceiling. The explorer's result is always shown *beside* the published one, never in place of it. Also have the export write a fixture of Python `run_backtest(target_vol=…, leverage_cap=…)` at 2–3 non-default settings, for step 50. | Every slider update completes in under 16 ms (the reference implementation takes 0.1 ms); a setting the engine would reject cannot be entered; the published numbers never change when the explorer moves |
| 50 | A | `feat/a50-web-explorer-tests` | Equivalence tests for step 49's browser overlay, owned by A the way step 28 was, because they guard the engine's own alignment rule. | JS at the default settings reproduces `engine.returns` to 1e-10 (the measured reference matches exactly at full precision and to 1.1e-11 from the 10-significant-figure payload); JS matches step 49's Python fixture at every non-default setting (the reference matched exactly at 6%, 15%, a 1.5x cap and a 0.25–2.0x cap); **switching the JS to "at or before" makes the test fail**, and the run differs on **exactly the rebalance dates and no others** (measured: all 235) — the `test_lookahead.py` mutation technique, with the date set as the positive control |
| 51 | B | `feat/b51-web-analyzer` | Analyzer settings and any-ticker speed. `/api/analyze` accepts `lookback` (252–1,260) and `weights` (≤ 5 candidate sizes in (0, 0.5], no duplicates), validated by the same rules as `analyze_position.py`, passed straight to `analyze_position(lookback_days=…, candidate_weights=…)`. Successful responses send `Cache-Control: public, s-maxage=86400, stale-while-revalidate=604800` keyed on the normalised ticker + parameters; unavailable ones send `no-store`, so a yfinance outage is never cached for a day. Measure the Python function's cold start first: locally an uncached ticker analyzes in 0.8 s and a cached one in 0.09 s, so the analysis itself is cheap and the serverless cold start (importing pandas/scipy) is the latency that matters. Only if that is slow, precompute ~25 common tickers (mega-caps, SPY/QQQ/GLD, a recent IPO, a US-traded ADR) in the export; if it is over 3 s, show a progress state saying what is being computed. | Invalid parameters return 400 with the CLI's message; a repeat query is served from the edge cache; a short-history IPO renders its shrunk-window note; cold-start latency is recorded in `web/QA.md` |
| 52 | Both | — | Final review of steps 45–51, tag `v1.5`. | Every number on the site traces to a function in `src/` through `export_data.py`, and both can explain how the browser explorer is proven equal to the engine |

Steps 46–48 depend only on 45 and can run in parallel; 49 needs 45's split files and 50 needs 49;
51 is independent of 46–50.

---
