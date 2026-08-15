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
