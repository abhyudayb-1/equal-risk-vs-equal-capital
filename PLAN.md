# Equal risk vs equal capital — design plan

*A multi-asset portfolio-construction study, built by two people in parallel. Originally
"Project 16 — Risk Parity & Volatility-Targeting Engine"; see Context for what changed and why.*

## Context

**The question:** across a multi-asset book, what does allocating equal *risk* buy you over
allocating equal *capital*, and what does it cost in volatility and return?

The two construction rules under comparison are **equal risk contribution (ERC)** and **equal
capital (1/N)**, over four asset classes, daily from 2006, with naive 1/σ, max-Sharpe and 60/40 as
reference points. A volatility-targeting overlay is part of the machinery but **not** part of the
thesis: two books at different volatilities cannot be compared on return, so the overlay exists to
put ERC at 1/N's own volatility, and the financing it charges is one of the costs being measured.
Stress tests through 2008, 2020 and 2022 are the evidence that the covariance-estimated
diversification ERC depends on is not stable, and that the overlay's lag is expensive.

The write-up must show the unflattering half as plainly as the flattering one — see the Phase 8
section below for the measured answers, which include equal risk *losing* to equal capital once the
leverage that makes the comparison fair is paid for.

**This was not the original framing.** Through `v1.4` the project was "a risk-parity +
volatility-targeting engine whose analytical payload is why risk parity failed in 2022", and two
further products were layered on it (incremental position analysis, head-to-head against a ticker).
Both were removed and the framing narrowed to the question above; `AGENTS.md`/`CLAUDE.md`'s ledger
records what went and which tags still hold it. One claim from that era was **refuted by this
repo's own measurement and must not be reintroduced**: the 2022 failure was *not* a stock-bond
correlation flipping positive. Over `2022-01-03 → 2022-10-12` the rolling 63-day SPY/TLT
correlation averaged **−0.09** and was positive on only 31% of days. What happened is that TLT fell
29.3% against SPY's 24.5% — the hedge lost more than the thing it hedged. See the 2022 section
below and the "Data reality check" in `TASKS.md`.

Two people build it in parallel on separate machines. The hard part of a 2-person split on a
quant pipeline is that the analysis half normally can't start until the engine half produces
output. This plan removes that dependency with a **frozen result contract + synthetic fixture**
on day one, so both people are unblocked from hour 2 onward.

**Home:** `shivakasula/portfolio-allocation-capstone-project` (private; the repo was created under
`abhyudayb-1` and later transferred, so older notes name the old owner). Working copy on Person A's
machine is `D:\Project 15`; local `main` tracks `origin/main`.

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
src/riskmetrics.py  VaR/CVaR, Sortino, Calmar, drawdown duration, the shared metric table
src/headtohead.py   one return series against another: correlation, beta, capture, tail block
src/benchmarks.py   equal-weight, 60/40, max-Sharpe (cvxpy), naive 1/vol risk parity
src/stress.py       crisis decomposition, drawdown table, correlation-regime diagnostics
src/viz.py          the report charts: four PNGs (the crisis table is the fifth deliverable, a CSV from stress.py)
run_backtest.py     entry point wiring it end to end
tests/fixtures.py   synthetic BacktestResult generator — B's unblocking mechanism
tests/test_riskparity.py   ERC math + B's independent convex oracle
tests/test_lookahead.py    proves no rebalance reads future data
```

`riskmetrics.py` and `headtohead.py` were salvaged out of two removed products and **have no
caller yet**; Phase 8 is what wires them in. See their module docstrings before deleting either as
unreferenced.

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
.\.venv\Scripts\python.exe run_backtest.py --start 2021-01-01 --no-charts        # fast smoke
```

Done means: `Σ RC_i = σ_p` holds to 1e-10; SLSQP and convex solutions agree to 1e-6; every
rebalance provably uses only past data; the crisis table reproduces the **measured** rankings —
and note that an earlier draft of this line predicted the wrong ones. Risk parity beats 60/40 in
2008 (−25.4% vs −29.9%) and 2022 (−17.7% vs −25.8%), but it is the **worst of all five strategies
in COVID at −29.10%**, because the overlay walked in at 2.33x. "Risk parity wins the crises" is not
what the data says. Finally, the report artifacts land in `reports/` — four PNGs (RC
stacked area, capital-vs-risk weight comparison, leverage time series, and the rolling correlation
drawn with the diversification ratio) plus the crisis drawdown table as a CSV. (Planned as five
charts; the correlation and diversification ratio share one figure, and the table is a CSV.)

---

## Phase 8 — The matched-risk ERC vs 1/N comparison (the current work)

Everything above describes the engine. This phase is the comparison the project now exists to
make, and it is the first phase whose numbers are already measured but **not yet reproducible from
anything in `src/`** — which is the gap step 53 closes. `TASKS.md` Phase 8 has the numbered steps;
this section is the design and the measurements that constrain it.

### What has to be compared, and why three columns rather than two

| Column | What it is | Why it is there |
| --- | --- | --- |
| **1/N** | equal capital, 25% a sleeve, rebalanced monthly | the naive rule being argued against |
| **ERC unlevered** | equal risk contribution, no leverage | the allocation claim on its own, at its own volatility |
| **ERC at 1/N's volatility** | the same ERC weights, levered by the overlay to 1/N's realized vol, **financing charged** | the only column whose *return* is comparable to 1/N's |

Two reference rows stay alongside: **naive 1/σ** (risk parity without correlations — it is the
control that says how much of ERC's edge comes from the solver at all) and **60/40**.

A fourth, diagnostic row is required and must be labelled as counterfactual: **ERC at 1/N's
volatility with the financing leg omitted.** It is not a strategy anyone can run. It exists because
the difference between it and the row above is the cost of leverage, and that difference is one of
the three findings.

### Measured already, by a scan, and the implementation must reproduce these

Run on live data through **2026-10-30**, 237 rebalances, 4,950 return days, using `run_backtest`,
`benchmarks.equal_weight`, `overlay.unlevered_returns` / `financed_returns` and
`metrics.summarize` — no new estimator, no hand-rolled alignment:

| | CAGR | vol | return/vol | max drawdown | mean leverage |
| --- | --- | --- | --- | --- | --- |
| ERC unlevered | 5.12% | 6.90% | **0.759** | −17.15% | 1.00x |
| Naive 1/σ | 4.86% | 6.64% | 0.747 | −16.69% | 1.00x |
| 1/N (equal weight) | 5.79% | 8.35% | **0.716** | −28.09% | 1.00x |
| Max Sharpe | 6.73% | 10.48% | 0.674 | −26.59% | 1.00x |
| 60/40 | 8.59% | 11.44% | 0.778 | −29.92% | 1.00x |
| ERC @ 10% target, financed | 7.14% | 11.25% | 0.669 | −29.10% | 1.76x |
| ERC @ 10% target, financing omitted | 8.32% | 11.25% | 0.767 | −29.05% | 1.76x |
| **ERC @ 1/N's vol, financed** | 6.28% | 9.40% | **0.696** | −24.85% | 1.47x |
| ERC @ 1/N's vol, financing omitted | 7.00% | 9.40% | 0.767 | −24.81% | 1.47x |

**Three findings, and the second is the one the project leads with:**

1. **Equal risk is a better risk allocation.** ERC hits 25% risk a sleeve to 1.6e-07 while its
   capital runs SPY 20.0% / TLT 21.8% / DBC 16.3% / TIP 41.9%. 1/N holds capital flat and lets risk
   go where it likes: SPY's risk share spans **8.9% to 60.3%** (mean 32.9%), DBC's averages
   **41.8%**, TIP's **9.0%**, and TLT's turns **negative** (−6.3%) in some months. Worst
   single-sleeve risk concentration 1/N reaches: **66.5%**. In effective bets (`1 / Σ RCᵢ²`),
   **ERC 4.00 against 1/N 2.83** out of a possible 4.00.
2. **Equalizing risk wins on risk-adjusted return until the leverage is paid for.** Unlevered,
   0.759 against 0.716 and a drawdown of −17.2% against −28.1%, for 67 bp/yr less return. Levered
   to 1/N's own volatility with the real T-bill charged: **0.696, below 1/N**. With the financing
   leg omitted: 0.767, above it. State both.
3. **Most of ERC's edge is the volatility scaling, not the correlation structure.** Naive 1/σ
   reaches 0.747 of ERC's 0.759 with no optimizer, no gradient and no restart ladder. The ERC
   solver is worth about one point of return-to-vol over inverse-volatility weighting on this
   universe. That is a real result about this project's own mathematical core and it is not
   flattering to it.

### Two measurement traps, both already hit once

- **The overlay overshoots its target by a consistent ~12.5%.** A 10% target realized 11.25%; an
  8.355% target realized 9.40%. So "ERC at 1/N's volatility" via the overlay lands at 9.40%, not
  8.35%, and the comparison is approximately rather than exactly matched. The implementation must
  either say so plainly next to the table or solve for the target that realizes 1/N's vol. Do not
  describe it as exactly matched.
- **Re-running the engine at a different `target_vol` does not change the weights.** Verified:
  `m.weights.equals(eng.weights)` is `True`. The target and the cap feed the leverage path only,
  never the ERC solve, which is what makes a matched-risk column cheap — one extra engine run, not
  a redesign. It is also why `TASKS.md` step 49's browser explorer is sound.

### What this phase must not do

- **Do not hand-roll the overlay or the one-day alignment** to build the matched column. Call
  `run_backtest(returns, risk_free=rf, target_vol=...)` and let `backtest.py` own it — see
  `AGENTS.md`/`CLAUDE.md` under `backtest.py`.
- **Do not run a rule only on a sub-window** and compare. Run full history, slice afterwards; a
  shorter start changes the covariance warm-up and therefore the weights.
- **Do not reuse `README.md`'s 2026-07-31 figures in the same table as these.** One data date per
  table, named.
