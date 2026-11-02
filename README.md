# Equal Risk vs Equal Capital

A systematic portfolio engine over four ETF sleeves — **SPY** (US equities), **TLT** (long
Treasuries), **DBC** (commodities) and **TIP** (inflation-linked bonds) — backtested daily from
February 2006 to the present, through the 2008 financial crisis, the March 2020 COVID crash and
the 2022 rate shock.

The question the project asks is narrow: **what does allocating equal *risk* buy you over
allocating equal *capital*, and what does it cost in volatility and return?** The two construction
rules being compared are **equal risk contribution (ERC)** and **equal capital (1/N)**, with naive
1/σ, max-Sharpe and 60/40 as reference points.

The engine does two things a conventional allocator does not:

1. **It allocates risk, not capital.** A 25%-each portfolio is not a balanced portfolio. Equities
   run roughly three times the volatility of TIP, so equal capital means equities dominate the
   portfolio's variance while the pie chart looks perfectly diversified. The engine instead solves
   each month for the weights at which every sleeve contributes an *equal share of portfolio
   volatility* — the **equal risk contribution (ERC)** portfolio.
2. **It can then put that book at any volatility you ask for.** ERC decides how risk is *split*; it
   says nothing about *how much*. A bond-heavy equal-risk book left alone runs far less volatility
   than equal capital does, so a **volatility-targeting overlay** levers the whole portfolio to a
   chosen level, subject to a leverage cap, with the borrowed portion charged at the 3-month T-bill
   rate.

**Point 2 is a measuring instrument, not a second product.** Two books at different volatilities
cannot be compared on return at all, so the overlay exists to put ERC at 1/N's *own* volatility and
make the comparison mean something — and the financing it charges is then one of the costs being
measured rather than a footnote. Omitting that term is the single most common way this strategy
gets oversold, and it is worth 1.10%/yr here.

**Three findings, measured, and the second one is not kind to risk parity:**

1. **Equal risk is a better risk allocation than equal capital, and it is not close.** ERC holds
   every sleeve at 25.0% of portfolio volatility to within 1.6e-07 while its *capital* runs from
   16% to 42% a sleeve. 1/N holds capital flat and lets risk go wherever volatility takes it: SPY's
   risk share spans **8.9% to 60.3%**, DBC's averages **41.8%**, and in twelve months TLT's turns
   *negative*. In effective bets — `1 / Σ RCᵢ²`, the reciprocal Herfindahl of the risk shares —
   that is **ERC 4.00 against 1/N 2.92**, out of a possible 4.00.
2. **And it still loses on risk-adjusted return once the leverage is paid for.** Unlevered, ERC
   scores **0.756** return-to-vol against 1/N's **0.712** and nearly halves the drawdown
   (−17.2% against −28.1%) — for 66bp/yr less return. Lever it to 1/N's own volatility so the
   returns are comparable and charge the real bill rate, and it scores **0.692 — below 1/N.** With
   the financing leg omitted it scores 0.763, above it. That pair of numbers is the sharpest result
   in the repo.
3. **Most of what equal risk buys comes from the volatility scaling, not the correlations.** Naive
   1/σ — inverse volatility, no optimizer, no gradient, no solver — reaches **0.745** of ERC's
   0.756. The correlation structure the ERC solver exists for is worth about a point of
   return-to-vol on this universe, not the whole gap. That is a result about this project's own
   mathematical core and it is reported as plainly as the others.

Both `weights` and `risk_contributions` are carried through the whole pipeline side by side,
because their divergence *is* the subject. The crisis chapters below are evidence for those three
findings and for their limits, rather than the thesis: COVID is where the leverage in finding 2
costs the most, and 2022 is where the estimated diversification equal risk depends on visibly stops
holding — and where the sleeve finding 1 underweights by construction turns out to be the only one
that paid.

Every figure in this document comes from one live run on one data date, **2026-11-02**, and
`compare_rules.py` regenerates every table in the comparison section.

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

Two CLIs. **`compare_rules.py` is the study** — the equal-risk-against-equal-capital comparison
this project exists to make. `run_backtest.py` is the engine run underneath it, scored against four
unlevered benchmarks.

```bash
.venv/bin/python compare_rules.py
```

It downloads prices and the risk-free series, runs both rules over the full history, and prints the
nine-row comparison table, the risk-allocation block, the relative and crisis tables, the turnover
and trading-cost block, and the caveats — then writes two CSVs and two charts to `reports/`.

| Flag | What it does |
|---|---|
| `--start YYYY-MM-DD` | First date of price history, for every rule alike. Default `2006-02-03`. Moving it moves the covariance warm-up and therefore the ERC weights: a later start is a different study, not a zoomed-in view of this one. |
| `--target-vol SIGMA` | Override the volatility the **matched column** is levered to. Default: 1/N's own realized volatility. This is *not* `run_backtest.py`'s `--target-vol`; an overridden match point is announced in the report. |
| `--sensitivities` | Also sweep the estimator — covariance halflife, shrinkage, warm-up and rebalance frequency, each across a grid bracketing its default, with both rules re-run together at every setting. About fifty engine runs, so it is off by default. |
| `--refresh-prices` | Bypass the on-disk caches and re-download. |
| `--reports-dir DIR` | Where the tables and charts are written. Default `reports/`. |
| `--no-charts` | Skip chart rendering — the slow part — and write the tables only. |
| `--verbose` | Show INFO logging from `src/`: downloads, the two engine runs, PSD repairs. |
| `--help` | The above, from the CLI. |

```bash
.venv/bin/python run_backtest.py
```

The engine on its own: one levered ERC run against equal weight, 60/40, naive 1/σ and max Sharpe,
with the run summary, the crisis decomposition and the regime diagnostics, writing a CSV and four
charts to `reports/`.

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

The first run downloads about twenty years of daily data for four tickers and takes a couple of
minutes; the results are cached under `data/cache/`, so subsequent runs are fast and work offline.
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
.venv/bin/python tests/test_riskmetrics.py     # VaR/CVaR, Sortino, Calmar, drawdown duration, the shared metric table
.venv/bin/python tests/test_headtohead.py      # one series against another: correlation/beta/capture, tail block, crisis table
.venv/bin/python tests/test_comparison.py      # the comparison: effective bets, the matched column, the naive-rescale mutation
.venv/bin/python tests/test_sensitivity.py     # turnover by hand, where a cost lands, the breakeven, the estimator sweep
```

No suite touches the network: all eleven pass offline, checked by blocking `socket.connect` and
re-running each one, and yfinance is monkeypatched wherever a fetch is
exercised. Most engine modules also carry a `__main__` self-check — `.venv/bin/python -m src.riskparity`
verifies the analytic gradient against central differences and solves 500 random covariances,
`.venv/bin/python -m src.covariance` checks that an infinite halflife reproduces the sample
covariance and that PSD repair fires on an indefinite input. **Every self-check runs offline and
writes nothing to disk.**

---

## Modules

One module per pipeline stage under `src/`, wired end to end by `run_backtest.py`:

```
config.py → data.py → covariance.py → riskparity.py → overlay.py → backtest.py → BacktestResult
                                                                                       ↓
                                                          metrics.py / benchmarks.py / stress.py / viz.py
                                                                                       ↓
                                        riskmetrics.py / headtohead.py → comparison.py → sensitivity.py
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
| `src/riskmetrics.py` | The realized-risk measures the metrics port does not carry: historical VaR and CVaR at `VAR_CONFIDENCE`, downside deviation, Sortino, Calmar, longest drawdown and whether it is still open — plus the one table formatter every report prints them with, so two reports cannot label the same metric differently. New code that *follows* `metrics.py`'s conventions rather than being added to a file that is a verbatim port. |
| `src/comparison.py` | **The study.** Equal risk against equal capital over one book: 1/N, ERC unlevered, and ERC levered by the overlay to 1/N's *own* realised volatility so the two returns are comparable — with naive 1/σ, max Sharpe and 60/40 as reference rows and the financing-omitted rows marked counterfactual. Also the risk-allocation block: capital weight beside risk share per sleeve, and effective bets `1 / Σ RCᵢ²`. The matched column comes from a second `run_backtest` call, never a hand-rolled overlay. |
| `src/headtohead.py` | One daily return series against another on a shared index: correlation, beta `cov(s, b) / var(b)`, tracking error, up/down capture, what the strategy did on the benchmark's worst `1 − VAR_CONFIDENCE` of days, and the per-crisis-window table. Named for roles (`strategy` vs `benchmark`), so the same functions serve any pairwise comparison. `src/comparison.py` is the caller. |
| `src/sensitivity.py` | The estimation-error answer. Sweeps the covariance halflife, shrinkage, warm-up and rebalance frequency across grids bracketing their defaults, re-running **both** rules together at each setting — 1/N needs no estimate at all, so the question is whether ERC's edge is an artifact of four well-chosen settings. Plus turnover, reconstructed from the drift the weights frames do not show, and the **breakeven cost** at which ERC stops beating 1/N. |
| `src/benchmarks.py` | The four unlevered comparators: equal weight, 60/40, naive inverse-volatility ("risk parity" as usually shipped, which ignores correlations) and a rolling long-only max-Sharpe tangency portfolio. All return the same `BacktestResult` the engine does. |
| `src/stress.py` | The crisis decomposition — every strategy across every crisis window, with the capital-weight-in-bonds and mean-leverage columns that make the ranking interpretable — plus the regime diagnostics: rolling SPY/TLT correlation and the diversification ratio `Σwᵢσᵢ / σₚ`. |
| `src/viz.py` | Six report charts. Four describe the engine: risk contributions through time, capital weight beside risk weight, the leverage multiplier, and correlation against the diversification ratio. Two describe the comparison: both rules' risk shares side by side on one scale, and every rule as a point on the return/volatility plane with constant-ratio rays and the levered book's path drawn on it. |
| `run_backtest.py` | The engine CLI. One of only two places the modules are wired together, and it owns no maths of its own — a wrong number here is an upstream bug. |
| `compare_rules.py` | The study CLI. Imports `load_data` from `run_backtest.py` rather than reimplementing it, so both CLIs see exactly one sample; two CLIs that fetched their own data would eventually disagree about what "the sample" is. |

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
overlay scales the whole portfolio to a chosen level instead.

**In this project that is a comparison instrument, not a second product.** Two books at different
volatilities cannot be compared on return, so the overlay's job is to put ERC at 1/N's own
volatility — and the financing it charges is then one of the costs being measured. It is also the
most expensive single mechanism in the repo (see the COVID result below), which is a finding in its
own right rather than a defect:

```
L_t = TARGET_VOL / σ̂_t,     clipped to [0.5, 3.0]
```

where `σ̂_t` is the trailing EWMA volatility of the current weights' portfolio returns, using only
data up to and including `t`. The vol halflife (42 days) is deliberately shorter than the
covariance halflife (60 days) — the overlay should react faster than the weights do.

**It does not hit the target, and the miss is systematic rather than noise.** A trailing estimator
is by construction behind, so asking for 10% realizes 11.25% and asking for 8.36% realizes 9.40% —
a consistent **+12.5%** overshoot. Every report here prints the realized volatility beside the
target for that reason, and the volatility-matched column is described as *approximately* matched
rather than matched. A second property makes that column cheap: **changing `target_vol` does not
change the ERC weights at all.** The target and the cap feed the leverage path only, never the
solve, so "the same book at a different size" is one extra engine run rather than a redesign —
and `compare_rules.py` asserts it on every run instead of trusting it.

Leverage is **financed**:

```
r_p = L_t · (r_assets − r_f) + r_f
```

The borrowed portion costs the 3-month bill rate. Dropping that term is the standard way a levered
backtest flatters itself; over this sample the financing leg is worth **1.10%/yr**, and it is the
whole difference between the matched column beating equal capital and losing to it. Transaction
costs and a borrow spread over the bill rate are *not* modelled, so the levered results here remain
optimistic even so — Results section 4 measures what that omission is worth rather than asserting
it is small.

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

*Every figure below comes from one live run, `python compare_rules.py --sensitivities`, on data
through **2026-11-02**: 2007-03-01 → 2026-11-02, **237 monthly rebalances, 4,951 return days, 0
solver failures, 0 covariances repaired**, worst risk-contribution deviation 1.6e-07, `run_health
clean`. The six charts referenced are committed under `reports/` so they can be read without
running anything. One data date per table — quoting two in one table is the one way to produce a
wrong comparison here.*

### 1. What each rule actually does with its risk

The engine hits its own specification: every rebalance lands within 2e-07 of equal risk
contribution. `reports/rule_risk_shares.png` puts both rules' risk shares on one scale, and it is
the whole argument in one frame — a flat panel beside one that is emphatically not.

| Sleeve | ERC capital | ERC risk | 1/N capital | 1/N risk |
|---|---|---|---|---|
| SPY (equities) | 20.0% | 25.0% | 25.0% | **32.9%** |
| TLT (long Treasuries) | 21.8% | 25.0% | 25.0% | 16.3% |
| DBC (commodities) | 16.3% | 25.0% | 25.0% | **41.8%** |
| TIP (inflation-linked) | **41.9%** | 25.0% | 25.0% | **9.0%** |

*Mean capital weight beside mean risk share, over all 237 rebalances.*

Equal capital is flat by definition and its **risk** is not. Equal risk is flat by construction and
its **capital** is not. TIP takes 42% of the money to earn its 25% of the risk; SPY takes 20% of
the money for the same 25%.

The means understate how far 1/N wanders, because they average over the wandering:

| | ERC | 1/N | Naive 1/σ | 60/40 | Max Sharpe |
|---|---|---|---|---|---|
| **Effective bets** (mean of `1 / Σ RCᵢ²`) | **4.00** | **2.92** | 3.75 | 1.41 | 1.78 |
| range across rebalances | 4.00 – 4.00 | 1.84 – 3.85 | 2.84 – 3.97 | 0.78 – 2.00 | 0.88 – 3.41 |
| worst single-sleeve risk share | 25.0% | **66.5%** | 41.7% | 112.9% | 106.6% |
| SPY's risk share, low to high | 25.0% – 25.0% | **8.9% – 60.3%** | −1.6% – 28.2% | 50.0% – 112.9% | −4.7% – 106.6% |
| rebalances with a *negative* risk share | 0 | 12 | 3 | 37 | 35 |

**4.00 effective bets against 2.92, out of a possible 4.00** is the compactest statement of the
thesis in the repo. Two notes on reading it, both measured rather than assumed:

- It is the **mean of the per-rebalance values**, not `1 / mean(Σ RC²)`. Because `1/x` is convex
  the second understates the first — 2.83 against 2.92 here — and the first is the one that reads
  as "the average number of bets". An earlier draft of this project quoted the 2.83.
- **The usual `[1, N]` bound is not guaranteed and is genuinely breached on this data.** A sleeve
  that hedges the rest carries a *negative* risk share; the shares still sum to 1, but `Σ RC²` can
  then exceed 1. 1/N bottoms at 1.84, but 60/40 bottoms at **0.78** and max Sharpe at 0.88. A test
  asserting `>= 1` would be asserting something about the rule, not about the formula.

### 2. What that buys, and what it costs

Nine rows, one calendar, `compare_rules.py`'s own output:

| | CAGR | Volatility | Return/vol | Max drawdown | Mean leverage |
|---|---|---|---|---|---|
| **ERC unlevered** | 5.10% | 6.90% | **0.756** | −17.15% | 1.00x |
| Naive 1/σ | 4.84% | 6.64% | 0.745 | −16.69% | 1.00x |
| **1/N (equal weight)** | 5.76% | 8.36% | **0.712** | −28.09% | 1.00x |
| Max Sharpe | 6.66% | 10.49% | 0.667 | −26.59% | 1.00x |
| 60/40 | 8.61% | 11.44% | **0.779** | −29.92% | 1.00x |
| ERC @ 10% target, financed | 7.09% | 11.25% | 0.665 | −29.10% | 1.76x |
| ERC @ 10% target, financing *omitted* | 8.27% | 11.25% | 0.763 | −29.05% | 1.76x |
| **ERC @ 1/N's vol, financed** | 6.24% | 9.40% | **0.692** | −24.85% | 1.47x |
| ERC @ 1/N's vol, financing *omitted* | 6.96% | 9.40% | 0.763 | −24.81% | 1.47x |

*Ratios are computed against a zero risk-free rate — a return-to-volatility ratio, applied
identically to every row, so the ranking is fair even though the level is not a true Sharpe. The
two financing-omitted rows are **counterfactual**: nobody borrows at zero. They are here only
because the distance from each to its financed twin is the cost of the leverage.*
`reports/return_vol_plane.png` plots all nine, with constant-ratio rays, so a row's ratio is the
ray it sits on.

**Unlevered, equalizing risk wins.** 0.756 against 0.712, and a drawdown of −17.2% against −28.1%
— it very nearly halves the worst loss. It gives up **66bp/yr of return** to do it.

**Levered to a comparable volatility, it loses.** Two books at 6.90% and 8.36% volatility are not
comparable on return, so the overlay re-runs the same ERC book at 1/N's own 8.36%. It lands at
9.40% rather than 8.36% — a trailing volatility estimator cannot hit a target, and this one
overshoots by a consistent **+12.5%**, so the column is *approximately* matched and the realized
figure is the one to read. At that size, with the real bill rate charged, ERC scores **0.692 —
below 1/N's 0.712.** With the financing leg omitted it scores 0.763, comfortably above. **The
financing eats the entire risk-allocation gain and a little more**, and a backtest that drops the
term reports the second number and calls it the first.

**And most of the gain is not the solver.** Naive 1/σ — inverse volatility, which ignores
correlations entirely — reaches 0.745 against ERC's 0.756 with no optimizer at all. The ERC
solver's whole reason to exist is the correlation structure, and on this universe that structure is
worth about one point of return-to-vol.

Plain 60/40 still beats everything here on risk-adjusted return (0.779), with no solver, no
leverage and no financing. If the question is "should you have run any of this instead of 60/40
since 2007", the measured answer is no.

### 3. Does the edge survive the covariance being estimated?

This is the standard objection: 1/N needs no estimate at all, and ERC needs a halflife, a shrinkage
intensity, a warm-up length and a rebalance frequency. If the advantage lives only at the settings
in `config.py`, it belongs to the settings. `compare_rules.py --sensitivities` moves one knob at a
time across a grid bracketing its default, re-running **both** rules together at every setting so
the window and the calendar never differ:

| Knob | Range swept | ERC's unlevered gap over 1/N |
|---|---|---|
| Covariance halflife | 20 – 250 days (default 60) | +0.038 to +0.060 |
| Shrinkage intensity | 0.00 – 0.50 (default 0.10) | +0.036 to +0.047 |
| Warm-up | 126 – 504 days (default 252) | +0.043 to +0.045 |
| Rebalance frequency | 1 – 12 months (default 1) | **−0.029** to +0.043 |

**Unlevered, ERC beat 1/N at 16 of the 17 settings swept.** The one failure is informative: at
**annual** rebalancing it scores 0.683 against 1/N's 0.712 and loses. Equal risk is a claim about a
covariance that moves, and a book that re-solves once a year is holding last year's answer. At
six-monthly the edge is already down to +0.003. **ERC's advantage requires re-optimising at least
quarterly**, which is the one estimator setting this result is genuinely sensitive to.

**Levered to 1/N's volatility, ERC was behind at all 17 settings**, from −0.010 (a 20-day halflife)
to −0.071 (six-monthly rebalancing). No estimator setting rescues it. That makes finding 2 stronger
rather than weaker: the gap is the financing, not the covariance.

### 4. What it would cost to trade

"No transaction costs are modelled" is honest everywhere in this repo and is also the obvious
attack on it — ERC re-optimises every month and 1/N does not. The obvious version of that attack
turns out to be wrong, and the reason is worth stating.

**Turnover here is reconstructed, not read off the weights.** Both rules are modelled as holding
constant weights between rebalances, which is arithmetically a book traded back to target every
day, so 1/N's weights frame shows exactly 0.0% turnover — which would say equal capital is free to
run, which is false. At each rebalance the previous target is carried forward by the sleeves'
realized returns and the trade is measured from *that* to the new target, one-way, as a fraction of
the book.

| | ERC | 1/N |
|---|---|---|
| Traded per rebalance | 1.89% | 1.26% |
| …of which drifting back to target | 1.03% | 1.26% |
| …of which re-optimising | 1.83% | **0.00%** |
| Worst single rebalance | 14.09% | 4.60% |
| Traded per year | **22.7%** | **15.1%** |

*Drift-back and re-optimisation do not sum to the total: they partly cancel, because ERC drifts
toward its next solution about as often as away from it.*

A 1/N book has to be traded back to 25% every month too, so ERC trades about **1.5x** what equal
capital does, not ten times it. Charge both rules the same cost per unit traded and ERC's unlevered
advantage survives until **291bp** — a cost nobody pays on four of the most liquid ETFs in
existence. At a pessimistic 25bp the gap goes from +0.043 to +0.040.

For the matched column there is nothing for a cost to decide: it is behind at zero cost already.

### 5. The crisis chapters

Measured against the rule this project actually argues with — **equal capital**, not 60/40 — and
from the peak close:

| Window | ERC return | ERC max DD | 1/N return | 1/N max DD |
|---|---|---|---|---|
| GFC (2008) | −5.15% | **−16.91%** | −16.16% | −28.09% |
| COVID (2020) | −8.91% | **−14.18%** | −12.63% | −16.53% |
| **Rate shock (2022)** | **−14.28%** | **−16.82%** | −12.25% | −16.33% |

ERC wins the GFC and COVID comfortably and **loses 2022 on both return and drawdown.** The
mechanism is in the next section, and it is a real limitation of equalizing risk rather than a
quirk.

**The levered book is a different story, and the worst drawdown in the sample is the overlay's.**

| Crisis window | ERC @ 10% target | Equal weight | 60/40 | Naive 1/σ | Max Sharpe |
|---|---|---|---|---|---|
| GFC (2008) | −25.43% | −28.09% | −29.92% | −16.11% | −20.08% |
| **COVID (2020)** | **−29.10%** | −16.53% | −17.76% | −13.52% | −12.55% |
| Rate shock (2022) | −17.72% | −16.33% | −25.83% | −16.34% | −23.19% |

The −29.10% COVID drawdown (peak 2020-02-20, trough 2020-03-18) is the deepest in the whole sample,
and the levered book is the **worst** of the five in that window by eleven percentage points. The
unlevered ERC book fell −14.18%. The gap is leverage: the strategy entered COVID at **2.33x, its
highest pre-crisis leverage of the entire run** (GFC 1.83x, 2022 1.74x), because 2019 was calm and
a trailing volatility estimator has nothing to say about a shock that has not happened yet. The vol
target levers hardest into quiet regimes, and the fastest crash in the sample followed the quietest
year in the sample.

That is not a bug in the overlay — it is the overlay working as specified. No causal filter can
de-lever before the volatility it is measuring has been realized. Vol targeting is a risk
*control*, not a risk *forecast*, and this is what that distinction costs.
`reports/leverage.png` shows the shape plainly: the line peaks just before each crisis band and
bottoms out inside or after it. Measured on the engine's own monthly leverage, the book was above
1.0x on **76%** of GFC days, **100%** of COVID days and **53%** of 2022 days; in COVID it held
2.33x and then 2.18x through the entire crash and only cut to 0.64x at the 2020-03-31 rebalance,
after the trough — selling risk at the bottom rather than before the fall.

This is the single most expensive finding in the repo, and it is why section 2 reports the
unlevered comparison beside the matched one rather than only the latter.

---

## The 2022 Failure Mode

The evidence chapter for the limits of findings 1 and 3: 2022 is where the *estimated
diversification* equal risk depends on stopped holding, rather than where the overlay mistimed. It
is also the one window equal risk loses outright, and the reason it loses is a property of the rule
rather than bad luck.

**Against equal capital, unlevered ERC loses 2022 on both return and drawdown** — −14.28% against
−12.25%, −16.82% against −16.33%. The cause is one sleeve. **DBC returned +21.4% over the window and
was the only thing that worked.** 1/N holds it at 25% of capital; ERC holds it at **16.3%**, because
commodities are the most volatile sleeve and equal *risk* therefore means less *money* there. In
2022 the naive rule accidentally held more of the one thing that paid.

That is not a quirk to note and move past. **The sleeve equal risk underweights by construction is
the high-volatility sleeve, and sometimes high volatility is where the return is.** It is the
structural limitation of finding 1: a better *risk* allocation is not the same thing as a better
allocation, and this window is what the difference looks like.

Measured against 60/40 instead, 2022 reads as a *win* for risk parity (−17.72% against −25.83%),
which is what earlier drafts of this section were written around. Both rankings are true; only one
of them is about the question this project asks.

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
means an outsized share of capital in duration. At the 2022 rebalances the book held **23.1% of
capital in TLT** carrying its mandated 25.0% of risk, plus a **39.2%** TIP position; the risk model,
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

**What the project concludes.** Equalizing risk is a genuinely better *risk* allocation than
equalizing capital — 4.00 effective bets against 2.92, a drawdown of −17.2% against −28.1%, and it
does not depend on the estimator settings. It is a worse *return* proposition once the leverage
that makes the comparison fair is paid for, and it is oversold in three specific ways this backtest
can measure: on a risk-adjusted return that disappears when the financing leg is charged, on a
correlation structure that is worth about a point of return-to-vol over plain inverse volatility,
and as diversification — which is a claim about a covariance estimated from the past and not
stable. 2022 is what it looks like when the estimate is still describing the previous regime, and
also when the sleeve the rule underweights by construction turns out to be the sleeve with the
return. The engine did its job at every rebalance to seven decimal places. That was never the part
at risk.

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
- **Rebalance frequency matters more than the other estimator settings**, and it is the one knob
  the unlevered conclusion is sensitive to: at annual rebalancing equal risk stops beating equal
  capital. See Results section 3.
- **Not modelled:** transaction costs, bid-ask spread, and any borrow spread over the bill rate, on
  any row, so every level reported here is optimistic. `src/sensitivity.py` measures what the
  omission is worth rather than asserting it is small: the two rules trade 22.7% and 15.1% of the
  book per year respectively, and equal risk's unlevered advantage survives a cost of **291bp per
  unit traded**, charged to both. These would move the levels; on this universe they would not
  change the ranking.
- **Four sleeves.** Equal risk and equal capital diverge further as the universe grows and as
  volatility dispersion widens, so a wider universe would strengthen finding 1. `DATA_START_DATE`
  is pinned to DBC's 2006 inception and `data._validate` rejects anything listed after it, so
  widening the universe is data-layer work rather than a config edit.

---

## Repository layout

```
compare_rules.py      the study CLI — equal risk against equal capital, the matched column,
                      turnover and the cost breakeven, and (with --sensitivities) the sweep
run_backtest.py       the engine CLI — one levered ERC run against four unlevered benchmarks
src/                  one module per pipeline stage (table above); comparison.py is the study
                      and sensitivity.py is the estimation-error and cost answer
tests/                standalone test scripts + the synthetic fixture
reports/              generated CSVs and charts (the committed set is from one live run)
data/cache/           cached network pulls (gitignored; delete a file to force a re-pull)
PLAN.md               design rationale and the pinned maths
TASKS.md              numbered build order, one owner per step
```
