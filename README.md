# Equal Risk vs Equal Capital

**What does allocating equal *risk* buy you over allocating equal *capital* — and what does it
cost?**

This project answers that question with a daily backtest of a four-asset portfolio, from February
2006 to November 2026. That period covers the 2008 financial crisis, the March 2020 COVID crash
and the 2022 rate shock. The four assets are liquid ETFs:

| Ticker | Asset class |
|---|---|
| **SPY** | US equities |
| **TLT** | Long-dated US Treasuries |
| **DBC** | Commodities |
| **TIP** | Inflation-linked Treasuries |

Two ways of building the portfolio are compared head to head:

- **Equal capital (1/N):** put 25% of the money in each asset.
- **Equal risk contribution (ERC, "risk parity"):** choose the weights so that each asset
  contributes 25% of the portfolio's *volatility*.

Three reference portfolios provide context: naive inverse-volatility (1/σ), a rolling max-Sharpe
portfolio, and a classic 60/40 stock/bond mix.

---

## The short answer

> **Equal risk is a better way to spread risk than equal capital, and it wins on risk-adjusted
> return when neither book uses leverage. Once you pay for the leverage needed to compare the two
> at the same volatility, it loses.**

| | Equal risk (ERC) | Equal capital (1/N) |
|---|---|---|
| Effective number of risk shares (out of 4) | **4.00** | 2.92 |
| Return / volatility, unlevered | **0.723** | 0.683 |
| Worst drawdown, unlevered | **−17.6%** | −29.0% |
| Annual return, unlevered | 4.83% | **5.43%** |
| Return / volatility at 1/N's realised volatility (hindsight), financing charged | 0.656 | **0.683** |

Four findings:

1. **Equal risk really does balance the risk.** ERC holds every asset at exactly 25% of portfolio
   risk. Under 1/N, equities' share of risk swings between 9% and 60%, commodities average 42%,
   and TIP barely matters (9%).
2. **Without leverage, equal risk wins on risk-adjusted return.** It earns 0.723 of return per
   unit of volatility against 0.683, and its worst loss is about 40% shallower. In exchange it
   gives up 0.60 percentage points of return a year.
3. **Once leverage is paid for, it loses.** To compare returns at the same risk, ERC is levered
   toward the volatility 1/N actually realised over the whole sample. That target is only known
   in hindsight, so this is a diagnostic of history rather than a strategy anyone could have run
   (see [section 3](#3-volatility-targeting-as-a-measuring-tool)). Borrowing at the T-bill rate
   costs about 0.7% a year of return, and that wipes out the whole advantage: 0.656 against
   0.683. Leave the borrowing cost out and ERC appears to win (0.725), which is how this strategy
   usually gets oversold.
4. **Most of the benefit doesn't need an optimizer.** Simply weighting each asset by 1/volatility
   scores 0.715, close to ERC's 0.723. Modelling the correlations between assets adds less than
   one point.

Plain 60/40 beat all of them on risk-adjusted return over this period (0.750).

Every portfolio is **rebalanced monthly**: traded to its target at each month end and then held,
drifting with prices, until the next one. The returns and the trading costs below are both
computed on that same held book.

---

## Quick start

Requires Python 3.10+. No API keys: prices come from Yahoo Finance (via `yfinance`) and the
risk-free rate comes from FRED's public CSV endpoint.

```bash
git clone https://github.com/abhyudayb-1/equal-risk-vs-equal-capital.git
cd equal-risk-vs-equal-capital

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows, use `python -m venv .venv` and `.\.venv\Scripts\python.exe` in place of
`.venv/bin/python` in every command below.

**Run the study:**

```bash
.venv/bin/python compare_rules.py                     # the comparison (about 6s once data is cached)
.venv/bin/python compare_rules.py --sensitivities     # adds the robustness sweep (about 25s)
```

**Run the engine on its own** (levered ERC against four unlevered benchmarks, with crisis
breakdowns):

```bash
.venv/bin/python run_backtest.py
```

The first run downloads about 20 years of daily data and caches it under `data/cache/`, so later
runs are fast and work offline. Both scripts print their tables and write CSVs and charts to
`reports/`. Pass `--help` to either one to see every option (start date, target volatility,
leverage cap, skipping charts, and so on).

---

## How it works

### 1. Risk contribution, not capital weight

For weights `w` and covariance matrix `C`, portfolio volatility is `σ_p = √(wᵀCw)`. Each asset's
**risk contribution** is

```
RC_i = w_i · (Cw)_i / σ_p
```

These add up exactly to `σ_p`, so `RC_i / σ_p` is that asset's share of portfolio risk. ERC is the
set of weights where every share equals 1/4.

Why this differs from 25% each: SPY is roughly three times as volatile as TIP. With equal money in
each, equities and commodities dominate the portfolio's ups and downs. Equalising risk means
holding *more* of the calm assets (bonds) and *less* of the volatile ones.

The solver minimises `Σ(RC_i − σ_p/N)²` with weights that are long-only and sum to 1. That problem
has false local minima when two assets are strongly negatively correlated, as SPY and TLT are. The
solver restarts to escape them. It is cross-checked against a separately written convex
formulation in the tests, and the two agree to about 1e-7.

### 2. Estimating the covariance

Each month, the covariance is estimated from past data only:

- **Exponentially weighted** with a 60-day halflife, so recent days count most and a change in
  market regime shows up within weeks.
- **Shrunk** 10% toward a diagonal matrix, because estimation error is concentrated in the
  correlations.
- **Repaired** if it is not positive semi-definite. The repair is done in correlation space, so
  each asset's volatility stays unchanged.

### 3. Volatility targeting, as a measuring tool

An unlevered ERC portfolio is bond-heavy, so it runs at lower volatility (6.85%) than 1/N (8.25%).
Returns at different risk levels aren't comparable. So an overlay scales the whole ERC portfolio:

```
leverage_t = target_vol / estimated_vol_t,    clipped to [0.5x, 3.0x]
```

Borrowed money is charged at the 3-month T-bill rate. On each day, at the leverage `L` the book
is carrying that day:

```
r_portfolio = L · (r_assets − r_f) + r_f
```

Three properties matter here:

- **The overlay overshoots its target by about 13%.** It uses trailing volatility, which is
  always a step behind, so a 10% target realises 11.3%. Every report prints the realised
  volatility next to the target, and the "matched" comparison is described as approximately
  matched.
- **The matched comparison uses hindsight.** Its target is 1/N's realised volatility over the
  *whole* sample, which nobody knew in 2007. Each month's leverage still uses only past data, but
  the level it aims at was chosen after the fact. Read it as "at the same realised risk as 1/N,
  which rule earned more?", not as a tradable strategy. It is labelled "(hindsight)" everywhere
  it appears. A target set from a calibration period before the sample would be tradable, but it
  would not match 1/N's risk.
- **The volatility target never changes the ERC weights.** It only scales the position size. This
  is checked on every run.

### 4. Monthly rebalancing, for returns and costs alike

At each month end, every portfolio is traded to its target: `L × w` of equity in the assets, and
`L − 1` borrowed (or lent, when `L < 1`). It is then **held** until the next month end. Each
holding grows with its own asset's return and the loan grows at the T-bill rate, so both the
weights and the leverage drift away from target during the month. 1/N therefore has to trade
back to 25% each month too.

The trading-cost analysis charges for exactly those trades, from the drifted book to the new
target, read from the same simulation the returns come from. An earlier version re-applied the
targets every day, which is arithmetically a book traded back to target daily. That return path
and the monthly cost model described two different strategies.

### 5. No look-ahead

At each rebalance date `t`, only data dated on or before `t` is used. A decision made at the close
of `t` earns returns from `t+1` onwards. All data access goes through one function,
`data.asof_slice()`. `tests/test_lookahead.py` checks that an injected future price spike cannot
affect past decisions. It also removes the guard and checks that the spike *does* leak, which
proves the test can actually fail.

---

## Results in detail

All figures come from one run of `compare_rules.py --sensitivities` and `run_backtest.py` on data
through **2026-11-06**: 237 monthly rebalances from 2007-02-28, 4,955 daily returns, 0 solver
failures.
The first year of data is used to warm up the covariance estimate. Charts referenced below are
committed in `reports/`.

Ratios are return ÷ volatility (a Sharpe ratio with a zero risk-free rate). The same ratio is used
for every portfolio, so the ranking is fair even though the absolute level differs from a true
Sharpe ratio.

### Where each rule puts its risk

| Asset | ERC capital | ERC risk | 1/N capital | 1/N risk |
|---|---|---|---|---|
| SPY | 20.0% | 25.0% | 25.0% | **32.9%** |
| TLT | 21.8% | 25.0% | 25.0% | 16.3% |
| DBC | 16.3% | 25.0% | 25.0% | **41.8%** |
| TIP | **41.9%** | 25.0% | 25.0% | **9.0%** |

*Average over all rebalances. See `reports/rule_risk_shares.png`.*

Averages hide how much 1/N moves around from month to month:

| | ERC | 1/N | Naive 1/σ | 60/40 | Max Sharpe |
|---|---|---|---|---|---|
| Effective risk shares, average | **4.00** | **2.92** | 3.75 | 1.41 | 1.78 |
| Effective risk shares, range | 4.00 – 4.00 | 1.84 – 3.85 | 2.84 – 3.97 | 0.78 – 2.00 | 0.88 – 3.41 |
| Largest single-asset risk share | 25.0% | **66.5%** | 41.7% | 112.9% | 106.6% |
| SPY's risk share, low – high | 25.0% – 25.0% | **8.9% – 60.3%** | −1.6% – 28.2% | 50.0% – 112.9% | −4.7% – 106.6% |
| Months with a negative risk share | 0 | 12 | 3 | 37 | 39 |

"Effective risk shares" (often called "effective bets") is `1 / Σ(risk share)²`, the reciprocal
Herfindahl index of the estimated risk shares. It equals 4 when risk is spread evenly over four
assets and 1 when one asset carries it all. It can fall below 1 when a hedging asset has a
negative risk share, as happens with 60/40 and max Sharpe.

It measures how **evenly the estimated risk is split**, not how many *independent* bets the
portfolio holds. ERC's 4.00 says each asset carries a quarter of the estimated variance. It does
not say those four quarters are independent: the assets are still correlated, and counting
independent bets would need the covariance's factor structure, which this statistic does not use.

### Performance

| | Annual return | Volatility | Return/vol | Max drawdown | Avg leverage |
|---|---|---|---|---|---|
| **ERC, unlevered** | 4.83% | 6.85% | **0.723** | −17.63% | 1.00x |
| Naive 1/σ | 4.61% | 6.61% | 0.715 | −17.12% | 1.00x |
| **1/N (equal capital)** | 5.43% | 8.25% | **0.683** | −28.98% | 1.00x |
| Max Sharpe | 6.37% | 10.48% | 0.641 | −26.59% | 1.00x |
| 60/40 | 8.07% | 11.19% | **0.750** | −31.40% | 1.00x |
| ERC @ 10% vol target, financed | 6.65% | 11.34% | 0.625 | −31.01% | 1.76x |
| ERC @ 10% vol target, *no financing cost* | 7.82% | 11.33% | 0.722 | −30.95% | 1.76x |
| **ERC @ 1/N's realised volatility (hindsight), financed** | 5.83% | 9.30% | **0.656** | −25.82% | 1.46x |
| ERC @ 1/N's realised volatility (hindsight), *no financing cost* | 6.51% | 9.30% | 0.725 | −25.77% | 1.46x |

*The "no financing cost" rows are hypothetical, since nobody borrows for free. They are included
to show how much the borrowing costs. The "(hindsight)" rows aim at a volatility known only at
the end of the sample; see [section 3](#3-volatility-targeting-as-a-measuring-tool). Average
leverage is the mean of the month-end decisions; the held book drifts around it during each
month. See `reports/return_vol_plane.png`.*

### Is the result robust to the estimator settings?

1/N needs no estimates; ERC depends on a covariance halflife, a shrinkage amount, a warm-up length
and a rebalance frequency. The `--sensitivities` sweep changes one setting at a time and re-runs
**both** rules at each value (17 settings in total):

| Setting | Range tested (default) | ERC's unlevered lead over 1/N |
|---|---|---|
| Covariance halflife | 20 – 250 days (60) | +0.035 to +0.054 |
| Shrinkage | 0.00 – 0.50 (0.10) | +0.034 to +0.043 |
| Warm-up | 126 – 504 days (252) | +0.040 to +0.042 |
| Rebalance frequency | 1 – 12 months (1) | **−0.029** to +0.040 |

- **Unlevered, ERC beats 1/N at 16 of 17 settings.** The exception is rebalancing once a year
  (0.693 against 0.722). At six-monthly the lead is already down to +0.001. Equal risk needs to be
  re-solved at least quarterly, because the covariance keeps changing.
- **Rebalance frequency moves 1/N as well.** A held 1/N book drifts further between less
  frequent rebalances, and its own ratio rises from 0.683 (monthly) to 0.722 (yearly). Part of
  ERC's lost lead at long intervals is 1/N getting better, not only ERC getting worse.
- **Levered to 1/N's realised volatility (hindsight), ERC loses at all 17 settings** (by −0.018
  to −0.080). The gap comes from the cost of financing, not from how the covariance is
  estimated.

### What about trading costs?

The standard objection is that ERC re-optimises every month while 1/N doesn't trade. But a 1/N
portfolio drifts away from 25% as prices move and has to be traded back too. Measured properly,
from drifted weights to new target weights:

| | ERC | 1/N |
|---|---|---|
| Traded per rebalance | 1.89% | 1.26% |
| Traded per year | **22.7%** | **15.1%** |
| Largest single rebalance | 14.09% | 4.60% |

ERC trades about 1.5 times as much, not ten times as much. Charging both rules the same cost per
dollar traded, ERC's unlevered lead survives until costs reach **2.68% per unit traded**. These
four ETFs cost far less than that to trade. At a pessimistic 0.25%, the lead shrinks from +0.040
to +0.036.

These are the trades the return path actually makes: each portfolio is held through the month,
and its trade is measured from where it drifted to. For the levered hindsight-matched book, the
trade is measured on its levered holdings, including bringing the drifted leverage back to
target. That book is already behind 1/N before any cost (0.656 against 0.683), so costs only
widen the gap.

### Crises

Unlevered, measured from each crisis's market peak:

| Window | ERC return | ERC max DD | 1/N return | 1/N max DD |
|---|---|---|---|---|
| GFC (2008) | −6.28% | **−16.76%** | −17.39% | −28.98% |
| COVID (2020) | −8.84% | **−14.50%** | −12.54% | −16.94% |
| **Rate shock (2022)** | **−14.64%** | **−17.32%** | −12.64% | −16.84% |

ERC protects well in 2008 and 2020 and **loses to 1/N in 2022**.

**Leverage made COVID the worst drawdown in the sample.** The ERC portfolio levered to a 10%
volatility target fell **−31.01%** in COVID, against −14.50% unlevered. It went into the crash at
**2.33x leverage, more than it carried into either of the other crises** (1.83x before 2008,
1.74x before 2022), because 2019 had been calm and a trailing volatility estimate cannot see a
shock coming. It cut leverage only at the 2020-03-31
rebalance, after the market bottom. Volatility targeting *controls* risk; it cannot *forecast* it.
See `reports/leverage.png`.

| Crisis | ERC @ 10% target | Equal weight | 60/40 | Naive 1/σ | Max Sharpe |
|---|---|---|---|---|---|
| GFC (2008) | −25.55% | −28.98% | −31.40% | −16.51% | −19.92% |
| **COVID (2020)** | **−31.01%** | −16.94% | −18.50% | −13.80% | −12.87% |
| Rate shock (2022) | −18.23% | −16.84% | −26.31% | −16.79% | −23.19% |

### What went wrong in 2022

**Commodities were the only asset that made money, and ERC held the least of them.** DBC returned
+21.4% over the drawdown. 1/N held it at 25% of capital. ERC held only about 18%, because
commodities are the most volatile asset and equal risk therefore means less money there. The
asset that equal risk underweights by design is the high-volatility one, and sometimes that is
where the return is.

**The stock/bond correlation did not flip positive, contrary to the usual story.** From 2022-01-03
to 2022-10-12 the rolling 63-day SPY/TLT correlation averaged **−0.09** and was positive on only
31% of days. The lasting flip came in 2023. What actually happened is that **the hedge fell further
than the thing it was hedging**:

| SPY | TLT | DBC | TIP |
|---|---|---|---|
| −24.5% | **−29.3%** | +21.4% | −12.7% |

ERC is bond-heavy by construction (about 23% in TLT and 39% in TIP in 2022), so it carried the
full weight of the bond selloff.

**The warning sign that worked was the diversification ratio** (`Σwᵢσᵢ / σ_p`, which measures how
much diversification is reducing risk; 1.0 means none). It fell for three years in a row and was
already near its sample low *during* 2022 (1.313 in December 2022; the low was 1.307, in October
2023), while the correlation still looked reassuring:

| 2008 | 2012 | 2020 | 2021 | **2022** | 2023 |
|---|---|---|---|---|---|
| 2.111 | 2.179 | 1.790 | 1.678 | **1.586** | 1.521 |

See `reports/correlation_diversification.png`.

---

## Project layout

```
compare_rules.py     the study: equal risk vs equal capital, robustness sweep, trading costs
run_backtest.py      the engine: levered ERC vs four unlevered benchmarks, crisis tables
src/                 the library (one module per pipeline stage, below)
tests/               standalone test scripts + a synthetic data fixture
reports/             charts and CSVs from the run quoted above
data/cache/          downloaded prices and rates (gitignored, created on first run)
```

What's in `reports/`:

| File | Written by | Shows |
|---|---|---|
| `rule_comparison.csv` | `compare_rules.py` | the performance table for every portfolio |
| `risk_allocation.csv` | `compare_rules.py` | each asset's share of capital and of risk under each rule |
| `rule_risk_shares.png` | `compare_rules.py` | the same risk shares as a chart |
| `return_vol_plane.png` | `compare_rules.py` | return against volatility, unlevered and financed |
| `risk_contributions.png` | `run_backtest.py` | ERC risk contributions over time (they stay equal) |
| `capital_vs_risk_weights.png` | `run_backtest.py` | how unequal the capital weights are that equalise risk |
| `leverage.png` | `run_backtest.py` | the vol-targeting overlay's leverage over time |
| `correlation_diversification.png` | `run_backtest.py` | stock–bond correlation and the diversification ratio |
| `crisis_drawdowns.csv` | `run_backtest.py` | return and drawdown of each portfolio in each named crisis |

The pipeline:

```
data → covariance → riskparity → overlay → backtest → BacktestResult
                                                          ↓
                         metrics / benchmarks / stress / viz / riskmetrics / headtohead
                                                          ↓
                                              comparison → sensitivity
```

| Module | What it does |
|---|---|
| `config.py` | Every constant: tickers, start date, halflives, leverage cap, crisis windows. Each one carries a comment explaining how its value was chosen. |
| `data.py` | Downloads total-return prices (yfinance) and the T-bill rate (FRED `DTB3`), caches them, and owns `asof_slice()`, the one look-ahead guard. Fails loudly rather than silently dropping an asset. |
| `cache.py` | Small CSV cache with atomic writes. |
| `types.py` | `BacktestResult`, the common output of every strategy: weights, risk contributions, leverage, daily returns and diagnostics. Validated on creation. |
| `covariance.py` | Exponentially-weighted covariance, shrinkage, and PSD repair. |
| `riskparity.py` | Risk contributions and the ERC solver. |
| `overlay.py` | Volatility targeting, the leverage cap, financing, and the held-book accrual every strategy's returns come from. |
| `backtest.py` | The monthly rebalance loop and the run-health summary. |
| `benchmarks.py` | Equal weight, 60/40, naive 1/σ and max Sharpe. |
| `metrics.py`, `riskmetrics.py` | Return, volatility, Sharpe, drawdown, VaR/CVaR, Sortino, Calmar, crisis-window slicing. |
| `headtohead.py` | One return series against another: correlation, beta, capture ratios, tail behaviour. |
| `stress.py` | Crisis breakdowns and the correlation and diversification-ratio diagnostics. |
| `comparison.py` | The study itself: the nine-row comparison and the risk-allocation tables. |
| `sensitivity.py` | The estimator sweep, turnover, and the trading-cost breakeven. |
| `viz.py` | The six charts. |

---

## Tests

The tests are plain Python scripts, with no test framework. Run any file directly; each one prints
`N checks passed` and exits with code 0. None of them touch the network.

```bash
for t in tests/test_*.py; do .venv/bin/python "$t"; done
```

| Suite | Covers |
|---|---|
| `test_contract.py` | The `BacktestResult` contract, the metrics, the fixtures |
| `test_data.py` | Date windowing, the rebalance calendar, the look-ahead guard |
| `test_riskparity.py` | The risk-contribution identity, an independent ERC solver as cross-check, edge cases |
| `test_lookahead.py` | Future data cannot leak into past decisions, including a check that removing the guard makes the test fail |
| `test_benchmarks.py` | The four benchmark portfolios |
| `test_crisis_slices.py` | Crisis-window slicing and summary statistics |
| `test_stress.py` | Crisis table, diversification ratio, charts |
| `test_riskmetrics.py` | VaR/CVaR, Sortino, Calmar, drawdown duration |
| `test_headtohead.py` | Correlation, beta, capture ratios, tail block |
| `test_comparison.py` | The comparison, including a known two-asset case with an exact answer |
| `test_sensitivity.py` | Turnover, trading costs, the breakeven, the sweep |

Most modules in `src/` also include a quick self-check on synthetic data. For example,
`.venv/bin/python -m src.riskparity` checks the solver's gradient and solves 500 random problems.

---

## Data and limitations

- **Start date:** 2006-02-03, when DBC launched. Nothing earlier is filled in with a proxy. The
  first rebalance is a year later, once the covariance estimate has enough history.
- **Prices** are total-return (dividends and coupons reinvested). This matters for TLT and TIP,
  which earn most of their return as income.
- **Risk-free rate:** the 3-month T-bill (FRED `DTB3`).
- **Rebalancing:** monthly, at month end. Between rebalances each portfolio is held and drifts
  with prices; returns and trading costs are both computed on that held book.
- **Not modelled:** transaction costs, bid-ask spreads, and any borrowing spread above the T-bill
  rate, so every level reported here is somewhat optimistic. The trading-cost section above
  measures how large costs would need to be to change the ranking.
- **Only four assets.** A larger universe with more spread in volatility would make the difference
  between equal risk and equal capital bigger.
- **One sample period.** The numbers above come from a single run on data through 2026-11-06.
  Re-running on a later date moves the ratios slightly (by about 0.01 a few days later), but not
  the ranking.
