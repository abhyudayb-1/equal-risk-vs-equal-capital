"""Constants, paths and policy for the risk parity engine.

Read this first when the pipeline's behaviour looks wrong -- every magic number
in the project lives here rather than inline at its call site.

Each constant carries a comment recording the measurement or reasoning that set
it, so read the comment before changing a value.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _directory in (DATA_DIR, CACHE_DIR, REPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe -- four asset classes, liquid ETF proxies
# ---------------------------------------------------------------------------
EQUITY_TICKER: str = "SPY"
NOMINAL_BOND_TICKER: str = "TLT"
COMMODITY_TICKER: str = "DBC"
INFLATION_TICKER: str = "TIP"
GOLD_TICKER: str = "GLD"  # optional 5th sleeve, off by default

UNIVERSE: tuple[str, ...] = (
    EQUITY_TICKER,
    NOMINAL_BOND_TICKER,
    COMMODITY_TICKER,
    INFLATION_TICKER,
)

SLEEVE_LABELS: dict[str, str] = {
    EQUITY_TICKER: "Equities",
    NOMINAL_BOND_TICKER: "Nominal bonds",
    COMMODITY_TICKER: "Commodities",
    INFLATION_TICKER: "Inflation-linked",
    GOLD_TICKER: "Gold",
}

# ---------------------------------------------------------------------------
# Backtest window
# ---------------------------------------------------------------------------
# The project brief asks for 2005, and the data cannot deliver it. DBC is the
# earliest liquid broad-commodity ETF at 2006-02-03; GSG and DJP are later
# still. Rather than back-fill a proxy silently, the backtest starts on the
# first date all four sleeves actually trade, and the README says so. All three
# crisis regimes are still fully covered.
DATA_START_DATE: str = "2006-02-03"

# Covariance needs history before the first decision can be made, so the first
# rebalance lands roughly one year after DATA_START_DATE (~Feb 2007) -- still
# ~8 months ahead of the October 2007 GFC peak.
COV_WARMUP_TRADING_DAYS: int = 252

# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252
MONTHS_PER_YEAR: int = 12

# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
# Halflife in trading days for the exponentially-weighted covariance. 60 days
# (~3 months) is the regime-responsiveness/noise trade-off; shorter reacts to
# 2020-style shocks faster but produces noisier weights.
COV_EWMA_HALFLIFE_DAYS: int = 60

# Shrinkage toward a diagonal target, as a weight on the target in [0, 1].
COV_SHRINKAGE: float = 0.10

# Eigenvalues below this are clipped when repairing a non-PSD covariance.
MIN_EIGENVALUE: float = 1e-10

# Log a warning when the covariance condition number exceeds this -- a sign the
# solver's answer is not trustworthy even if it converges.
#
# Shrinkage maps rho -> 0.9*rho, which caps the shrunk correlation matrix's
# condition number at 37 for four sleeves even at rho = 1 exactly. Over 20,000
# random covariances built at realistic ETF-like volatilities, the worst observed
# condition number was 1,183. 10,000 sits ~8.5x above that, so a normal run does
# not warn on itself, while a genuinely pathological covariance -- a broken price
# series, a near-duplicate sleeve -- still has a threshold it can reach. (A
# higher value such as 1e8 would need a ~1,600:1 volatility ratio between two
# sleeves and could never fire on this project's data.)
MAX_CONDITION_NUMBER: float = 10_000.0

# ---------------------------------------------------------------------------
# ERC solver
# ---------------------------------------------------------------------------
ERC_MAX_ITER: int = 500

# Tolerance at which the independent convex oracle in tests/test_riskparity.py
# (derived separately from the solver it checks) must agree with the SLSQP solver.
ERC_AGREEMENT_TOLERANCE: float = 1e-6

# Long-only floor. A hard zero lets an asset leave the portfolio entirely, which
# makes the risk-contribution identity degenerate; a small floor keeps it well
# posed.
MIN_WEIGHT: float = 1e-6

# SLSQP's ftol is an ABSOLUTE threshold on the objective sum_i (RC_i - sigma_p/N)^2,
# and that objective is already ~1e-12 at the 1e-6 risk-fraction bar above (on the
# scale-normalised covariance solve_erc actually solves). So an ftol of 1e-12 stops the optimiser exactly AT the bar instead of well inside it.
# Measured on 200 random 4x4 covariances: ftol 1e-12 misses the bar 35 times,
# 1e-14 misses it 4 times (and those 4 are the local-minimum case below, not
# precision).
ERC_SLSQP_FTOL: float = 1e-14

# sum_i (RC_i - sigma_p/N)^2 is NOT convex. It has local minima that pin an asset
# to MIN_WEIGHT at a *negative* risk contribution, reliably when two sleeves are
# strongly negatively correlated -- exactly the SPY/TLT structure this project is
# built on. The escape is a restart from a fresh feasible point, which only fires
# when the first solve misses the bar, so the mean cost is ~0.08 extra solves.
# Restart points are drawn from a fixed seed so a backtest stays reproducible.
#
# This does NOT drive the failure rate to zero, and the returns diminish sharply
# -- measured over 4000 adversarial random 4x4 covariances: 0.075% of solves
# still miss the bar at 48 restarts, 0.050% at 150, 0.025% at 400. A handful of
# matrices have a true-optimum basin small enough that random restarts rarely
# land in it. 150 is chosen as the knee, not as a guarantee.
#
# On covariances drawn around the REAL four-sleeve correlation structure the rate
# is 0 in 2000. The residual is an artefact of adversarial |rho| up to 0.95, well
# outside anything SPY/TLT/DBC/TIP produce. Anything asserting across many random
# covariances (such as the oracle test) should assert agreement where both solvers
# converge rather than assuming every draw converges.
ERC_MAX_RESTARTS: int = 150

# How negative an eigenvalue may be before solve_erc rejects the covariance as
# not PSD, as a FRACTION of the largest variance on the diagonal. Relative and
# not absolute on purpose: these covariances are annualised variances of ETF
# sleeves, so the diagonal runs ~1e-2 and an absolute bar would mean something
# different at every scale. Note this is a rejection bar for a matrix that was
# never repaired, not MIN_EIGENVALUE, which is the floor nearest_psd clips TO
# once it has decided to repair.
#
# Set from the gap between the two populations, both measured over 2000 draws:
# genuinely PSD-but-singular matrices (rho = 1 exactly, and random rank-deficient
# B@B.T) bottom out at -1.0e-15 of the largest variance, which is eigvalsh's own
# rounding noise; the non-PSD matrix that must be rejected sits at -0.42.
# 1e-10 sits five orders above the noise and eight below the rejection, so the
# choice is not delicate -- anything in that range behaves identically.
ERC_MIN_EIGENVALUE_RATIO: float = 1e-10

# ---------------------------------------------------------------------------
# Volatility-targeting overlay
# ---------------------------------------------------------------------------
TARGET_VOL: float = 0.10  # annualized

# Shorter than the covariance halflife on purpose: the overlay should react to a
# vol spike faster than the weights do.
VOL_EWMA_HALFLIFE_DAYS: int = 42

# (min, max) leverage. The cap is what stops a low-vol regime from producing an
# absurd notional -- and, in 2022, what limits the damage.
LEVERAGE_CAP: tuple[float, float] = (0.5, 3.0)

# Minimum observations before a trailing vol estimate is trusted. Set from the
# EWMA's effective sample size N_eff = 1/sum(w^2), which is what actually governs
# the estimate's precision -- not from the window length, which only bounds it.
#
# At VOL_EWMA_HALFLIFE_DAYS the geometric weights make N_eff saturate at 121.2 no
# matter how much history is supplied. Measured over 4000 synthetic draws per
# window length, at 12% true vol:
#
#     n=63  -> N_eff  57.9, relative std of the estimate 9.4%
#     n=126 -> N_eff  94.3, 7.4%
#     n=252 -> N_eff 117.5, 6.45%
#     n=504 -> N_eff 121.1, 6.38%
#
# 252 captures 97% of the reachable N_eff and lands within 1% of the precision
# floor; doubling it again buys 0.07 percentage points. It also matches
# COV_WARMUP_TRADING_DAYS, so the first rebalance is governed by one warmup
# rather than by whichever of two is longer.
VOL_WARMUP_TRADING_DAYS: int = 252

# ---------------------------------------------------------------------------
# Risk-free rate and financing
# ---------------------------------------------------------------------------
# Keyless FRED CSV endpoint -- consistent with the house rule that every data
# source in these projects is free and requires no API key.
FRED_CSV_URL: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
RISK_FREE_SERIES_ID: str = "DTB3"  # 3-month Treasury bill, secondary market

# Sharpe is reported on excess returns; 0.0 makes it a return-to-vol ratio.
ANNUAL_RISK_FREE_RATE: float = 0.0

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
SIXTY_FORTY_WEIGHTS: dict[str, float] = {EQUITY_TICKER: 0.60, NOMINAL_BOND_TICKER: 0.40}
MAX_SHARPE_LOOKBACK_DAYS: int = 252

# ---------------------------------------------------------------------------
# Crisis windows (peak-to-trough, inclusive)
# ---------------------------------------------------------------------------
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC (2008)": ("2007-10-09", "2009-03-09"),
    "COVID (2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock (2022)": ("2022-01-03", "2022-10-12"),
}

# The 2022 diagnostic: rolling correlation of equities vs nominal bonds. When
# this crosses positive, the diversification premise of risk parity is gone.
CORRELATION_WINDOW_DAYS: int = 63
CORRELATION_PAIR: tuple[str, str] = (EQUITY_TICKER, NOMINAL_BOND_TICKER)

# ---------------------------------------------------------------------------
# HTTP / caching
# ---------------------------------------------------------------------------
HTTP_USER_AGENT: str = "risk-parity-engine abhyudayb11@gmail.com"
HTTP_TIMEOUT_SECONDS: float = 30.0

PRICE_CACHE_MAX_AGE_HOURS: float = 12.0
RISK_FREE_CACHE_MAX_AGE_HOURS: float = 24.0

# ---------------------------------------------------------------------------
# Regime diagnostics
# ---------------------------------------------------------------------------
# Measured on live data, and it corrected the project's original thesis: the
# equity/bond correlation did NOT turn meaningfully positive during the 2022
# drawdown. Across 2022-01-03 to 2022-10-12 the rolling 63-day SPY/TLT
# correlation averaged -0.09 and was positive on only 31% of days. The
# persistent flip is a 2023-onward phenomenon (66%, 65%, 70% of days positive in
# 2023/2024/2025), so it gets its own window rather than being folded into the
# crisis list. ``None`` as the end date means "through the end of the sample".
REGIME_SHIFT_WINDOW: tuple[str, str | None] = ("2021-01-01", None)

# The diversification ratio, ``sum(w_i * sigma_i) / sigma_p``, is the diagnostic
# that actually degrades THROUGH 2021-22 rather than after it. Inverse-vol
# weighted annual means on live data: 2.18 (2012 peak), 2.11 (2008), 1.78
# (2020), 1.68 (2021), 1.59 (2022), 1.52 (2023 -- lowest in the sample).
#
# A correlation-SIGN alarm fires in 2023, after the damage. This one is already
# falling while the damage is being done, which is why the regime diagnostics key off it.
DIVERSIFICATION_WINDOW_DAYS: int = CORRELATION_WINDOW_DAYS

# The diversification ratio is measured on an UNSHRUNK covariance, unlike every
# other covariance in the project. Shrinkage exists to stabilise a matrix that an
# optimiser is about to invert; the ratio only reads the matrix, and pulling the
# off-diagonals toward a diagonal target makes the sleeves look more independent
# than they were -- it biases the diagnostic in exactly the direction the
# diagnostic is supposed to detect.
#
# Measured, and this is what set the constant: at COV_SHRINKAGE = 0.10 the annual
# means come out 2.098 (2008) / 1.807 (2020) / 1.702 (2021) / 1.615 (2022) /
# 1.552 (2023) -- the same shape, uniformly ~0.03 too diversified. At 0.0 they
# reproduce the published path above to three decimals: 2.111 / 2.179 (2012 peak)
# / 1.790 / 1.678 / 1.586 / 1.521. A diagnostic whose absolute level is quoted in
# the write-up has to be the reproducible one.
DIVERSIFICATION_SHRINKAGE: float = 0.0

# ---------------------------------------------------------------------------
# Run summary reporting
# ---------------------------------------------------------------------------
# Presentation only. Neither of these changes a single number the engine
# computes -- they govern how the run summary talks about numbers it was given,
# and they live here because this file is where a reader looks for a threshold
# rather than because they are policy.
#
# How many offending dates a summary line names before it collapses to a count.
# Six fits one terminal line at these label widths; the full list is always in
# meta["diagnostics"], which is the machine-readable record.
SUMMARY_MAX_DATES_LISTED: int = 6

# Wrap width for the summary's continuation lines. 88 rather than 79 because the
# lines carry dates and percentages that read worse broken up, and rather than
# unlimited because a terminal wrapping a 190-character sentence at whatever
# column it happens to be breaks it mid-word.
SUMMARY_LINE_WIDTH: int = 88

# Above this share of rebalances at the leverage CEILING, the summary adds a
# sentence saying the overlay is saturated. A saturated overlay is not targeting
# volatility any more, it is holding a constant notional, and that is invisible
# in the leverage series itself -- which just looks flat. The exact cut is a
# judgement about when the reader should be told, not a policy limit: the cap
# itself is LEVERAGE_CAP, and the fraction is printed at every level regardless.
SATURATION_REPORT_FRACTION: float = 0.25

# ---------------------------------------------------------------------------
# Tail risk -- src/riskmetrics.py, src/headtohead.py
# ---------------------------------------------------------------------------
# Confidence level for the historical VaR / CVaR in every realized-risk table,
# and for the tail block of a pairwise comparison. 95%, not 99%, on purpose, and
# what sets it is the SHORTEST window worth measuring rather than the full
# sample. Take COV_WARMUP_TRADING_DAYS (252) as that floor -- it is already this
# project's "fewest daily observations we trust" bar: a 5% tail there is ~13
# observations, where a 1% tail would be ~3, far too few for an
# average-of-the-tail measure like CVaR to mean anything. On the engine's own
# 4,897-day sample both levels would work (~245 against ~49), so 99% is not
# wrong everywhere -- it is wrong at the short end, and one level applied
# everywhere keeps two reports' tables directly comparable where a per-report
# choice would not. Every table prints its own observation count beside it for
# the same reason.
VAR_CONFIDENCE: float = 0.95

# ---------------------------------------------------------------------------
# The equal-risk vs equal-capital comparison -- src/comparison.py
# ---------------------------------------------------------------------------
# Floor on the volatility an ERC book may be levered *to*. The matched column's
# target is another rule's realized volatility, so a degenerate benchmark would
# otherwise hand the overlay a nonsense target and get a straight-faced answer
# back: a numerically-flat return series (a constant that is not exactly
# representable in binary) measures 1.7e-18 rather than 0.0, which is positive,
# finite, and passes a bare `vol <= 0` guard. Levering an ERC book to 1.7e-18
# volatility just pins the overlay at its leverage FLOOR and reports a column.
#
# Set four orders of magnitude below anything real and ten above the noise:
# every rule in this project realizes 6%-12% annualized, the thinnest window
# worth measuring is still ~1e-3, and the degenerate cases measure ~1e-18. There
# is no scale to make this relative to -- the target is itself a volatility --
# so it is absolute, which is only safe because a volatility's units are fixed.
MIN_MATCHABLE_VOL: float = 1e-8

# ---------------------------------------------------------------------------
# The estimation-error sweep and trading costs -- src/sensitivity.py
# ---------------------------------------------------------------------------
# The standard objection to the whole comparison is that ERC needs an ESTIMATED
# covariance and 1/N needs nothing, so ERC's edge could be an artifact of four
# well-chosen estimator settings. These grids are what answers it, and each one
# brackets its default rather than exploring around it: a sweep whose settings
# all sit on one side of the default measures a slope, not a sensitivity.
#
# Halflives from 20 days (a month of memory -- the estimate is mostly last
# month) to 250 (a year, nearly the whole warm-up window), around the 60 in
# COV_EWMA_HALFLIFE_DAYS.
SENSITIVITY_COV_HALFLIVES: tuple[int, ...] = (20, 40, 60, 120, 250)

# Shrinkage from 0.0 -- the raw EWMA, no shrinkage at all, which is the setting
# that says how much of ERC's edge the shrinkage is responsible for -- to 0.50,
# halfway to a diagonal covariance, i.e. halfway to ignoring correlations and
# therefore halfway to naive 1/sigma. Around the 0.10 in COV_SHRINKAGE.
SENSITIVITY_SHRINKAGES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50)

# Warm-up from half a year to two, around the year in COV_WARMUP_TRADING_DAYS.
# This one moves the sample as well as the estimate -- a longer warm-up starts
# the comparison later -- so both rules take it together and the row prints its
# own observation count.
SENSITIVITY_WARMUPS: tuple[int, ...] = (126, 252, 504)

# Rebalance every month, quarter, half-year or year. Monthly is the default and
# the most expensive; annual is where an estimated covariance has the least
# chance to be wrong often and the most chance to be stale.
SENSITIVITY_REBALANCE_MONTHS: tuple[int, ...] = (1, 3, 6, 12)

# One-way cost per unit of capital traded, in basis points, for the cost
# sensitivity. Nothing in this project charges a cost by default, so 0 is the
# headline and the rest are the question. The range brackets what these four
# ETFs plausibly cost: SPY/TLT/TIP quote inside a basis point and DBC is the
# wide one, so 1-5bp is realistic retail-to-institutional, 10-25bp is
# pessimistic, and 50bp is there to find the breakeven rather than to be
# believed.
COST_RATES_BPS: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0)

# Upper bound for the breakeven-cost search, as a decimal. 5% one-way per unit
# traded is far past any real market; the search reports "beyond this" rather
# than a number if it gets there, because an extrapolated breakeven is not a
# measurement.
MAX_BREAKEVEN_COST: float = 0.05

# ---------------------------------------------------------------------------
# Is a ratio gap distinguishable from noise -- src/significance.py
# ---------------------------------------------------------------------------
# Confidence level of the interval printed beside every ratio gap. 95% because
# it is the level a reader assumes when none is stated, and the same level as
# VAR_CONFIDENCE, so the report carries one notion of "the tail" rather than
# two. The level is printed in the table's own column header regardless.
RATIO_TEST_CONFIDENCE: float = 0.95

# Mean block length of the stationary bootstrap, in trading days. The blocks
# exist to carry serial dependence into each resample, and what these returns
# have is volatility clustering rather than autocorrelation in the returns
# themselves: on the live sample (data through 2026-11-06) the lag-1
# autocorrelation of ERC's daily return is 0.04 and of 1/N's 0.00, while for
# the ABSOLUTE return it is 0.26 / 0.24 at lag 1 and still 0.10 / 0.11 at lag
# 63.
#
# Measured, and this is what set the constant -- the standard error of the
# ERC-minus-1/N ratio gap (point estimate +0.040) by mean block length, 10,000
# draws each:
#
#     1 day (i.i.d.)  0.080        42 days   0.084
#     5 days          0.075        63 days   0.086
#     10 days         0.077        126 days  0.084
#     21 days         0.081        252 days  0.084
#
# It rises to a plateau by about 42 days and is flat from there, so 63 -- one
# quarter, and the same length as CORRELATION_WINDOW_DAYS -- sits on the
# plateau and happens to be its widest point. That is the conservative choice
# and it is not a delicate one: the interval spans zero at every length in the
# list, and the closed-form Memmel standard error, which assumes no dependence
# at all, is 0.079. With ~4,950 observations a 63-day mean still leaves ~80
# blocks per resample.
BOOTSTRAP_MEAN_BLOCK_DAYS: float = 63.0

# Resamples per interval. Set by the Monte Carlo noise in the interval's own
# endpoints, measured as their standard deviation across 8 seeds on the live
# sample: 0.007 / 0.007 (lower / upper) at 1,000 draws, 0.003 / 0.006 at 5,000,
# 0.002 / 0.004 at 10,000. The endpoints are printed to three decimals and sit
# ~0.16 either side of the estimate, so at 10,000 the simulation moves the
# third decimal and nothing a reader would act on. It costs about 2.3s for the
# nine-column table; doubling it again would buy a third decimal that the
# sampling error (0.086) makes meaningless.
BOOTSTRAP_DRAWS: int = 10_000

# Seed for the resampling. The value is arbitrary; what matters is that it is
# fixed, so two runs on the same data print the same interval and a number in
# the README can be reproduced rather than approximately recovered.
BOOTSTRAP_SEED: int = 20261106
