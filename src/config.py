"""Constants, paths and policy for the risk parity engine.

Read this first when the pipeline's behaviour looks wrong -- every magic number
in the project lives here rather than inline at its call site.

This file is SHARED between Person A and Person B and is **append-only**: add
constants at the bottom of the relevant block, never reorder or reformat what is
already here, and name the block you touched in your PR body.
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

BENCHMARK_TICKER: str = EQUITY_TICKER

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
MAX_CONDITION_NUMBER: float = 1e8

# ---------------------------------------------------------------------------
# ERC solver
# ---------------------------------------------------------------------------
ERC_MAX_ITER: int = 500
ERC_TOLERANCE: float = 1e-12

# Tolerance at which Person B's independent convex oracle must agree with
# Person A's SLSQP solver (tests/test_riskparity.py).
ERC_AGREEMENT_TOLERANCE: float = 1e-6

# Long-only floor. A hard zero lets an asset leave the portfolio entirely, which
# makes the risk-contribution identity degenerate; a small floor keeps it well
# posed.
MIN_WEIGHT: float = 1e-6

# SLSQP's ftol is an ABSOLUTE threshold on the objective sum_i (RC_i - sigma_p/N)^2,
# and that objective is already ~1e-12 at the 1e-6 risk-fraction bar above (on the
# scale-normalised covariance solve_erc actually solves). So ERC_TOLERANCE used as
# an ftol stops the optimiser exactly AT the bar instead of well inside it.
# Measured on 200 random 4x4 covariances: ftol 1e-12 misses the bar 35 times,
# 1e-14 misses it 4 times (and those 4 are the local-minimum case below, not
# precision). This supersedes ERC_TOLERANCE as the value passed to SLSQP.
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
# covariances (step 19's oracle) should assert agreement where both solvers
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
# rounding noise; the non-PSD matrix step 20 requires be rejected sits at -0.42.
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
# falling while the damage is being done, which is why step 25 keys off it.
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
