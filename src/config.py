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
