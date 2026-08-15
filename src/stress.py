"""Crisis decomposition and the regime diagnostics (steps 24-25).

This module answers the question the whole project is built around: *when the
diversification stopped working, what does the evidence look like?* It produces
two things.

**The crisis table** (step 24). Every strategy, every window in
``config.CRISIS_WINDOWS``, one row each: total return, drawdown, Sharpe -- plus a
**capital-weight-in-bonds** column. That column is not decoration. Every strategy
in the comparison lost money in 2022, and the ranking between them is very nearly
a monotone function of how much of the book each one had in TLT while TLT fell
31.2%. Reporting the ranking without it states *that* risk parity lost; reporting
it with the column states *why*, which is the difference between a backtest and
an analysis.

**The regime diagnostics** (step 25), and this is where the project corrected
itself. The intuitive story is "stocks and bonds started moving together", and it
is measurably wrong for 2022: across ``2022-01-03 -> 2022-10-12`` the rolling
63-day SPY/TLT correlation averaged **-0.09** and was positive on only 31% of
days. The persistent flip is 2023 onward. A correlation-sign alarm therefore
fires roughly a year *after* the damage, which makes it a poor diagnostic and an
excellent chart -- so :func:`rolling_correlation` ships for the picture and
:func:`rolling_diversification_ratio` ships as the actual instrument.

The diversification ratio ``sum(w_i * sigma_i) / sigma_p`` is 1.0 for a portfolio
of one asset (or of perfectly correlated ones) and rises as the sleeves genuinely
offset each other. It degrades *through* the episode rather than after it:
measured on live data at 2.11 (2008), 1.78 (2020), 1.68 (2021), 1.59 (2022),
1.52 (2023). Correlation rising toward zero without changing sign is enough to
move it, which is exactly the mechanism the sign test cannot see.

Both diagnostics were run against live prices (2006-02-03 -> 2026-08-14) while
this module was written, and they reproduce the numbers ``config.py`` records:
:func:`rolling_diversification_ratio` gives 2.111 / 1.790 / 1.678 / 1.586 / 1.521
for 2008 / 2020 / 2021 / 2022 / 2023, and :func:`regime_summary` puts the 2022
window at a mean correlation of **-0.092, positive on 30.6% of days**. That is
the whole argument in two rows: by the correlation test nothing had happened yet,
and the ratio was already at its lowest level in the sample.

Everything here consumes ``BacktestResult`` and plain return frames, so it runs
identically against ``tests.fixtures`` and against the live Phase 5 run. Per the
"Data reality check" in TASKS.md the fixtures are shape-accurate but not
magnitude-accurate: build against them freely, quote nothing from them.

Run a self-check from the repo root::

    .venv/bin/python -m src.stress
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CORRELATION_PAIR,
    CORRELATION_WINDOW_DAYS,
    CRISIS_WINDOWS,
    DIVERSIFICATION_SHRINKAGE,
    DIVERSIFICATION_WINDOW_DAYS,
    NOMINAL_BOND_TICKER,
    REPORTS_DIR,
    TRADING_DAYS_PER_YEAR,
)
from .covariance import estimate_covariance
from .data import asof_slice
from .metrics import (
    Window,
    resolve_window,
    slice_dated,
    slice_to_window,
    summarize_window,
    window_label,
)
from .types import BacktestResult

log = logging.getLogger(__name__)

CRISIS_TABLE_FILENAME = "crisis_drawdowns.csv"

# Column order for the crisis table. Fixed here rather than left to dict
# insertion order because this frame is written to CSV and read by a human: the
# two columns the analysis turns on -- what it lost and how much of the book was
# in bonds while it lost it -- belong next to each other and near the front.
CRISIS_TABLE_COLUMNS: tuple[str, ...] = (
    "window",
    "strategy",
    "total_return",
    "max_drawdown",
    "capital_weight_bonds",
    "risk_weight_bonds",
    "mean_leverage",
    "annual_volatility",
    "sharpe_ratio",
    "cagr",
    "hit_rate",
    "n_periods",
    "covered_start",
    "covered_end",
    "fully_covered",
)


class StressError(ValueError):
    """A stress computation was asked for something it cannot honestly produce.

    Raised rather than returning NaN. A NaN row in the crisis table reads as "the
    strategy was flat through the GFC", which is a finding; "this was never
    measured" is not, and the two must not look alike.
    """


# ---------------------------------------------------------------------------
# Step 24 -- crisis decomposition and the cross-strategy table
# ---------------------------------------------------------------------------
def _as_result_mapping(
    results: Mapping[str, BacktestResult] | Sequence[BacktestResult] | BacktestResult,
) -> dict[str, BacktestResult]:
    """Accept a single result, a sequence of them, or an explicit name -> result map.

    A sequence is keyed by each result's own ``meta['strategy']``. Duplicate names
    raise: two strategies sharing a label would silently overwrite each other's
    row in the pivoted drawdown table, which is a wrong table rather than a
    missing one.
    """
    if isinstance(results, BacktestResult):
        return {results.strategy: results}
    if isinstance(results, Mapping):
        return dict(results)

    out: dict[str, BacktestResult] = {}
    for result in results:
        if result.strategy in out:
            raise StressError(
                f"two results share the strategy name {result.strategy!r}; pass an "
                "explicit {name: result} mapping so the rows stay distinguishable"
            )
        out[result.strategy] = result
    return out


def bond_exposure(result: BacktestResult, ticker: str = NOMINAL_BOND_TICKER) -> tuple[float, float]:
    """(mean capital weight, mean risk-contribution share) in the bond sleeve.

    The pair rather than either one alone, because their divergence is the
    project's thesis in a single row: 60/40 holds 40% of its *capital* in TLT and
    much less than 40% of its *risk*; an ERC portfolio holds far more capital
    there in order to hold exactly 1/N of the risk. In 2022 it was the capital
    number that determined the loss, because what happened was a price move, not
    a volatility event.

    A strategy whose universe genuinely excludes the sleeve gets ``nan`` and a
    warning rather than an exception -- an equity-only comparator is a legitimate
    thing to put in this table, and it has no bond weight to report.
    """
    if ticker not in result.weights.columns:
        log.warning(
            "%s has no %s column, so its bond exposure is reported as nan "
            "(universe: %s).", result.strategy, ticker, list(result.weights.columns),
        )
        return float("nan"), float("nan")
    return (
        float(result.weights[ticker].mean()),
        float(result.risk_contributions[ticker].mean()),
    )


def crisis_decomposition(
    results: Mapping[str, BacktestResult] | Sequence[BacktestResult] | BacktestResult,
    windows: Mapping[str, tuple[str, str]] | Sequence[Window] | None = None,
    bond_ticker: str = NOMINAL_BOND_TICKER,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """One row per (window, strategy): what it did, and what it was holding.

    Parameters
    ----------
    results:
        A ``BacktestResult``, a sequence of them, or a ``{name: result}`` mapping.
        Names are what the table's ``strategy`` column carries.
    windows:
        Defaults to ``config.CRISIS_WINDOWS``. Accepts any mapping of label ->
        (start, end), or a sequence of window names / ``(start, end)`` pairs --
        ``config.REGIME_SHIFT_WINDOW`` goes straight in, open end and all.

    Notes
    -----
    The slicing is :func:`src.metrics.slice_to_window`'s, which anchors the
    rebalance frames to the **last rebalance at or before the window opens**
    rather than to the window itself. That matters here beyond contract
    bookkeeping: the allocation in force on the window's first day is the one the
    window's loss should be attributed to, so it is the one whose bond weight
    this reports.

    ``mean_leverage`` is in the table for the same reason the bond weight is: risk
    parity's 2022 loss is a *levered* loss, and a table setting it beside four
    unlevered benchmarks without saying so would be comparing different amounts
    of money at risk and calling it a strategy comparison.

    A strategy whose data does not reach a window is skipped with a warning
    rather than dropping the whole table -- benchmarks and the engine can have
    different start dates during development, and a half-built comparison is
    still worth looking at as long as the gap is *visible*. Everything that is
    reported carries ``fully_covered`` so a partial window can never be quoted as
    a whole one.
    """
    result_map = _as_result_mapping(results)
    if not result_map:
        raise StressError("no results to decompose")

    if windows is None:
        windows = CRISIS_WINDOWS
    if isinstance(windows, Mapping):
        labelled = list(windows.items())
    else:
        labelled = [
            (w, CRISIS_WINDOWS[w]) if isinstance(w, str) and w in CRISIS_WINDOWS
            else (window_label(*resolve_window(w)), w)
            for w in windows
        ]
    if not labelled:
        raise StressError("no windows to decompose")

    rows: list[dict] = []
    for label, window in labelled:
        for name, result in result_map.items():
            # Sliced once, then read three ways. The metrics, the bond weights and
            # the mean leverage all have to describe the *same* set of rebalances
            # -- slicing separately for each would let them drift apart the moment
            # the anchoring rule in metrics.slice_to_window changes.
            try:
                sliced = slice_to_window(result, window)
            except (ValueError, KeyError) as exc:
                log.warning(
                    "%s has no usable data in %s (%s); the row is omitted rather than "
                    "reported as a flat window.", name, label, exc,
                )
                continue

            capital, risk = bond_exposure(sliced, ticker=bond_ticker)
            row = summarize_window(sliced, periods_per_year=periods_per_year, name=label)
            # summarize_window's row identifies itself with `name`, which is the
            # window label -- the same string this table already carries in
            # `window`. Two identical columns in a CSV a human reads is an
            # invitation to wonder which one is authoritative, so the generic one
            # goes and the specific one stays.
            row.pop("name", None)
            rows.append(
                {
                    **row,
                    "window": label,
                    "strategy": name,
                    "capital_weight_bonds": capital,
                    "risk_weight_bonds": risk,
                    "mean_leverage": float(sliced.leverage.mean()),
                }
            )

    if not rows:
        raise StressError(
            "every (strategy, window) pair was skipped; the results and the windows "
            "do not overlap at all"
        )

    table = pd.DataFrame(rows)
    ordered = [c for c in CRISIS_TABLE_COLUMNS if c in table.columns]
    remaining = [c for c in table.columns if c not in ordered]
    return table[ordered + remaining].reset_index(drop=True)


def drawdown_table(
    results: Mapping[str, BacktestResult] | Sequence[BacktestResult] | BacktestResult,
    windows: Mapping[str, tuple[str, str]] | Sequence[Window] | None = None,
    value: str = "max_drawdown",
    bond_ticker: str = NOMINAL_BOND_TICKER,
) -> pd.DataFrame:
    """The cross-strategy view: windows down the side, strategies across the top.

    A pivot of :func:`crisis_decomposition` on ``value`` (``max_drawdown`` by
    default, ``total_return`` being the other one worth looking at). Strategy
    columns keep the order they were passed in, so the engine can be put first
    and the comparison reads left to right.

    This is the readable form; the long form from :func:`crisis_decomposition` is
    the one to write to disk, because it carries the bond-weight and coverage
    columns that the pivot cannot.
    """
    # Resolved ONCE, and the mapping is what gets passed on. This used to call
    # _as_result_mapping again after the decomposition, which quietly produced a
    # table with zero strategy columns whenever `results` was a one-shot iterable
    # -- the decomposition consumed it, the second pass saw an empty sequence, and
    # the column order it derived was []. An empty table that raises nothing is
    # precisely the silent-wrong-output failure this repo does not tolerate.
    result_map = _as_result_mapping(results)
    long = crisis_decomposition(result_map, windows, bond_ticker=bond_ticker)
    if value not in long.columns:
        raise StressError(
            f"{value!r} is not a column of the decomposition; available: "
            f"{sorted(long.columns)}"
        )
    order = list(result_map)
    pivot = long.pivot(index="window", columns="strategy", values=value)
    pivot = pivot[[c for c in order if c in pivot.columns]]
    pivot.columns.name = None

    # Preserve the window order the caller gave rather than pandas' alphabetical
    # sort: the crisis windows are chronological, and a table that reads
    # COVID/GFC/rate-shock invites the wrong story about what followed what.
    return pivot.reindex([w for w in long["window"].unique() if w in pivot.index])


def write_crisis_table(
    table: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Write the decomposition to ``reports/crisis_drawdowns.csv`` and return the path.

    Step 24's "done when". Written with ``index=False`` because the long form's
    identity is the ``(window, strategy)`` pair rather than a row number, and a
    CSV whose first column is an unnamed counter is worse to read in a
    spreadsheet than one without it.
    """
    destination = Path(path) if path is not None else REPORTS_DIR / CRISIS_TABLE_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    log.info("Wrote %d crisis rows to %s", len(table), destination)
    return destination


# ---------------------------------------------------------------------------
# Step 25 -- the regime diagnostics
# ---------------------------------------------------------------------------
def rolling_correlation(
    returns: pd.DataFrame,
    pair: tuple[str, str] = CORRELATION_PAIR,
    window_days: int = CORRELATION_WINDOW_DAYS,
) -> pd.Series:
    """Rolling correlation of two sleeves -- **for the chart, not for the alarm**.

    This is the intuitive picture of risk parity's premise and it is genuinely
    worth plotting, but read what it does and does not say. Over the 2022
    drawdown the SPY/TLT version of this series averaged -0.09 and spent 69% of
    its days negative: by its own sign test, nothing was wrong. What broke was
    that bonds fell harder than equities *while still being negatively
    correlated*. Correlation is the wrong instrument for that, and
    :func:`rolling_diversification_ratio` is the right one.

    Returns are dropped where the window is incomplete rather than
    ``min_periods``-padded: a 63-day correlation computed off 12 days is a
    different, much noisier estimator wearing the same label.
    """
    left, right = pair
    missing = [t for t in pair if t not in returns.columns]
    if missing:
        raise StressError(
            f"cannot compute the {left}/{right} correlation: {missing} not in "
            f"{list(returns.columns)}"
        )
    if window_days < 2:
        raise StressError(f"a correlation window needs at least 2 days, got {window_days}")

    frame = returns.sort_index()
    series = (
        frame[left]
        .rolling(window_days, min_periods=window_days)
        .corr(frame[right])
        .dropna()
    )
    if series.empty:
        raise StressError(
            f"{len(frame)} observations is fewer than the {window_days}-day correlation "
            "window, so no correlation can be computed"
        )
    return series.rename(f"corr_{left}_{right}")


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """``sum(w_i * sigma_i) / sigma_p`` -- the diagnostic step 25 keys off.

    The weighted average of the sleeves' *standalone* volatilities over the
    volatility they actually produce together. It is exactly 1.0 when the sleeves
    are perfectly correlated (nothing offsets anything) and rises as they
    genuinely diversify, so it is a direct measure of how much risk the
    portfolio's structure is cancelling.

    Why this rather than a correlation. It aggregates the whole covariance
    *including the volatilities*, so it moves when correlations drift toward zero
    without changing sign -- which is what 2022 was -- and it moves when the
    sleeves' relative volatilities shift the weights around. A pairwise
    correlation sees neither. It is also weight-dependent by construction, which
    is the honest framing: diversification is a property of the portfolio you
    hold, not of the asset class list.
    """
    w = np.asarray(weights, dtype="float64").ravel()
    c = np.asarray(cov, dtype="float64")
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise StressError(f"cov must be square, got shape {c.shape}")
    if w.shape[0] != c.shape[0]:
        raise StressError(f"weights has length {w.shape[0]} but cov is {c.shape[0]}x{c.shape[0]}")

    variances = np.diag(c)
    if np.any(variances < 0.0):
        raise StressError("covariance has a negative variance on the diagonal")

    weighted_vol = float(w @ np.sqrt(variances))
    portfolio_variance = float(w @ c @ w)
    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0.0:
        raise StressError(
            f"portfolio variance is {portfolio_variance}, so the diversification ratio "
            "is undefined; the covariance estimate is degenerate"
        )
    return weighted_vol / float(np.sqrt(portfolio_variance))


def diagnostic_covariance(window: pd.DataFrame) -> np.ndarray:
    """The covariance the diversification ratio is measured on.

    ``src.covariance.estimate_covariance`` with shrinkage turned off -- see
    ``config.DIVERSIFICATION_SHRINKAGE`` for the measurement that set it. Briefly:
    shrinkage stabilises a matrix that is about to be *inverted*, the ratio only
    *reads* one, and pulling the off-diagonals toward a diagonal target makes the
    sleeves look more independent than they were. That is a bias in precisely the
    direction this diagnostic exists to detect, and it moved the published annual
    path by about +0.03 throughout.

    PSD repair and annualisation are kept, so this is still the project's
    estimator rather than a second implementation of one.
    """
    return estimate_covariance(window, shrinkage=DIVERSIFICATION_SHRINKAGE)


class _ConditionNumberTally(logging.Filter):
    """Collapse ``src.covariance``'s per-window condition-number warnings into a count.

    Not a silencer, and the distinction matters here. ``estimate_covariance``
    warns whenever the condition number exceeds ``MAX_CONDITION_NUMBER``, which
    is right for the ~230 rebalances of a backtest and wrong for the ~5000
    windows of a daily diagnostic sweep: unshrunk 63-day windows of four sleeves
    trip it routinely, and five thousand identical lines is how a genuine warning
    elsewhere in the run gets scrolled past.

    The count is logged once at the end instead. It is also much less alarming
    than it looks -- a high condition number matters because it makes a matrix
    *inverse* unstable, and the diversification ratio never inverts anything; it
    reads a weighted sum and a quadratic form, both of which are perfectly
    well-conditioned on the same matrix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if "condition number" in record.getMessage():
            self.count += 1
            return False
        return True


def _inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """``w_i proportional to 1/sigma_i``, the weighting the live path was measured on.

    Deliberately not imported from ``src.riskparity.naive_inv_vol``, which raises
    on a non-positive variance because a solver cannot proceed past one. Here a
    degenerate day should drop out of a long diagnostic series, not abort it, so
    the caller gets ``nan`` for that day instead.
    """
    variances = np.diag(np.asarray(cov, dtype="float64"))
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        return np.full(len(variances), np.nan)
    inverse = 1.0 / np.sqrt(variances)
    return inverse / inverse.sum()


def rolling_diversification_ratio(
    returns: pd.DataFrame,
    weights: pd.DataFrame | pd.Series | np.ndarray | None = None,
    window_days: int = DIVERSIFICATION_WINDOW_DAYS,
    cov_estimator=None,
    step_days: int = 1,
) -> pd.Series:
    """The diversification ratio through time, on a rolling covariance window.

    Parameters
    ----------
    weights:
        What to measure the ratio *of*. A ``BacktestResult.weights`` frame is
        forward-filled onto the daily index (with the one-day shift the rest of
        the project uses, so day ``t``'s ratio describes the allocation actually
        in force on day ``t``). ``None`` -- the default -- uses inverse-volatility
        weights recomputed from each window's own covariance, which is the
        weighting the live path in ``config.py`` was measured on and the one that
        makes the series a property of the *market* rather than of one strategy's
        solver.
    cov_estimator:
        Defaults to :func:`diagnostic_covariance` -- the project's estimator with
        shrinkage off, which is what the published annual path was measured on.
        Pass ``src.covariance.estimate_covariance`` to see the ratio through the
        exact matrix the solver optimises against instead.
    step_days:
        Compute every n-th day and forward-fill between. The default recomputes
        daily; raising it is the cheap way to draw the chart over a long sample.

    Notes
    -----
    Each point uses only the trailing ``window_days`` observations at or before
    that date -- routed through ``src.data.asof_slice``, the project's single
    look-ahead chokepoint, rather than re-deriving the ``<= t`` guard here.

    Verified against live prices with the defaults, which is step 25's "done
    when". Calendar-year means over 2006-02-03 -> 2026-08-14 come out at **2.111
    (2008), 2.179 (2012 peak), 1.790 (2020), 1.678 (2021), 1.586 (2022), 1.521
    (2023)**, reproducing the path in ``config.py`` to three decimals. Read the
    2021 -> 2023 run: the ratio falls for three consecutive years, and it is
    already at its lowest level in the sample *during* 2022. The correlation sign
    test does not turn over until 2023.
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise StressError("returns must have a DatetimeIndex")
    if window_days < 2:
        raise StressError(f"a covariance window needs at least 2 days, got {window_days}")
    if step_days < 1:
        raise StressError(f"step_days must be at least 1, got {step_days}")

    frame = returns.sort_index()
    if len(frame) < window_days:
        raise StressError(
            f"{len(frame)} observations is fewer than the {window_days}-day window"
        )

    estimate = cov_estimator or diagnostic_covariance
    aligned = _align_weights(weights, frame)

    dates = frame.index[window_days - 1 :: step_days]
    values: list[float] = []

    tally = _ConditionNumberTally()
    covariance_log = logging.getLogger("src.covariance")
    covariance_log.addFilter(tally)
    try:
        for date in dates:
            window = asof_slice(frame, date).iloc[-window_days:]
            try:
                cov = estimate(window)
                w = (
                    _inverse_vol_weights(cov)
                    if aligned is None
                    else aligned.loc[date].to_numpy()
                )
                values.append(
                    np.nan if not np.all(np.isfinite(w)) else diversification_ratio(w, cov)
                )
            except (StressError, ValueError) as exc:
                # One degenerate window should leave a hole in a diagnostic
                # series, not abort it -- but a silent hole is how a chart ends
                # up with an interpolated line nobody knows is interpolated, so
                # it is logged.
                log.warning("Diversification ratio undefined at %s (%s)", date.date(), exc)
                values.append(float("nan"))
    finally:
        covariance_log.removeFilter(tally)

    if tally.count:
        log.info(
            "%d of %d diversification windows exceeded MAX_CONDITION_NUMBER. Expected on "
            "unshrunk short windows, and harmless here: the ratio reads the covariance "
            "rather than inverting it.",
            tally.count, len(dates),
        )

    series = pd.Series(values, index=dates, name="diversification_ratio")
    if step_days > 1:
        series = series.reindex(frame.index[window_days - 1 :]).ffill()
    return series


def _align_weights(
    weights: pd.DataFrame | pd.Series | np.ndarray | None, returns: pd.DataFrame
) -> pd.DataFrame | None:
    """Put a weight specification onto the daily return index, or return ``None``.

    A rebalance-indexed frame is forward-filled and shifted one day, matching
    ``src.benchmarks._unlevered_result``: the allocation chosen at ``t`` is the
    one in force *from* ``t+1``, so measuring day ``t``'s ratio against the
    weights decided that same evening would be the same one-day look-ahead the
    whole project is arranged to avoid.
    """
    if weights is None:
        return None

    if isinstance(weights, pd.DataFrame):
        missing = [c for c in weights.columns if c not in returns.columns]
        if missing:
            raise StressError(f"weights carry columns {missing} absent from the returns")
        aligned = weights.reindex(returns.index).ffill().shift(1)
        # A sleeve the strategy does not hold is a real 0.0 -- 60/40 declines DBC.
        # A *day before the first rebalance* is not: there is no allocation yet.
        # Filling both with 0.0 (which this used to do) fabricated an all-zero
        # portfolio for every such day, whose variance is 0, so each one took the
        # degenerate-covariance path and logged a warning about an estimate that
        # was never the problem -- 40 of them on a 2006-09 fixture. Left as NaN,
        # they drop out of the series cleanly, which is what "not invested yet"
        # actually means.
        absent = [c for c in returns.columns if c not in weights.columns]
        aligned = aligned.reindex(columns=returns.columns)
        if absent:
            aligned[absent] = aligned[absent].fillna(0.0)
        return aligned

    values = np.asarray(
        weights.reindex(returns.columns) if isinstance(weights, pd.Series) else weights,
        dtype="float64",
    ).ravel()
    if values.shape[0] != returns.shape[1]:
        raise StressError(
            f"a constant weight vector must have one entry per asset "
            f"({returns.shape[1]}), got {values.shape[0]}"
        )
    return pd.DataFrame(
        np.tile(values, (len(returns), 1)), index=returns.index, columns=returns.columns
    )


def annual_means(series: pd.Series) -> pd.Series:
    """Calendar-year means of a daily diagnostic series.

    The form the live diversification-ratio path in ``config.py`` is quoted in --
    2.11 (2008), 1.78 (2020), 1.68 (2021), 1.59 (2022), 1.52 (2023) -- so a live
    run can be checked against it directly. Annual rather than smoothed because a
    year is the unit the narrative is told in, and because a mean over a fixed
    calendar boundary is reproducible by anyone with the same data.
    """
    clean = series.dropna()
    if clean.empty:
        raise StressError("no finite observations to average")
    grouped = clean.groupby(clean.index.year).mean()
    grouped.index.name = "year"
    return grouped.rename(series.name)


def correlation_regime_table(
    returns: pd.DataFrame,
    pair: tuple[str, str] = CORRELATION_PAIR,
    window_days: int = CORRELATION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Per calendar year: mean rolling correlation and the fraction of days positive.

    Both columns, because either alone misleads. A year averaging -0.05 sounds
    like diversification intact; if 48% of its days were positive, the sign has
    stopped meaning anything. This is the table that separates 2022 (mean -0.09,
    31% positive -- still a hedge, and it still failed) from 2023 onward (66-70%
    positive -- a different regime), and it is the evidence behind
    ``config.REGIME_SHIFT_WINDOW``.
    """
    series = rolling_correlation(returns, pair=pair, window_days=window_days)
    grouped = series.groupby(series.index.year)
    table = pd.DataFrame(
        {
            "mean_correlation": grouped.mean(),
            "fraction_positive": grouped.apply(lambda s: float((s > 0).mean())),
            "n_days": grouped.size().astype(int),
        }
    )
    table.index.name = "year"
    return table


def regime_summary(
    returns: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    pair: tuple[str, str] = CORRELATION_PAIR,
    windows: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per window: what correlation said, and what the ratio said.

    The two diagnostics side by side, which is the comparison step 25 exists to
    make. In 2008 and 2020 they agree. In 2022 they do not -- the correlation
    columns look survivable and the ratio is already down -- and that
    disagreement is the finding.
    """
    correlation = rolling_correlation(returns, pair=pair)
    ratio = rolling_diversification_ratio(returns, weights=weights)

    rows = []
    for label, window in (windows or CRISIS_WINDOWS).items():
        try:
            corr_window = slice_dated(correlation, window)
            ratio_window = slice_dated(ratio, window)
        except ValueError as exc:
            log.warning("%s is not covered by the data (%s); skipping.", label, exc)
            continue
        rows.append(
            {
                "window": label,
                "mean_correlation": float(corr_window.mean()),
                "fraction_positive": float((corr_window > 0).mean()),
                "mean_diversification_ratio": float(ratio_window.mean()),
                "end_diversification_ratio": float(ratio_window.dropna().iloc[-1])
                if ratio_window.notna().any() else float("nan"),
            }
        )
    if not rows:
        raise StressError("no window overlaps the data")
    return pd.DataFrame(rows).set_index("window")


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    from tests.fixtures import make_fake_result, make_fake_returns

    from .benchmarks import equal_weight, max_sharpe, naive_risk_parity, sixty_forty

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    pd.set_option("display.width", 140)

    fake_returns = make_fake_returns()
    strategies = {
        "Risk parity (fixture)": make_fake_result(strategy="Risk parity (fixture)"),
        "Equal weight": equal_weight(fake_returns),
        "60/40": sixty_forty(fake_returns),
        "Max Sharpe": max_sharpe(fake_returns),
        "Naive 1/vol": naive_risk_parity(fake_returns),
    }

    decomposition = crisis_decomposition(strategies)
    written = write_crisis_table(decomposition)

    print(f"crisis decomposition -> {written}  ({len(decomposition)} rows)")
    print(
        decomposition[
            ["window", "strategy", "total_return", "max_drawdown",
             "capital_weight_bonds", "mean_leverage"]
        ].to_string(index=False, float_format=lambda v: f"{v: .4f}")
    )

    print("\nmax drawdown by window and strategy")
    print(drawdown_table(strategies).to_string(float_format=lambda v: f"{v: .4f}"))

    print(f"\ncorrelation regime ({CORRELATION_PAIR[0]}/{CORRELATION_PAIR[1]}, "
          f"{CORRELATION_WINDOW_DAYS}-day), last 6 years")
    print(correlation_regime_table(fake_returns).tail(6).to_string(
        float_format=lambda v: f"{v: .3f}"))

    print("\ndiversification ratio, annual means (inverse-vol weighted)")
    ratio = rolling_diversification_ratio(fake_returns, step_days=5)
    print(annual_means(ratio).to_string(float_format=lambda v: f"{v: .3f}"))

    print("\nthe two diagnostics side by side")
    print(regime_summary(fake_returns).to_string(float_format=lambda v: f"{v: .3f}"))
    print(
        "\nFixture output -- shape-accurate, not magnitude-accurate. Nothing above is a\n"
        "finding; the live numbers come from the Phase 5 run (TASKS.md steps 32-34)."
    )
