"""Fixed-weight comparators for the risk parity engine.

These are the strategies risk parity has to beat to be worth building, and the
ones it has to *lose* to in 2022 for the write-up to be honest. Both produce the
same :class:`~src.types.BacktestResult` the engine does, so every metric, stress
test and chart treats them identically.

Equal weight is the sharpest illustration of the project's thesis. Holding 25%
of *capital* in each sleeve does not buy 25% of the *risk* in each sleeve --
equities are three times the volatility of TIP, so they dominate the portfolio's
variance while looking perfectly diversified on a pie chart. The gap between
``weights`` and ``risk_contributions`` in the results below is the entire reason
the ERC solver exists.

Neither benchmark is levered, so ``leverage`` is pinned at 1.0 throughout and the
financing cost never enters: with ``L = 1`` the overlay identity
``r_p = L*(r - r_f) + r_f`` collapses to ``r_p = r`` exactly. That is deliberate
-- benchmarks that quietly borrowed would not be benchmarks.

Run a self-check from the repo root::

    .venv/bin/python -m src.benchmarks
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .config import (
    COV_EWMA_HALFLIFE_DAYS,
    COV_WARMUP_TRADING_DAYS,
    SIXTY_FORTY_WEIGHTS,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from .data import asof_slice, month_end_rebalance_dates
from .types import BacktestResult

log = logging.getLogger(__name__)

# A covariance estimator maps a trailing return window (already sliced to <= t)
# to an annualized covariance matrix.
CovEstimator = Callable[[pd.DataFrame], np.ndarray]


def _ewma_cov(returns: pd.DataFrame) -> np.ndarray:
    """Annualized exponentially-weighted covariance of daily returns.

    A stand-in for ``src.covariance.ewma_covariance`` (Person A, TASKS.md step
    14), which does not exist yet. It is kept private and injectable rather than
    exported so that Phase 5 swaps it out by passing ``cov_estimator=`` at one
    call site instead of rewriting anything here -- and so nobody imports a
    second, competing covariance estimator out of this module in the meantime.

    Weights decay with ``COV_EWMA_HALFLIFE_DAYS``, most recent observation
    heaviest. The ``1 - sum(w^2)`` divisor is the reliability-weight bias
    correction -- the weighted analogue of ``ddof=1``. Without it the estimate is
    biased low, which would inflate every Sharpe ratio computed downstream.
    """
    x = returns.to_numpy(dtype="float64")
    n = len(x)
    decay = 0.5 ** (1.0 / COV_EWMA_HALFLIFE_DAYS)
    w = decay ** np.arange(n - 1, -1, -1, dtype="float64")
    w /= w.sum()

    deviations = x - (w @ x)
    cov = (deviations * w[:, None]).T @ deviations
    cov /= 1.0 - float((w**2).sum())
    return cov * TRADING_DAYS_PER_YEAR


def risk_contribution_fractions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of portfolio variance, summing to 1.0.

    ``RC_i = w_i * (cov @ w)_i / sigma_p`` is asset i's contribution in
    volatility units and sums to ``sigma_p``; dividing through by ``sigma_p``
    gives the fractions ``src.types.BacktestResult`` documents. Equivalently
    ``w_i * (cov @ w)_i / (w' cov w)``, which is what is computed here -- one
    fewer square root, and it sums to exactly 1.0 in floating point rather than
    to within a rounding error of it.

    Person A owns the volatility-unit version in ``src/riskparity.py`` (step 16);
    this is the normalised form the contract stores, not a competing
    implementation of the solver's objective.
    """
    marginal = cov @ weights
    variance = float(weights @ marginal)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            f"portfolio variance is {variance}, so risk contributions are undefined; "
            "the covariance estimate is degenerate"
        )
    return weights * marginal / variance


def fixed_weight(
    returns: pd.DataFrame,
    target: dict[str, float],
    strategy: str,
    cov_estimator: CovEstimator | None = None,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
) -> BacktestResult:
    """Backtest a constant target allocation, rebalanced monthly.

    Parameters
    ----------
    returns:
        Daily simple returns, one column per ticker. From
        ``src.data.daily_returns`` in a real run, or
        ``tests.fixtures.make_fake_returns`` while the engine is still being
        built.
    target:
        Ticker -> capital weight. Must sum to 1.0 and be long-only. Its keys
        become the columns of the result, in order. An explicit ``0.0`` is
        meaningful and is kept: it says "this strategy considered the sleeve and
        declined it", which is what keeps every benchmark on the same columns as
        the engine so they can be charted against each other.
    cov_estimator:
        Override the covariance used for ``risk_contributions``. Defaults to
        :func:`_ewma_cov`; pass ``src.covariance``'s estimator in Phase 5 so the
        benchmarks and the engine are judged on the same covariance.

    Notes
    -----
    Weights are held constant *between* rebalances rather than allowed to drift
    with returns, matching ``tests.fixtures.make_fake_result``. This is the
    convention to reconcile with ``src/backtest.py`` in Phase 5 -- see the module
    docstring of the fixture for why the one-day shift below is not negotiable.
    """
    if returns.empty:
        raise ValueError("cannot backtest an empty return frame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")

    tickers = list(target)
    missing = [t for t in tickers if t not in returns.columns]
    if missing:
        raise ValueError(
            f"{strategy} allocates to {missing}, which is not in the return frame "
            f"(has {list(returns.columns)})"
        )

    w = np.array([target[t] for t in tickers], dtype="float64")
    if (w < 0.0).any():
        raise ValueError(f"{strategy} target weights must be long-only: {target}")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"{strategy} target weights sum to {w.sum():.6f}, not 1.0")

    returns = returns[tickers].sort_index()
    estimate_cov = cov_estimator or _ewma_cov
    rebalance = month_end_rebalance_dates(returns.index, warmup_trading_days)

    # Risk contributions move even though capital weights do not -- that is the
    # comparison the project is built to make, so they are recomputed from the
    # covariance as known at each rebalance date.
    rc_rows = []
    for date in rebalance:
        trailing = asof_slice(returns, date)
        if len(trailing) < warmup_trading_days:
            raise ValueError(
                f"only {len(trailing)} observations at {date.date()}, fewer than the "
                f"{warmup_trading_days}-day warmup; the rebalance calendar is wrong"
            )
        rc_rows.append(risk_contribution_fractions(w, estimate_cov(trailing)))

    weights = pd.DataFrame(
        np.tile(w, (len(rebalance), 1)), index=rebalance, columns=tickers
    )
    risk_contributions = pd.DataFrame(rc_rows, index=rebalance, columns=tickers)
    leverage = pd.Series(1.0, index=rebalance, name="leverage")

    # Decision at t earns the return from t to t+1. Dropping the shift would let
    # the allocation collect the move of the very day it was chosen on.
    applied = weights.reindex(returns.index).ffill().shift(1)
    daily = (applied * returns).sum(axis=1)
    daily = daily.loc[applied.dropna(how="any").index].rename("returns")

    result = BacktestResult(
        weights=weights,
        risk_contributions=risk_contributions,
        leverage=leverage,
        returns=daily,
        meta={
            "strategy": strategy,
            "target_vol": None,  # unlevered by construction
            "solver_failures": 0,
            "target_weights": dict(target),
            "cov_estimator": getattr(estimate_cov, "__name__", repr(estimate_cov)),
        },
    )
    result.validate()
    return result


def equal_weight(
    returns: pd.DataFrame,
    tickers: tuple[str, ...] | list[str] | None = None,
    cov_estimator: CovEstimator | None = None,
    strategy: str = "Equal weight",
) -> BacktestResult:
    """Equal *capital* in every sleeve -- 25% each over the four-asset universe.

    The naive diversification everyone reaches for first, and the one the ERC
    solver is arguing against: equal capital concentrates risk in whichever
    sleeve happens to be most volatile.
    """
    tickers = list(tickers if tickers is not None else returns.columns)
    if not tickers:
        raise ValueError("equal_weight needs at least one ticker")
    return fixed_weight(
        returns,
        {t: 1.0 / len(tickers) for t in tickers},
        strategy,
        cov_estimator,
    )


def sixty_forty(
    returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
    universe: tuple[str, ...] | list[str] | None = None,
    cov_estimator: CovEstimator | None = None,
    strategy: str = "60/40",
) -> BacktestResult:
    """The institutional default: 60% equities, 40% nominal bonds, 0% elsewhere.

    The benchmark the 2022 result turns on, and the reason is *not* the one this
    docstring originally gave. Two corrections, both worth carrying:

    First, the mechanism. The stock/bond correlation did not flip positive during
    the 2022 drawdown -- measured over 2022-01-03 to 2022-10-12 it averaged -0.09
    and was positive on only 31% of days. What happened is that TLT fell 31.2%
    against SPY's 24.1%: the hedge lost more than the thing it hedged, with
    correlation still negative. Risk parity's exposure is worse than 60/40's here
    not because the sign changed but because it holds *more* bonds by capital
    (low volatility earns a large weight) and then levers the whole thing.

    Second, the ranking is an open question, not a known result. 60/40 was itself
    down roughly 23% over 2022, and which strategy loses by less is exactly what
    the crisis table in step 24 is built to answer. Stating the outcome in
    advance is how a backtest ends up confirming it -- so it is not stated here.

    The unallocated sleeves are carried as explicit zero columns over
    ``universe`` (default: every column in ``returns``) rather than dropped, so
    this result has the same shape as the engine's and the two can go into one
    weight chart without reindexing.
    """
    weights = dict(weights or SIXTY_FORTY_WEIGHTS)
    columns = list(universe if universe is not None else returns.columns)

    unplaceable = [t for t in weights if t not in columns]
    if unplaceable:
        raise ValueError(
            f"60/40 allocates to {unplaceable}, which is outside the universe {columns}"
        )

    return fixed_weight(
        returns, {t: weights.get(t, 0.0) for t in columns}, strategy, cov_estimator
    )


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    from tests.fixtures import make_fake_returns

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fake = make_fake_returns()

    for result in (equal_weight(fake), sixty_forty(fake)):
        print(f"\n{result.strategy}")
        print(f"  rebalances : {len(result.weights)}  "
              f"({result.weights.index[0].date()} -> {result.weights.index[-1].date()})")
        print(f"  daily obs  : {len(result.returns)}  "
              f"(first {result.returns.index[0].date()})")
        print(f"  leverage   : {result.leverage.min():.2f} - {result.leverage.max():.2f}")
        comparison = pd.DataFrame({
            "capital": result.weights.mean(),
            "mean risk": result.risk_contributions.mean(),
            "max risk": result.risk_contributions.max(),
        })
        print(comparison.round(4).to_string())

    print(f"\nBoth results validated against the contract "
          f"({UNIVERSE} universe, {len(fake)} daily observations).")
