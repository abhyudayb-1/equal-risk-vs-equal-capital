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

:func:`naive_risk_parity` is the fourth comparator and the one closest to the
engine: inverse-volatility weights, which is what most descriptions of risk
parity actually implement. It coincides with true equal risk contribution only
when all pairwise correlations are equal, so the gap between it and the engine is
what the correlation structure is worth.

No benchmark here is levered, so ``leverage`` is pinned at 1.0 throughout and the
financing cost never enters: with ``L = 1`` the overlay identity
``r_p = L*(r - r_f) + r_f`` collapses to ``r_p = r`` exactly. That is deliberate
-- benchmarks that quietly borrowed would not be benchmarks.

Run a self-check from the repo root::

    .venv/bin/python -m src.benchmarks
"""

from __future__ import annotations

import logging
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import (
    ANNUAL_RISK_FREE_RATE,
    COV_WARMUP_TRADING_DAYS,
    MAX_SHARPE_LOOKBACK_DAYS,
    SIXTY_FORTY_WEIGHTS,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from .covariance import estimate_covariance, nearest_psd
from .data import asof_slice, month_end_rebalance_dates
from .riskparity import naive_inv_vol
from .types import BacktestResult

log = logging.getLogger(__name__)

# A covariance estimator maps a trailing return window (already sliced to <= t)
# to an annualized covariance matrix.
CovEstimator = Callable[[pd.DataFrame], np.ndarray]


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
    rebalance_months: int = 1,
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
        ``src.covariance.estimate_covariance``, the same estimator the engine
        uses -- pass a different one only to deliberately judge the benchmarks
        against a different covariance than the engine saw.
    rebalance_months:
        Rebalance every ``n``-th month-end rather than every one, thinned from
        the front so the first date does not move. It matches the engine's
        argument of the same name and exists for the same reason: a sweep over
        rebalance frequency has to move *both* rules, or it is comparing two
        different windows as well as two different frequencies.

        Note what it does and does not change here. The capital weights are
        constant, so thinning the calendar leaves this benchmark's **returns**
        unchanged -- only the risk-contribution rows, which are estimated per
        rebalance, thin out with it. That is not a hole in the argument; see the
        note on this convention below.

    Notes
    -----
    Weights are held constant *between* rebalances rather than allowed to drift
    with returns, matching ``tests.fixtures.make_fake_result``. This is the
    convention to reconcile with ``src/backtest.py`` in Phase 5 -- see the module
    docstring of the fixture for why the one-day shift below is not negotiable.

    That convention has a consequence worth naming, because it is easy to read
    this function as modelling a monthly-rebalanced book and it does not.
    Holding the weights constant between rebalances is arithmetically a book
    traded back to target *every day*. ``src/backtest.py`` does the same with
    its own weights, so the comparison stays fair -- both rules get identical
    treatment -- but neither rule's **trading** is described by its weights
    frame, and the turnover either would really incur has to be reconstructed
    from the drift the frame does not show. ``src/sensitivity.py`` does exactly
    that, and says so, rather than reading 0.0% turnover off 1/N's constant
    weights and concluding that equal capital is free to run.
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

    if int(rebalance_months) != rebalance_months or rebalance_months < 1:
        raise ValueError(
            f"rebalance_months must be a whole number of months >= 1, got {rebalance_months!r}"
        )

    returns = returns[tickers].sort_index()
    estimate_cov = cov_estimator or estimate_covariance
    # Thinned from the front, so the first rebalance date -- and therefore the
    # warm-up and the start of the return series -- does not move with the
    # frequency. The engine does the same, for the same reason.
    rebalance = month_end_rebalance_dates(returns.index, warmup_trading_days)
    rebalance = rebalance[:: int(rebalance_months)]

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

    return _unlevered_result(
        weights,
        risk_contributions,
        returns,
        strategy,
        {
            "solver_failures": 0,
            "rebalance_months": int(rebalance_months),
            "target_weights": dict(target),
            "cov_estimator": getattr(estimate_cov, "__name__", repr(estimate_cov)),
        },
    )


def _unlevered_result(
    weights: pd.DataFrame,
    risk_contributions: pd.DataFrame,
    returns: pd.DataFrame,
    strategy: str,
    meta: dict,
) -> BacktestResult:
    """Accrue daily returns for an unlevered allocation and seal the contract.

    Shared by every benchmark in this module so the one-day alignment below is
    written once. It is the subtle part: getting it wrong is a look-ahead bug
    that still produces a plausible equity curve.
    """
    leverage = pd.Series(1.0, index=weights.index, name="leverage")

    # Decision at t earns the return from t to t+1. Dropping the shift would let
    # the allocation collect the move of the very day it was chosen on.
    applied = weights.reindex(returns.index).ffill().shift(1)
    daily = (applied * returns[weights.columns]).sum(axis=1)
    daily = daily.loc[applied.dropna(how="any").index].rename("returns")

    result = BacktestResult(
        weights=weights,
        risk_contributions=risk_contributions,
        leverage=leverage,
        returns=daily,
        meta={"strategy": strategy, "target_vol": None, **meta},
    )
    result.validate()
    return result


def equal_weight(
    returns: pd.DataFrame,
    tickers: tuple[str, ...] | list[str] | None = None,
    cov_estimator: CovEstimator | None = None,
    strategy: str = "Equal weight",
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
    rebalance_months: int = 1,
) -> BacktestResult:
    """Equal *capital* in every sleeve -- 25% each over the four-asset universe.

    The naive diversification everyone reaches for first, and the one the ERC
    solver is arguing against: equal capital concentrates risk in whichever
    sleeve happens to be most volatile.

    ``warmup_trading_days`` and ``rebalance_months`` are forwarded rather than
    left at their defaults because this is the rule the engine is measured
    against. A sweep that moved the engine's warm-up and not 1/N's would compare
    two rules over two different windows and report the difference as a
    sensitivity to the warm-up.
    """
    tickers = list(tickers if tickers is not None else returns.columns)
    if not tickers:
        raise ValueError("equal_weight needs at least one ticker")
    return fixed_weight(
        returns,
        {t: 1.0 / len(tickers) for t in tickers},
        strategy,
        cov_estimator,
        warmup_trading_days=warmup_trading_days,
        rebalance_months=rebalance_months,
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
    and was positive on only 31% of days. What happened is that TLT fell 29.3%
    against SPY's 24.5% (peak-to-trough, measured from the window's opening
    close): the hedge lost more than the thing it hedged, with correlation still
    negative. Risk parity's exposure is worse than 60/40's here
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


def naive_risk_parity(
    returns: pd.DataFrame,
    cov_estimator: CovEstimator | None = None,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
    strategy: str = "Naive 1/vol",
) -> BacktestResult:
    """Inverse-volatility weights, re-solved monthly -- "risk parity" as usually shipped.

    Weight proportional to ``1/sigma_i``, ignoring correlations entirely. This is
    what most descriptions of risk parity actually implement, and it equals the
    true equal-risk-contribution portfolio **only** when every pairwise
    correlation is the same. The four sleeves here are nowhere near that -- TLT
    correlates about -0.3 with SPY and about +0.85 with TIP -- so the two diverge
    visibly, and showing where is one of the write-up's exhibits.

    It belongs among the benchmarks rather than in the engine for exactly that
    reason: it is the strategy the project is arguing *against*, and the crisis
    table (TASKS.md step 24) needs it as a row. Unlevered, like every other
    comparator here, so the ERC engine's crisis losses are never being compared
    against a differently-geared book without the ``mean_leverage`` column saying
    so.

    The weights come from ``src.riskparity.naive_inv_vol`` rather than being
    recomputed here -- one implementation of a quantity, per the contract's whole
    reason for existing.
    """
    if returns.empty:
        raise ValueError("cannot backtest an empty return frame")

    returns = returns.sort_index()
    tickers = list(returns.columns)
    estimate_cov = cov_estimator or estimate_covariance
    rebalance = month_end_rebalance_dates(returns.index, warmup_trading_days)

    weight_rows, rc_rows = [], []
    for date in rebalance:
        cov = nearest_psd(estimate_cov(asof_slice(returns, date)))
        w = naive_inv_vol(cov)
        weight_rows.append(w)
        rc_rows.append(risk_contribution_fractions(w, cov))

    return _unlevered_result(
        pd.DataFrame(weight_rows, index=rebalance, columns=tickers),
        pd.DataFrame(rc_rows, index=rebalance, columns=tickers),
        returns,
        strategy,
        {
            "solver_failures": 0,
            "cov_estimator": getattr(estimate_cov, "__name__", repr(estimate_cov)),
        },
    )


def _tangency_weights(mean_excess: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    """Long-only maximum-Sharpe weights, or ``None`` if no such portfolio exists.

    Maximising ``(mu' w) / sqrt(w' S w)`` subject to ``sum(w) = 1, w >= 0`` is not
    convex as written. The standard change of variable ``y = w / (mu' w)`` turns
    it into a QP that is::

        minimise   y' S y
        subject to mu' y = 1,  y >= 0

    after which ``w = y / sum(y)`` is non-negative and sums to 1 by construction.
    The substitution requires ``mu' w > 0`` for some feasible w -- i.e. at least
    one asset with a positive trailing excess return. When every sleeve has lost
    money over the lookback, no long-only portfolio has a positive expected
    excess return, the constraint set is empty, and there is genuinely no
    tangency portfolio. That is a real state of the world, not a solver bug, and
    it happened across this universe in 2022; the caller substitutes minimum
    variance and records it.
    """
    if not (mean_excess > 0).any():
        return None

    n = len(mean_excess)
    y = cp.Variable(n, nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.Constant(cov))), [mean_excess @ y == 1.0]
    )
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate") or y.value is None:
        return None

    total = float(y.value.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return np.asarray(y.value, dtype="float64") / total


def _min_variance_weights(cov: np.ndarray) -> np.ndarray | None:
    """Long-only minimum-variance weights -- the fallback when no tangency exists."""
    n = cov.shape[0]
    w = cp.Variable(n, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.Constant(cov))), [cp.sum(w) == 1.0])
    problem.solve()
    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None
    return np.asarray(w.value, dtype="float64")


def _clean_weights(w: np.ndarray) -> np.ndarray:
    """Clip solver dust to zero and renormalise onto the simplex.

    Interior-point solvers return values like ``-3e-13``, which the contract's
    long-only check tolerates but which would propagate a negative capital weight
    into a chart. Renormalising after the clip keeps the row summing to 1.
    """
    w = np.clip(np.asarray(w, dtype="float64"), 0.0, None)
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("solver returned weights that do not sum to anything positive")
    return w / total


def max_sharpe(
    returns: pd.DataFrame,
    lookback_days: int = MAX_SHARPE_LOOKBACK_DAYS,
    cov_estimator: CovEstimator | None = None,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
    strategy: str = "Max Sharpe",
) -> BacktestResult:
    """Rolling long-only tangency portfolio, re-solved at every rebalance.

    The most demanding benchmark in the set, and the most flattering to itself:
    it optimises Sharpe directly on the trailing window, so it is the comparator
    that shows how much of an in-sample optimum survives out of sample. It is
    also the one that most needs the one-day shift in :func:`_unlevered_result`,
    because an optimiser fed today's close and paid today's return would look
    extraordinary.

    Expected returns are the trailing ``lookback_days`` mean, annualised, in
    excess of ``annual_risk_free_rate``. Covariance comes from ``cov_estimator``
    over all history at or before the rebalance date -- the same estimate the
    other benchmarks use, so the strategies differ in their objective rather
    than in what they were shown.

    Sample means are a notoriously noisy input to an optimiser; that is a
    property of the benchmark being modelled, not a defect here, and it is part
    of what the comparison is meant to expose.

    Notes
    -----
    Two states are counted in ``meta`` rather than hidden. ``tangency_fallbacks``
    is the number of rebalances with no positive trailing excess return anywhere
    in the universe, where minimum variance is substituted; ``solver_failures``
    is the number where cvxpy itself did not converge and the previous
    allocation was carried forward.
    """
    if returns.empty:
        raise ValueError("cannot backtest an empty return frame")
    if lookback_days < 2:
        raise ValueError(f"lookback_days must be at least 2, got {lookback_days}")

    returns = returns.sort_index()
    tickers = list(returns.columns)
    estimate_cov = cov_estimator or estimate_covariance
    rebalance = month_end_rebalance_dates(returns.index, warmup_trading_days)

    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    weight_rows, rc_rows = [], []
    fallbacks = failures = 0
    previous: np.ndarray | None = None

    for date in rebalance:
        trailing = asof_slice(returns, date)
        cov = nearest_psd(estimate_cov(trailing))

        window = trailing.iloc[-lookback_days:]
        mean_excess = (
            window.to_numpy(dtype="float64").mean(axis=0) - daily_rf
        ) * TRADING_DAYS_PER_YEAR

        w = _tangency_weights(mean_excess, cov)
        if w is None:
            # No long-only portfolio has a positive expected excess return over
            # the window. Minimum variance is the defensible substitute: it is
            # the one point on the frontier that does not depend on the means.
            w = _min_variance_weights(cov)
            if w is None:
                failures += 1
                if previous is None:
                    raise RuntimeError(
                        f"cvxpy failed on the first rebalance ({date.date()}) with no "
                        "previous allocation to carry forward"
                    )
                log.warning("cvxpy did not converge at %s; carrying the previous "
                            "allocation forward.", date.date())
                w = previous
            else:
                fallbacks += 1
                log.info("No positive trailing excess return at %s; using minimum "
                         "variance.", date.date())

        w = _clean_weights(w)
        previous = w
        weight_rows.append(w)
        rc_rows.append(risk_contribution_fractions(w, cov))

    if fallbacks:
        log.warning(
            "%d of %d rebalances had no tangency portfolio and fell back to minimum "
            "variance.", fallbacks, len(rebalance),
        )

    return _unlevered_result(
        pd.DataFrame(weight_rows, index=rebalance, columns=tickers),
        pd.DataFrame(rc_rows, index=rebalance, columns=tickers),
        returns,
        strategy,
        {
            "solver_failures": failures,
            "tangency_fallbacks": fallbacks,
            "lookback_days": lookback_days,
            "cov_estimator": getattr(estimate_cov, "__name__", repr(estimate_cov)),
        },
    )


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    from tests.fixtures import make_fake_returns

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fake = make_fake_returns()

    for result in (
        equal_weight(fake), sixty_forty(fake), max_sharpe(fake), naive_risk_parity(fake)
    ):
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

    print(f"\nAll results validated against the contract "
          f"({UNIVERSE} universe, {len(fake)} daily observations).")
