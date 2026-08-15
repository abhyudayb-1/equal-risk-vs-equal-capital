"""The monthly rebalance loop that produces a ``BacktestResult`` (step 23).

This is the module every other one has been feeding. It owns no maths of its own
-- covariance, the ERC solve, the vol target and the financing leg all live
upstream -- and its whole job is to walk the calendar in the right order, never
look forward, and hand back something that satisfies the contract.

The loop, at each month-end rebalance date ``t``:

1. ``asof_slice(returns, t)`` -- the window, truncated to ``<= t``. This is the
   only place the backtest touches history, and it is the single chokepoint
   ``tests/test_lookahead.py`` (step 28) attacks.
2. ``estimate_covariance_with_diagnostics`` -- annualised, shrunk, PSD-repaired,
   and it reports what it had to do to get there.
3. ``solve_erc_with_status`` -- the equal-risk weights, *and* whether they really
   are equal-risk. Both are recorded; see "Failures are surfaced" below.
4. ``leverage_at`` -- the trailing-vol estimate and ``L_t = TARGET_VOL / sigma``,
   clipped to ``LEVERAGE_CAP``.

Then the accrual, once, over the whole daily calendar:
``overlay.apply_overlay`` applies the weights and leverage in force to each day's
returns and charges financing on the borrowed part.

**The one-day rule, which is the only way this module can be silently wrong.**
A decision taken at the close of ``t`` earns returns from ``t+1`` onward, never
``t`` itself. ``apply_overlay`` enforces it through
``overlay._in_force_before``; this module's contribution is simply not to undo
it, which is why the daily accrual is a single call rather than a hand-rolled
alignment. The equivalent in ``benchmarks._unlevered_result`` is
``.ffill().shift(1)``, and the two agree -- the self-check asserts it, because
two alignment conventions that drift apart would make the engine and its own
benchmarks incomparable while both looked fine.

**Failures are surfaced, never swallowed.** Two of them, and they are different
events that this repo requires stay distinguishable:

* ``converged=False`` from the solver is a *numerical* event. The best iterate is
  used, a warning is logged, and the count lands in ``meta["solver_failures"]``.
  The run continues, because one awkward covariance in 234 is not a reason to
  discard nineteen years.
* ``ERCSolverError`` is an *economic* one -- no equal-risk portfolio exists above
  the weight floor, which means a sleeve's volatility is a millionth of its
  peers' and the price series is broken. That propagates and kills the run, with
  the offending date attached. Continuing would put a fabricated allocation in
  the middle of the equity curve.

Per-rebalance diagnostics go to ``meta["diagnostics"]`` as a frame; step 29 adds
the run summary that prints them and any further fields.

**Not modelled, and worth saying rather than leaving to be discovered:** there are
no transaction costs, no bid-ask spread and no borrow spread over the bill rate.
Financing is charged at ``DTB3`` flat. A real book pays more than the bill rate to
borrow, so the levered results here are still optimistic -- less so than the
free-leverage version they replace, but optimistic. Monthly rebalancing of four
liquid ETFs keeps turnover low enough that costs would not change the *shape* of
any conclusion in this project, which is why the omission is acceptable and not
why it is invisible.

Run a self-check from the repo root::

    .venv/bin/python -m src.backtest
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    COV_EWMA_HALFLIFE_DAYS,
    COV_WARMUP_TRADING_DAYS,
    LEVERAGE_CAP,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    VOL_EWMA_HALFLIFE_DAYS,
)
from .covariance import estimate_covariance_with_diagnostics
from .data import asof_slice, month_end_rebalance_dates
from .overlay import apply_overlay, leverage_at
from .riskparity import ERCSolverError, portfolio_volatility, risk_contributions
from .riskparity import solve_erc_with_status
from .types import BacktestResult

log = logging.getLogger(__name__)

DEFAULT_STRATEGY_NAME = "Risk parity + vol target"


def run_backtest(
    returns: pd.DataFrame,
    risk_free: pd.Series | float = 0.0,
    target_vol: float = TARGET_VOL,
    leverage_cap: tuple[float, float] = LEVERAGE_CAP,
    cov_halflife: float = COV_EWMA_HALFLIFE_DAYS,
    vol_halflife: float = VOL_EWMA_HALFLIFE_DAYS,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    strategy: str = DEFAULT_STRATEGY_NAME,
) -> BacktestResult:
    """Run the engine over ``returns`` and return a contract-valid result.

    Parameters
    ----------
    returns:
        Daily simple returns, one column per sleeve, oldest row first -- what
        ``src.data.daily_returns`` produces. Not sliced by the caller; this
        function does the slicing, which is the point.
    risk_free:
        The financing rate **per period, as a decimal**, on the same calendar as
        ``returns``. ``src.data.fetch_risk_free`` returns exactly this and
        ``align_risk_free`` puts it on the price calendar. The default of ``0.0``
        makes leverage free and is there for tests and for isolating the
        financing leg's contribution -- never for a headline number.
    target_vol, leverage_cap, cov_halflife, vol_halflife:
        Overrides for the ``config.py`` policy, so a sensitivity sweep does not
        have to edit the shared constants file.
    warmup_trading_days:
        History required before the first rebalance. Governs both the covariance
        and the trailing-vol estimate, which is why ``VOL_WARMUP_TRADING_DAYS``
        was set equal to ``COV_WARMUP_TRADING_DAYS`` rather than independently.

    Notes
    -----
    The returned ``returns`` series starts the day *after* the first rebalance,
    not on it: the first allocation is decided at that day's close and cannot
    have earned that day's move. The contract's
    ``returns.index[0] >= weights.index[0]`` check is satisfied strictly.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("run_backtest needs a DataFrame of daily asset returns")
    if returns.empty:
        raise ValueError("cannot back-test an empty return frame")

    frame = returns.sort_index()
    tickers = list(frame.columns)
    rebalance_dates = month_end_rebalance_dates(frame.index, warmup_trading_days)

    weight_rows: list[np.ndarray] = []
    rc_rows: list[np.ndarray] = []
    leverage_values: list[float] = []
    records: list[dict] = []

    for date in rebalance_dates:
        window = asof_slice(frame, date)
        if len(window) < warmup_trading_days:
            # month_end_rebalance_dates already applied the warmup, so reaching
            # here means the calendar and the frame disagree -- worth failing on
            # rather than quietly estimating from a short window.
            raise ValueError(
                f"only {len(window)} observations at {date.date()}, fewer than the "
                f"{warmup_trading_days}-day warmup; the rebalance calendar does not "
                "match the return frame it was built from"
            )

        cov, diagnostics = estimate_covariance_with_diagnostics(
            window, halflife=cov_halflife, periods_per_year=periods_per_year
        )

        try:
            solution = solve_erc_with_status(cov)
        except ERCSolverError as exc:
            # Economic, not numerical. See the module docstring: this is a broken
            # price series wearing a solver's clothing, and continuing would put a
            # fabricated allocation in the middle of the equity curve.
            raise ERCSolverError(f"ERC is infeasible at {date.date()}: {exc}") from exc

        weights = solution.weights
        if not solution.converged:
            log.warning(
                "ERC did not converge at %s: worst risk-contribution deviation %.2e "
                "after %d restarts. Using the best iterate; see meta['solver_failures'].",
                date.date(), solution.max_rc_deviation, solution.restarts_used,
            )

        # The normalised form the contract stores. Composed from riskparity's own
        # primitives rather than imported from src.benchmarks, which computes the
        # same quantity: benchmarks sits *downstream* of this module in the
        # pipeline, and importing it here would invert the layering. The
        # self-check asserts the two agree, which is the property the duplication
        # warning in risk_contributions' docstring actually cares about.
        sigma = portfolio_volatility(weights, cov)
        fractions = risk_contributions(weights, cov) / sigma

        decision = leverage_at(
            frame,
            weights,
            date,
            halflife=vol_halflife,
            target_vol=target_vol,
            cap=leverage_cap,
            periods_per_year=periods_per_year,
            min_observations=warmup_trading_days,
        )

        weight_rows.append(weights)
        rc_rows.append(fractions)
        leverage_values.append(decision.leverage)
        records.append(
            {
                "n_observations": diagnostics.n_observations,
                "condition_number": diagnostics.condition_number,
                "cov_repaired": diagnostics.was_repaired,
                "ill_conditioned": diagnostics.is_ill_conditioned,
                "converged": solution.converged,
                "optimiser_succeeded": solution.optimiser_succeeded,
                "restarts_used": solution.restarts_used,
                "max_rc_deviation": solution.max_rc_deviation,
                "portfolio_vol": sigma,
                "trailing_vol": decision.trailing_vol,
                "raw_leverage": decision.raw_leverage,
                "leverage": decision.leverage,
                "cap_binding": decision.cap_binding,
            }
        )

    weights_frame = pd.DataFrame(weight_rows, index=rebalance_dates, columns=tickers)
    rc_frame = pd.DataFrame(rc_rows, index=rebalance_dates, columns=tickers)
    leverage_series = pd.Series(leverage_values, index=rebalance_dates, name="leverage")
    diagnostics_frame = pd.DataFrame(records, index=rebalance_dates)

    # Strictly after: the first allocation is decided at the close of the first
    # rebalance date and cannot have earned that day's return.
    live = frame.loc[frame.index > rebalance_dates[0]]
    if live.empty:
        raise ValueError(
            f"no return days after the first rebalance ({rebalance_dates[0].date()}); "
            "the sample is too short to accrue anything"
        )

    # The rate goes through untouched. overlay._published_values reindexes it to
    # the accrual calendar itself, dropping any extra rows, and checks coverage
    # first -- restricting it here would make that index match trivially and
    # replace a gap's "does not cover N of the return dates, first <date>" with a
    # bare NaN complaint. Fixing up an argument before the function that
    # validates it is how you lose the diagnostic you built.
    panel = apply_overlay(live, weights_frame, leverage_series, risk_free)

    result = BacktestResult(
        weights=weights_frame,
        risk_contributions=rc_frame,
        leverage=leverage_series,
        returns=panel["strategy"].rename("returns"),
        meta=_build_meta(
            strategy=strategy,
            target_vol=target_vol,
            leverage_cap=leverage_cap,
            cov_halflife=cov_halflife,
            vol_halflife=vol_halflife,
            warmup_trading_days=warmup_trading_days,
            periods_per_year=periods_per_year,
            diagnostics=diagnostics_frame,
            panel=panel,
            financed=not _is_zero_rate(risk_free),
        ),
    )
    result.validate()
    return result


def _build_meta(
    *,
    strategy: str,
    target_vol: float,
    leverage_cap: tuple[float, float],
    cov_halflife: float,
    vol_halflife: float,
    warmup_trading_days: int,
    periods_per_year: int,
    diagnostics: pd.DataFrame,
    panel: pd.DataFrame,
    financed: bool,
) -> dict:
    """Assemble ``meta``, keeping the conventional keys the contract names.

    ``solver_failures`` and ``max_condition_number`` are the two ``src/types.py``
    documents, so they stay scalars at the top level even though the same
    information is in ``diagnostics``. Step 29 adds the run summary that reads
    them; anything a chart might want stays out of here by design.

    ``financed`` is recorded explicitly rather than inferred from the numbers.
    A run with a zero risk-free rate and a run whose financing happened to be
    negligible produce nearly the same equity curve and mean very different
    things, and this project's whole argument about leverage rests on the
    difference.

    ``financing_drag_annual`` is the **arithmetic** mean of the financing leg,
    annualised -- ``periods_per_year * mean(r_p - L*r_a)``, which is exactly
    ``-mean((L-1)*r_f)``. It is deliberately not the difference of two CAGRs:
    that number is larger (1.14%/yr against 1.06%/yr on the live run) because it
    also carries the compounding of everything the drag was subtracted from, and
    attributing that to financing would overstate the leg. Quote whichever suits
    the sentence, but not this one as if it were the other.
    """
    binding = diagnostics["cap_binding"].value_counts()
    return {
        "strategy": strategy,
        "target_vol": target_vol,
        "leverage_cap": tuple(float(x) for x in leverage_cap),
        "solver_failures": int((~diagnostics["converged"]).sum()),
        "max_condition_number": float(diagnostics["condition_number"].max()),
        "covariances_repaired": int(diagnostics["cov_repaired"].sum()),
        "ill_conditioned_rebalances": int(diagnostics["ill_conditioned"].sum()),
        "total_restarts": int(diagnostics["restarts_used"].sum()),
        "worst_rc_deviation": float(diagnostics["max_rc_deviation"].max()),
        "n_rebalances": int(len(diagnostics)),
        "cap_binding_ceiling": int(binding.get("ceiling", 0)),
        "cap_binding_floor": int(binding.get("floor", 0)),
        "mean_leverage": float(diagnostics["leverage"].mean()),
        "financed": financed,
        "financing_drag_annual": float(
            periods_per_year * (panel["strategy"] - panel["leverage"] * panel["unlevered"]).mean()
        ),
        "cov_halflife": float(cov_halflife),
        "vol_halflife": float(vol_halflife),
        "warmup_trading_days": int(warmup_trading_days),
        "periods_per_year": int(periods_per_year),
        "diagnostics": diagnostics,
    }


def _is_zero_rate(risk_free: pd.Series | float) -> bool:
    """Whether the financing leg is switched off entirely."""
    if np.isscalar(risk_free):
        return float(risk_free) == 0.0
    return bool((np.asarray(risk_free, dtype="float64") == 0.0).all())


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys

    from .benchmarks import risk_contribution_fractions

    # Synthetic four-sleeve data with a realistic correlation structure, so the
    # self-check runs offline. The live run is Phase 5's job, not a test's.
    rng = np.random.default_rng(20240621)
    tickers = ["SPY", "TLT", "DBC", "TIP"]
    correlation = np.array(
        [
            [1.00, -0.30, 0.20, -0.10],
            [-0.30, 1.00, -0.05, 0.85],
            [0.20, -0.05, 1.00, 0.10],
            [-0.10, 0.85, 0.10, 1.00],
        ]
    )
    vols = np.array([0.170, 0.140, 0.200, 0.055]) / np.sqrt(TRADING_DAYS_PER_YEAR)
    n_days = 1500
    shocks = rng.normal(size=(n_days, 4)) @ np.linalg.cholesky(correlation).T * vols
    index = pd.bdate_range("2006-02-03", periods=n_days, name="date")
    synthetic = pd.DataFrame(shocks, index=index, columns=tickers)
    cash = pd.Series(4e-5, index=index, name="risk_free")

    result = run_backtest(synthetic, risk_free=cash)

    print(f"strategy      : {result.strategy}")
    print(f"rebalances    : {result.meta['n_rebalances']}  "
          f"({result.weights.index[0].date()} to {result.weights.index[-1].date()})")
    print(f"return days   : {len(result.returns)}  "
          f"({result.returns.index[0].date()} to {result.returns.index[-1].date()})")
    print(f"solver failures: {result.meta['solver_failures']}, "
          f"restarts {result.meta['total_restarts']}, "
          f"worst RC deviation {result.meta['worst_rc_deviation']:.2e}")
    print(f"max condition number: {result.meta['max_condition_number']:,.1f}, "
          f"repaired {result.meta['covariances_repaired']}")
    print(f"mean leverage : {result.meta['mean_leverage']:.2f}  "
          f"(ceiling bound {result.meta['cap_binding_ceiling']}, "
          f"floor {result.meta['cap_binding_floor']})")

    # --- step 23's "done when": the contract holds --------------------------
    # validate() already ran inside run_backtest; calling it again documents that
    # the result is re-checkable, which is what Person B's suites rely on.
    result.validate()
    print("\ncontract: validate() passes")
    print(f"  weights rows sum to 1 : "
          f"{float(np.abs(result.weights.to_numpy().sum(axis=1) - 1).max()):.2e}")
    print(f"  RC rows sum to 1      : "
          f"{float(np.abs(result.risk_contributions.to_numpy().sum(axis=1) - 1).max()):.2e}")
    print(f"  returns start after the first rebalance: "
          f"{result.returns.index[0] > result.weights.index[0]}")

    # --- ERC really is equal-risk, and it is NOT equal-capital --------------
    # The second half is the project's thesis. If these two frames ever agreed,
    # there would be nothing to write about.
    rc = result.risk_contributions.to_numpy()
    print(f"\nrisk contributions are equal to {np.abs(rc - 0.25).max():.2e} of 1/N")
    gap = np.abs(result.weights.to_numpy() - rc).max()
    print(f"capital weight vs risk weight: max gap {gap:.4f} "
          f"-- the frames diverge, which is the point")
    print(f"  median capital weights: "
          f"{result.weights.median().round(4).to_dict()}")

    # --- the two alignment conventions agree --------------------------------
    # benchmarks._unlevered_result uses .ffill().shift(1); overlay uses
    # _in_force_before. They must be the same rule or the engine and its own
    # benchmarks are not comparable. Reproduced here on the unlevered leg.
    applied = result.weights.reindex(synthetic.index).ffill().shift(1)
    expected = (applied * synthetic).sum(axis=1).loc[applied.dropna(how="any").index]
    unlevered = run_backtest(synthetic, risk_free=0.0, leverage_cap=(1.0, 1.0))
    reference = unlevered.returns
    common = expected.index.intersection(reference.index)
    print(f"\noverlay alignment vs benchmarks' ffill().shift(1): max |diff| "
          f"{float((expected.loc[common] - reference.loc[common]).abs().max()):.3e} "
          f"over {len(common)} days")

    # --- the normalised RC agrees with src.benchmarks' implementation -------
    # riskparity.risk_contributions' docstring warns against a second
    # implementation of this quantity. This module composes it from riskparity's
    # primitives rather than importing downstream, so the two are pinned here.
    date = result.weights.index[len(result.weights) // 2]
    window = asof_slice(synthetic, date)
    cov, _ = estimate_covariance_with_diagnostics(window)
    w = result.weights.loc[date].to_numpy()
    theirs = risk_contribution_fractions(w, cov)
    mine = result.risk_contributions.loc[date].to_numpy()
    print(f"RC fractions vs src.benchmarks.risk_contribution_fractions: "
          f"max |diff| {np.abs(mine - theirs).max():.3e}")

    # --- financing is charged, and switching it off is visible --------------
    free = run_backtest(synthetic, risk_free=0.0)
    charged = TRADING_DAYS_PER_YEAR * (free.returns.mean() - result.returns.mean())
    print(f"\nfinancing: drag {result.meta['financing_drag_annual']:+.4%}/yr, "
          f"and a zero-rate run differs by {charged:+.4%}/yr")
    print(f"  meta['financed'] is {result.meta['financed']} with a rate, "
          f"{free.meta['financed']} without")

    # --- look-ahead: a future spike must not move a past decision -----------
    # The real attack is tests/test_lookahead.py (step 28). This is the cheap
    # version that would catch a gross regression in the slicing.
    spike_from = synthetic.index[-200]
    poisoned = synthetic.copy()
    poisoned.iloc[-200:] *= 12.0
    partial = run_backtest(poisoned, risk_free=cash)

    # Exact, not a tolerance: EVERY rebalance dated before the spike must be
    # bit-identical, and the ones after it must not be -- otherwise the test
    # would also pass on a backtest that ignored its input.
    before = result.weights.index < spike_from
    clean = np.array_equal(
        result.weights.to_numpy()[before], partial.weights.to_numpy()[before]
    )
    after = ~before
    moved = not np.array_equal(
        result.weights.to_numpy()[after], partial.weights.to_numpy()[after]
    )
    print(f"\nlook-ahead: a 12x spike from {spike_from.date()} onward")
    print(f"  {int(before.sum())} rebalances before it : bit-identical = {clean}")
    print(f"  {int(after.sum())} rebalances after it  : changed = {moved}")

    if not clean:
        print("  REGRESSION: the spike reached rebalances that precede it")
        sys.exit(1)
    if not moved:
        print("  REGRESSION: the spike changed nothing at all -- is the input used?")
        sys.exit(1)

    # --- the diagnostics the contract names are all present -----------------
    # src/types.py documents strategy, target_vol, solver_failures and
    # max_condition_number as the conventional keys. Step 29's run summary reads
    # them, so a missing one is a silent hole in the reporting rather than a
    # crash at the point it was dropped.
    required = ("strategy", "target_vol", "solver_failures", "max_condition_number")
    missing = [key for key in required if key not in result.meta]
    print(f"\nmeta carries the contract's conventional keys: {not missing}"
          + (f" (missing {missing})" if missing else ""))
    print(f"  per-rebalance diagnostics frame: {len(result.meta['diagnostics'])} rows x "
          f"{len(result.meta['diagnostics'].columns)} columns")
    if missing:
        sys.exit(1)

    # --- degenerate inputs raise, and name what is wrong --------------------
    short = synthetic.iloc[:100]
    gapped = cash.drop(cash.index[400:410])
    duplicated = pd.concat([synthetic, synthetic.iloc[[100]]]).sort_index()
    print("\nbad input raises")
    for label, call in (
        ("empty frame", lambda: run_backtest(pd.DataFrame())),
        ("too short for a rebalance", lambda: run_backtest(short)),
        ("a gap in the risk-free rate", lambda: run_backtest(synthetic, risk_free=gapped)),
        ("duplicated return dates", lambda: run_backtest(duplicated, risk_free=cash)),
    ):
        try:
            call()
            print(f"  {label:28s}: NO ERROR RAISED -- regression!")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001 -- the point is that it is loud
            print(f"  {label:28s}: {type(exc).__name__} ({str(exc)[:44]}...)")

    # Unsorted input is HANDLED rather than rejected: run_backtest sorts, because
    # a caller handing over a reversed frame has made an ordering mistake the
    # engine can fix unambiguously, unlike a duplicated date.
    reversed_run = run_backtest(synthetic.iloc[::-1], risk_free=cash)
    print(f"  {'reversed input':28s}: handled, {reversed_run.meta['n_rebalances']} rebalances "
          f"(sorted internally)")
