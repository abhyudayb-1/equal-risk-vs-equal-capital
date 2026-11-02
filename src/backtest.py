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

Per-rebalance diagnostics go to ``meta["diagnostics"]`` as a frame, and step 29
adds three ways the same failure reaches a human without anyone having to go
looking: a warning logged at the moment it happens, one logged at the end of the
run summarising whether anything went wrong at all, and
:func:`format_run_summary`, which marks the offending line with a leading ``!``
and names the dates. ``meta["run_health"]`` is the one-phrase version, and it is
what ``run_backtest.py``'s generic meta dump surfaces without needing to know
about any of this.

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
import textwrap

import numpy as np
import pandas as pd

from .config import (
    COV_EWMA_HALFLIFE_DAYS,
    COV_SHRINKAGE,
    COV_WARMUP_TRADING_DAYS,
    LEVERAGE_CAP,
    SATURATION_REPORT_FRACTION,
    SUMMARY_LINE_WIDTH,
    SUMMARY_MAX_DATES_LISTED,
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
    shrinkage: float = COV_SHRINKAGE,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
    rebalance_months: int = 1,
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
    target_vol, leverage_cap, cov_halflife, vol_halflife, shrinkage:
        Overrides for the ``config.py`` policy, so a sensitivity sweep does not
        have to edit the shared constants file. ``shrinkage`` goes straight to
        ``covariance.estimate_covariance_with_diagnostics``; at 0.0 the estimator
        is the raw EWMA, which is what the sweep in ``src/sensitivity.py`` uses
        to ask how much of ERC's edge is the shrinkage rather than the solver.
    warmup_trading_days:
        History required before the first rebalance. Governs both the covariance
        and the trailing-vol estimate, which is why ``VOL_WARMUP_TRADING_DAYS``
        was set equal to ``COV_WARMUP_TRADING_DAYS`` rather than independently.
    rebalance_months:
        Rebalance every ``n``-th month-end rather than every one. The calendar is
        still built month-by-month and then thinned, so the **first** rebalance
        date -- and therefore the warm-up and the start of the return series --
        does not move with this argument. That is what makes two frequencies
        comparable: a quarterly run that started a quarter later would differ for
        two reasons at once.

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

    if int(rebalance_months) != rebalance_months or rebalance_months < 1:
        raise ValueError(
            f"rebalance_months must be a whole number of months >= 1, got {rebalance_months!r}"
        )

    frame = returns.sort_index()
    tickers = list(frame.columns)
    rebalance_dates = month_end_rebalance_dates(frame.index, warmup_trading_days)
    # Thinned from the front, so the first date is the same at every frequency.
    rebalance_dates = rebalance_dates[:: int(rebalance_months)]

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
            window,
            halflife=cov_halflife,
            shrinkage=shrinkage,
            periods_per_year=periods_per_year,
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
            shrinkage=shrinkage,
            warmup_trading_days=warmup_trading_days,
            rebalance_months=int(rebalance_months),
            periods_per_year=periods_per_year,
            diagnostics=diagnostics_frame,
            panel=panel,
            financed=not _is_zero_rate(risk_free),
        ),
    )
    result.validate()

    # The aggregate, once, at the end. The per-rebalance warnings above fire at
    # the moment of failure and are easy to lose in a long run's log; this one
    # says whether the run as a whole is trustworthy, and it fires even for a
    # caller that never looks at meta or prints the summary. Silence here means
    # a clean run, which is the only thing silence is allowed to mean.
    health = result.meta["run_health"]
    if health != "clean":
        log.warning(
            "Run completed with issues: %s. See backtest.format_run_summary(result) "
            "and meta['diagnostics'] for the offending dates.",
            health,
        )
    else:
        log.info(
            "Run clean: %d rebalances, no solver failures, no covariance repairs.",
            len(rebalance_dates),
        )
    return result


def _build_meta(
    *,
    strategy: str,
    target_vol: float,
    leverage_cap: tuple[float, float],
    cov_halflife: float,
    vol_halflife: float,
    shrinkage: float,
    warmup_trading_days: int,
    rebalance_months: int,
    periods_per_year: int,
    diagnostics: pd.DataFrame,
    panel: pd.DataFrame,
    financed: bool,
) -> dict:
    """Assemble ``meta``, keeping the conventional keys the contract names.

    ``solver_failures`` and ``max_condition_number`` are the two ``src/types.py``
    documents, so they stay scalars at the top level even though the same
    information is in ``diagnostics``. Anything a chart might want stays out of
    here by design.

    **The split between ``diagnostics`` and the scalars around it (step 29).**
    ``diagnostics`` is the machine-readable record: one row per rebalance, every
    field, nothing summarised. Everything else here is for a *reader* -- a human
    or ``run_backtest.py``'s meta dump -- which is why the date fields are short
    strings rather than lists of ``Timestamp``. That is deliberate and not
    laziness: a list prints as its length in a generic dump, so
    ``solver_failure_dates`` as a list would appear as "2 entries" and the date
    of the failure -- the whole point of recording it -- would be invisible in
    the one place someone actually reads. Code that wants the dates as data
    should filter ``diagnostics``, which is lossless.

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
    n = int(len(diagnostics))
    failures = int((~diagnostics["converged"]).sum())
    repaired = int(diagnostics["cov_repaired"].sum())
    ill_conditioned = int(diagnostics["ill_conditioned"].sum())
    ceiling = int(binding.get("ceiling", 0))
    floor = int(binding.get("floor", 0))

    return {
        "strategy": strategy,
        "target_vol": target_vol,
        "leverage_cap": tuple(float(x) for x in leverage_cap),
        # -- the run at a glance, for a reader rather than a chart -------------
        # run_health is the one field to look at first: "clean", or a phrase
        # naming everything that went wrong. Step 29's bar is that a failed
        # rebalance is *visible in the output*, and a count buried among twenty
        # other counts is not visible -- it has to say so in words.
        "run_health": _run_health(failures, repaired, ill_conditioned),
        "first_rebalance": _iso(diagnostics.index[0]),
        "last_rebalance": _iso(diagnostics.index[-1]),
        "n_rebalances": n,
        "n_return_days": int(len(panel)),
        # -- solver -----------------------------------------------------------
        "solver_failures": failures,
        # The dates, not just the count. A run with two failures out of 234 is a
        # different object depending on *when* they were: October 2008 is the
        # covariance being genuinely hard, a quiet Tuesday is a bug.
        "solver_failure_dates": _flagged_dates(diagnostics.index, ~diagnostics["converged"]),
        "total_restarts": int(diagnostics["restarts_used"].sum()),
        "worst_rc_deviation": float(diagnostics["max_rc_deviation"].max()),
        "worst_rc_deviation_date": _iso(diagnostics["max_rc_deviation"].idxmax()),
        # -- covariance -------------------------------------------------------
        "max_condition_number": float(diagnostics["condition_number"].max()),
        # The max alone is one date's number and says nothing about the run it
        # came from; a median of 40 with a max of 5000 is a single hard window,
        # a median of 3000 is an estimator that should not be trusted anywhere.
        "median_condition_number": float(diagnostics["condition_number"].median()),
        "worst_condition_date": _iso(diagnostics["condition_number"].idxmax()),
        "covariances_repaired": repaired,
        "covariance_repair_dates": _flagged_dates(diagnostics.index, diagnostics["cov_repaired"]),
        # Read "0 ill-conditioned" as "nothing pathological", NOT as evidence
        # that the matrices are well conditioned -- at COV_SHRINKAGE = 0.10 this
        # flag is very nearly unreachable, and that is a property of the
        # shrinkage rather than of the data. Shrinking maps rho -> 0.9*rho, which
        # caps the shrunk *correlation* matrix's condition number at
        # (1 + (N-1)(1-d)) / d = 37 for four sleeves -- measured 37.00 at rho = 1
        # exactly, where the unshrunk matrix is 3.2e32. The covariance then adds
        # only the variance spread, so reaching MAX_CONDITION_NUMBER = 1e8 needs
        # a volatility ratio of about 1,600:1 between two sleeves (measured:
        # 1,000:1 gives 2.1e7, 10,000:1 gives 2.1e9). Over 20,000 random
        # covariances with realistic vols (3%-45%) the worst was 1,183 -- five
        # orders below the bar. The informative fields are the condition numbers
        # themselves, which is why the median is reported next to the max.
        "ill_conditioned_rebalances": ill_conditioned,
        "ill_conditioned_dates": _flagged_dates(diagnostics.index, diagnostics["ill_conditioned"]),
        # -- overlay ----------------------------------------------------------
        "cap_binding_ceiling": ceiling,
        "cap_binding_floor": floor,
        "cap_binding_fraction": (ceiling + floor) / n if n else 0.0,
        "mean_leverage": float(diagnostics["leverage"].mean()),
        "min_leverage": float(diagnostics["leverage"].min()),
        "max_leverage": float(diagnostics["leverage"].max()),
        # What the overlay *thought* the book's risk was, averaged. Read against
        # target_vol it says whether the strategy was levering up or down over
        # the sample; read against the realised volatility of `returns` (which
        # src.metrics computes -- see format_run_summary on why not here) it says
        # whether the overlay hit what it was aiming at.
        "mean_trailing_vol": float(diagnostics["trailing_vol"].mean()),
        "financed": financed,
        "financing_drag_annual": float(
            periods_per_year * (panel["strategy"] - panel["leverage"] * panel["unlevered"]).mean()
        ),
        "cov_halflife": float(cov_halflife),
        "shrinkage": float(shrinkage),
        "rebalance_months": int(rebalance_months),
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


# ---------------------------------------------------------------------------
# Step 29 -- the run summary. A failure nobody prints is a silent failure.
# ---------------------------------------------------------------------------
# Label column width. Local rather than in config.py because it is not a
# threshold anyone would look for or tune -- it is the width of the longest
# label below ("not modelled"), and changing it changes nothing but alignment.
_SUMMARY_LABEL_WIDTH: int = 13


def _iso(date) -> str:
    """A date as ``YYYY-MM-DD``, for a field a human reads."""
    return pd.Timestamp(date).date().isoformat()


def _as_float(value) -> float | None:
    """``float(value)``, or ``None`` if the field is absent or not a number.

    ``meta`` is free-form and shared with the benchmarks, which set
    ``target_vol`` to an explicit ``None`` because an unlevered strategy has no
    target. ``"target_vol" in meta`` is therefore true while the value is
    unformattable, which is precisely the case a presence check misses.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flagged_dates(
    index: pd.DatetimeIndex, mask, limit: int = SUMMARY_MAX_DATES_LISTED
) -> str:
    """The dates where ``mask`` is true, as a short string, or ``"none"``.

    Truncated past ``limit`` because the point is to be readable in a summary
    line; ``meta["diagnostics"]`` keeps every row. ``"none"`` rather than an
    empty string so a clean run reads as an answer instead of a missing field.
    """
    flagged = index[np.asarray(mask, dtype=bool)]
    if len(flagged) == 0:
        return "none"
    shown = ", ".join(_iso(d) for d in flagged[:limit])
    return shown if len(flagged) <= limit else f"{shown}, +{len(flagged) - limit} more"


def _run_health(failures: int, repaired: int, ill_conditioned: int) -> str:
    """One phrase saying whether anything went wrong, for ``meta["run_health"]``.

    The three conditions are kept apart rather than collapsed into a count of
    "problems", because they are different events with different responses: a
    solver failure means the weights at that date are a best iterate rather than
    an ERC portfolio, a repaired covariance means the *estimator* produced
    something that was not a covariance, and an ill-conditioned one means the
    matrix was invertible but barely. A run can be clean on two and alarming on
    the third.

    Cap saturation is deliberately **not** counted here. An overlay pinned to its
    ceiling is not a failure -- nothing broke, the run is exactly what the policy
    asked for -- but it does mean the vol target is not what produced the
    returns. That belongs on the overlay line, whose marker says "read this",
    rather than in a field that says "something went wrong".
    """
    problems = []
    if failures:
        problems.append(f"{failures} solver failure{'s' if failures != 1 else ''}")
    if repaired:
        problems.append(f"{repaired} covariance{'s' if repaired != 1 else ''} repaired")
    if ill_conditioned:
        problems.append(f"{ill_conditioned} ill-conditioned rebalance"
                        f"{'s' if ill_conditioned != 1 else ''}")
    return "clean" if not problems else "; ".join(problems)


def format_run_summary(result: BacktestResult) -> str:
    """The engine's own account of a run, as printable text (TASKS.md step 29).

    Reports what the *machinery* did -- solver, covariance, overlay, financing --
    and lines that describe something going wrong are marked with a leading
    ``!``, so a bad run is visible at a glance rather than being one number among
    twenty. That marker is the step's "done when" in a single character.

    **No performance metrics here, deliberately.** Sharpe, CAGR and drawdown live
    in ``src.metrics``, which sits *downstream* of this module in the pipeline
    (see ``src/__init__.py``); importing it here would invert the layering for
    the same reason ``risk_contributions`` is composed from ``riskparity``'s
    primitives rather than imported from ``benchmarks``. ``run_backtest.py``
    prints both blocks, which is where they belong together: this one says
    whether the numbers can be trusted, that one says what they were.

    Written to survive being handed a result it did not produce. A benchmark
    carries almost none of these keys, and ``target_vol`` it carries as an
    explicit ``None`` -- an unlevered strategy has no target, which is a
    different statement from "the field is missing". So each section is anchored
    on a field it cannot render without, optional clauses are added only when
    present, and a section with nothing to say is dropped rather than printed
    empty or printed as ``nan``.
    """
    meta = result.meta
    # (is a warning, label, headline, detail lines). Details are kept apart from
    # the headline rather than embedded with "\n" so that wrapping and indenting
    # happen once, at render time, instead of being hand-aligned at each append.
    rows: list[tuple[bool, str, str, list[str]]] = []

    if "n_rebalances" in meta:
        rows.append((
            False, "sample",
            f"{meta['n_rebalances']} rebalances "
            f"{meta.get('first_rebalance', '?')} -> {meta.get('last_rebalance', '?')}, "
            f"{meta.get('n_return_days', len(result.returns))} return days",
            [],
        ))
    target_vol = _as_float(meta.get("target_vol"))
    cap = meta.get("leverage_cap")
    if target_vol is not None and cap is not None:
        low, high = (float(x) for x in cap)
        text = f"target vol {target_vol:.1%}, leverage cap {low:.2f}-{high:.2f}x"
        cov_hl, vol_hl = _as_float(meta.get("cov_halflife")), _as_float(meta.get("vol_halflife"))
        if cov_hl is not None and vol_hl is not None:
            text += f", halflives {cov_hl:.0f}d cov / {vol_hl:.0f}d vol"
        if "warmup_trading_days" in meta:
            text += f", warmup {meta['warmup_trading_days']}d"
        rows.append((False, "policy", text, []))

    if "solver_failures" in meta:
        failures = int(meta["solver_failures"])
        n_rebalances = int(meta.get("n_rebalances", len(result.weights)))
        parts = [f"{failures} failure{'s' if failures != 1 else ''} in {n_rebalances}"]
        if "total_restarts" in meta:
            parts.append(f"{meta['total_restarts']} restarts")
        worst_rc = _as_float(meta.get("worst_rc_deviation"))
        if worst_rc is not None:
            parts.append(
                f"worst RC deviation {worst_rc:.2e} "
                f"({meta.get('worst_rc_deviation_date', '?')})"
            )
        details = []
        if failures:
            # Named, and named as what it is. The best iterate is still a
            # portfolio and the run continued on purpose -- see the module
            # docstring on why a numerical failure is not an economic one -- but
            # "the weights at these dates are not ERC" has to be sayable.
            details.append(
                f"at {meta.get('solver_failure_dates', '?')} -- best iterate used, "
                f"so those weights are not equal-risk"
            )
        rows.append((bool(failures), "solver", ", ".join(parts), details))

    max_condition = _as_float(meta.get("max_condition_number"))
    if max_condition is not None:
        repaired = int(meta.get("covariances_repaired", 0))
        ill = int(meta.get("ill_conditioned_rebalances", 0))
        median_condition = _as_float(meta.get("median_condition_number"))
        text = f"{repaired} repaired, {ill} ill-conditioned, condition number "
        text += "" if median_condition is None else f"median {median_condition:,.1f} / "
        text += f"max {max_condition:,.1f} ({meta.get('worst_condition_date', '?')})"
        details = []
        if repaired:
            details.append(
                f"repaired at {meta.get('covariance_repair_dates', '?')} -- the "
                f"estimator returned a matrix that was not a covariance"
            )
        if ill:
            details.append(
                f"ill-conditioned at {meta.get('ill_conditioned_dates', '?')} -- "
                f"solvable, but the optimum there is unstable"
            )
        rows.append((bool(repaired or ill), "covariance", text, details))

    mean_leverage = _as_float(meta.get("mean_leverage"))
    if mean_leverage is not None:
        n = int(meta.get("n_rebalances", len(result.weights))) or 1
        ceiling = int(meta.get("cap_binding_ceiling", 0))
        floor = int(meta.get("cap_binding_floor", 0))
        low_lev, high_lev = _as_float(meta.get("min_leverage")), _as_float(meta.get("max_leverage"))
        text = f"leverage mean {mean_leverage:.2f}x"
        if low_lev is not None and high_lev is not None:
            text += f", range {low_lev:.2f}-{high_lev:.2f}x"
        text += (
            f"; cap bound {ceiling + floor}/{n} "
            f"({(ceiling + floor) / n:.1%}: ceiling {ceiling}, floor {floor})"
        )
        details = []
        saturated = (ceiling / n) > SATURATION_REPORT_FRACTION
        if saturated:
            details.append(
                f"the ceiling bound on {ceiling / n:.0%} of rebalances -- at the cap the "
                f"overlay holds a constant notional rather than targeting volatility, so "
                f"the target is not what produced these returns"
            )
        trailing = _as_float(meta.get("mean_trailing_vol"))
        if trailing is not None and target_vol is not None:
            details.append(
                f"mean trailing vol estimate {trailing:.2%} against a "
                f"{target_vol:.1%} target"
            )
        rows.append((bool(saturated), "overlay", text, details))

    if "financed" in meta:
        if meta["financed"]:
            text = (
                f"charged at the bill rate, drag "
                f"{meta.get('financing_drag_annual', float('nan')):+.2%}/yr"
            )
            warn = False
        else:
            # Not a defect, but never a headline number: PLAN.md's warning about
            # how this strategy gets oversold is precisely this run.
            text = ("NOT charged -- leverage is free in this run, which overstates "
                    "the strategy and is not a headline number")
            warn = True
        rows.append((warn, "financing", text, []))

    if "run_health" in meta:
        rows.append((meta["run_health"] != "clean", "health", str(meta["run_health"]), []))

    # Stated in the output rather than only in a docstring, because this is the
    # block someone screenshots. A real book pays more than DTB3 to borrow and
    # pays to trade, so these results are optimistic even after financing.
    rows.append((False, "not modelled",
                 "transaction costs, bid-ask, borrow spread over the bill rate", []))

    # Continuation lines are indented past the "! label : " column, so a detail
    # is visibly subordinate to the line it explains and a reader scanning the
    # left edge for "!" is not interrupted by prose.
    indent = " " * (2 + _SUMMARY_LABEL_WIDTH + 2)
    lines = [f"{result.strategy} -- run summary"]
    for is_warning, label, headline, details in rows:
        lines.append(f"{'!' if is_warning else ' '} {label:<{_SUMMARY_LABEL_WIDTH}s}: {headline}")
        for detail in details:
            lines.extend(
                textwrap.fill(
                    detail,
                    width=SUMMARY_LINE_WIDTH,
                    initial_indent=indent,
                    subsequent_indent=indent,
                ).splitlines()
            )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys
    from dataclasses import replace

    from .benchmarks import equal_weight, risk_contribution_fractions

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

    # =====================================================================
    # Step 29 -- diagnostics and the run summary
    # =====================================================================
    print("\n" + "=" * 72)
    print(format_run_summary(result))
    print("=" * 72)

    # --- step 29's "done when": a FAILED rebalance is visible ----------------
    # A clean run cannot demonstrate this, and none of the 234 real rebalances
    # fails, so the failure is injected: solve_erc_with_status is wrapped to
    # report converged=False on one chosen date. That is the same technique
    # tests/test_lookahead.py uses on the look-ahead guard, and for the same
    # reason -- the reporting path is what is being tested, so it has to be
    # exercised with something to report. Injecting at the seam also leaves the
    # solver itself untouched: the weights are the real ones, only the status
    # differs, which is exactly the "numerical failure, best iterate used" case.
    print("\nstep 29: an injected solver failure must reach the output")
    real_solver = solve_erc_with_status
    failure_date = result.weights.index[7]

    def flaky_solver(cov, *args, **kwargs):
        solution = real_solver(cov, *args, **kwargs)
        if flaky_solver.calls == 7:
            solution = replace(solution, converged=False, max_rc_deviation=3.4e-3,
                               message="injected failure (step 29 self-check)")
        flaky_solver.calls += 1
        return solution

    flaky_solver.calls = 0
    globals()["solve_erc_with_status"] = flaky_solver
    try:
        failed = run_backtest(synthetic, risk_free=cash)
    finally:
        globals()["solve_erc_with_status"] = real_solver

    summary = format_run_summary(failed)
    lines = summary.splitlines()
    # A continuation line is indented past the label column; a row line starts
    # with the marker at column 0. That is what separates "detail about the line
    # above" from "the next thing the summary has to say".
    _CONTINUATION = " " * (2 + _SUMMARY_LABEL_WIDTH + 2)

    def block_at(prefix: str, source: list[str]) -> str:
        """A summary line plus every continuation line under it.

        Taken to the next unindented line rather than a fixed number of lines,
        because details are wrapped to SUMMARY_LINE_WIDTH and a longer date list
        would spill onto a third. Asserting against the whole summary instead
        would be satisfied by the date appearing in the worst-RC-deviation
        clause, which names the same date for an unrelated reason -- so the
        check has to look at the block that claims a *failure*.
        """
        start = next(i for i, line in enumerate(source) if line.startswith(prefix))
        end = start + 1
        while end < len(source) and source[end].startswith(_CONTINUATION):
            end += 1
        return "\n".join(source[start:end])

    solver_block = block_at("! solver", lines)

    visible = {
        "meta['solver_failures'] counts it": failed.meta["solver_failures"] == 1,
        "meta['solver_failure_dates'] names the date":
            failed.meta["solver_failure_dates"] == str(failure_date.date()),
        "meta['run_health'] stops saying 'clean'":
            failed.meta["run_health"] == "1 solver failure",
        "the summary marks the solver line with '!'": solver_block.startswith("! solver"),
        "the solver block names the failing date": str(failure_date.date()) in solver_block,
        "and says the weights there are not equal-risk":
            "best iterate used" in solver_block and "not equal-risk" in solver_block,
        "the health line is marked too": any(line.startswith("! health") for line in lines),
        "the run still produced a valid result": failed.returns.shape == result.returns.shape,
        "and the clean run still reads clean": result.meta["run_health"] == "clean",
    }
    for label, ok in visible.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")

    print("\n  what a reader sees (marked lines and their detail):")
    show = False
    for line in lines:
        if not line.startswith(_CONTINUATION):
            show = line.startswith("!")
        if show:
            print(f"    {line}")

    if not all(visible.values()):
        print("  REGRESSION: a failed rebalance is not visible in the output")
        sys.exit(1)

    # --- the two failure modes stay distinguishable -------------------------
    # A solver that did not converge is a numerical event: use the best iterate,
    # count it, keep going. ERCSolverError is an economic one -- no equal-risk
    # portfolio exists above the weight floor, which means a price series is
    # broken -- and it must kill the run with the date attached. Collapsing them
    # would hide whichever one mattered, so the distinction is checked rather
    # than trusted.
    def broken_solver(cov, *args, **kwargs):
        if broken_solver.calls == 7:
            raise ERCSolverError("injected infeasibility (step 29 self-check)")
        broken_solver.calls += 1
        return real_solver(cov, *args, **kwargs)

    broken_solver.calls = 0
    globals()["solve_erc_with_status"] = broken_solver
    try:
        run_backtest(synthetic, risk_free=cash)
        print("\n  REGRESSION: an infeasible rebalance did not stop the run")
        sys.exit(1)
    except ERCSolverError as exc:
        dated = str(failure_date.date()) in str(exc)
        print(f"\n  infeasible rebalance kills the run: ERCSolverError, "
              f"date attached = {dated}")
        if not dated:
            print("  REGRESSION: the failing date is not in the message")
            sys.exit(1)
    finally:
        globals()["solve_erc_with_status"] = real_solver

    # --- a saturated overlay is called out, and it is NOT a failure ---------
    # Reachable without injecting anything: a tight ceiling is what
    # `--max-leverage` does. The run is clean -- nothing went wrong -- but the
    # target vol is no longer what produced the returns, and the summary has to
    # say so or a flat leverage series just looks like a calm market.
    print("\nstep 29: a saturated overlay is visible without being called a failure")
    capped = run_backtest(synthetic, risk_free=cash, leverage_cap=(0.5, 1.2))
    capped_summary = format_run_summary(capped)
    overlay_marked = any(line.startswith("! overlay") for line in capped_summary.splitlines())
    print(f"  ok   overlay line marked         : {overlay_marked}")
    print(f"  ok   ceiling bound on            : {capped.meta['cap_binding_fraction']:.0%} "
          f"of rebalances")
    print(f"  ok   health stays 'clean'        : {capped.meta['run_health'] == 'clean'} "
          f"(saturation is policy working, not machinery failing)")
    if not overlay_marked or capped.meta["run_health"] != "clean":
        print("  REGRESSION: saturation is either invisible or miscounted as a failure")
        sys.exit(1)
    print("\n  what a reader sees:")
    for line in block_at("! overlay", capped_summary.splitlines()).splitlines():
        print(f"    {line}")

    # --- the covariance branches render, on a doctored meta -----------------
    # Neither a repair nor an ill-conditioned window can be provoked on plausible
    # four-sleeve data -- shrinkage maps rho -> 0.9*rho, which caps the shrunk
    # correlation matrix's condition number at 37 and puts MAX_CONDITION_NUMBER
    # out of reach short of a 1,600:1 volatility ratio (see the note in
    # _build_meta). The reporting path is still what step 29 is accountable for,
    # so it is exercised directly on a doctored meta rather than left unrun until
    # the day a real repair happens.
    doctored = replace(result, meta={
        **result.meta,
        "covariances_repaired": 2,
        "covariance_repair_dates": "2008-09-30, 2008-10-31",
        "ill_conditioned_rebalances": 1,
        "ill_conditioned_dates": "2008-10-31",
        "run_health": _run_health(0, 2, 1),
    })
    doctored_summary = format_run_summary(doctored)
    cov_marked = any(line.startswith("! covariance") for line in doctored_summary.splitlines())
    named = "2008-09-30" in doctored_summary and "not a covariance" in doctored_summary
    print("\nstep 29: the covariance branches render (doctored meta, not a real repair)")
    print(f"  ok   covariance line marked      : {cov_marked}")
    print(f"  ok   dates and reason named      : {named}")
    print(f"  ok   health phrase               : {doctored.meta['run_health']!r}")
    if not (cov_marked and named):
        print("  REGRESSION: a repaired covariance would not be visible")
        sys.exit(1)

    # --- the summary survives a result it did not produce -------------------
    # Benchmarks carry none of these keys. format_run_summary is read with
    # defaults throughout so that a caller can hand it any BacktestResult, which
    # is what run_backtest.py will do the first time someone loops over results.
    bench = equal_weight(synthetic)
    bench_summary = format_run_summary(bench)
    print(f"\nformat_run_summary on a benchmark ({bench.strategy}): "
          f"{len(bench_summary.splitlines())} lines, no KeyError")
    print("  " + "\n  ".join(bench_summary.splitlines()))

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
