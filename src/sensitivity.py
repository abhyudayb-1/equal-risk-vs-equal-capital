"""How much of ERC's edge survives the covariance being *estimated*.

``src/comparison.py`` answers the project's question at one setting of the
estimator. This answers the standard objection to that answer: **1/N needs no
estimate at all, and ERC needs four of them.** Equal capital cannot be wrong
about a covariance because it never looks at one. Equal risk reads a halflife, a
shrinkage intensity, a warm-up length and a rebalance frequency, and if its
advantage lives only at the settings in ``config.py`` then the advantage belongs
to the settings rather than to the rule.

Two separate things are measured here, and they fail in opposite directions:

1. **The estimator sweep.** Each knob is moved across a grid that brackets its
   default, with ERC and 1/N re-run *together* at every setting -- both rules,
   one window, one calendar -- and the row reports ERC's return-to-volatility and
   effective bets against 1/N's. A setting where ERC stops winning is reported in
   the output rather than left in the table for someone to notice.

2. **Trading cost.** ERC re-optimises on every rebalance and 1/N does not, so
   "no transaction costs are modelled" is honest everywhere in this repo and is
   also the obvious attack on it. The cost does not fall evenly, and the thing to
   report is not a cost estimate but a **breakeven**: the cost per unit traded at
   which ERC's advantage over 1/N disappears. That number is a property of the
   two rules' turnover and does not require anyone to agree about spreads.

**The turnover convention, which is the one worth reading before trusting a number
in here.** Every strategy's weights frame holds its *targets*, and every return
path -- ``backtest.run_backtest`` and ``benchmarks.fixed_weight`` alike -- accrues
through ``overlay.apply_overlay``, which trades to those targets at a month-end
and then **holds** the book until the next one. So 1/N's weights frame is
constant, but its book is not: it drifts off 25% during the month and has to be
traded back. Turnover therefore cannot be read off the weights frame (that would
report exactly 0.0% for 1/N and say equal capital is free to run); it is
reconstructed from the same drift the returns were earned on. At each rebalance
the previous target is carried forward by the sleeves' realised returns, and the
turnover is the distance from *that* to the new target::

    w_drifted = w_prev * (1 + R) / sum(w_prev * (1 + R))
    traded    = 0.5 * sum(|w_new - w_drifted|)

which is the fraction of the book that changes hands -- the one-way convention,
so a cost per unit traded multiplies it directly. It is reported in two parts,
because the split is the whole argument: the **drift-back** component both rules
pay, and the **re-optimisation** component that is ERC's alone.

A levered book is measured the same way on its holdings per unit of equity,
read from ``overlay.pre_trade_holdings`` -- the very simulation that produced its
returns -- so the trade that re-gears a leverage which drifted during the month
is charged too.

An earlier version held the targets constant every day in the return path, which
is a book traded back to target *daily*, while charging costs on a book that
drifted for a month. The return path and the cost model described two different
strategies; they now describe one.

Run a self-check from the repo root (offline, fixtures only, writes nothing)::

    .venv/bin/python -m src.sensitivity
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .benchmarks import equal_weight
from .comparison import (
    EQUAL_CAPITAL,
    ERC_UNLEVERED,
    ComparisonError,
    effective_bets,
    matched_target_vol,
)
from .config import (
    COST_RATES_BPS,
    COV_EWMA_HALFLIFE_DAYS,
    COV_SHRINKAGE,
    COV_WARMUP_TRADING_DAYS,
    LEVERAGE_CAP,
    MAX_BREAKEVEN_COST,
    SENSITIVITY_COV_HALFLIVES,
    SENSITIVITY_REBALANCE_MONTHS,
    SENSITIVITY_SHRINKAGES,
    SENSITIVITY_WARMUPS,
    SUMMARY_LINE_WIDTH,
    TRADING_DAYS_PER_YEAR,
)
from .covariance import estimate_covariance
from .metrics import summarize
from .overlay import pre_trade_holdings, unlevered_returns

log = logging.getLogger(__name__)

# The knobs, in the order the report walks them, with the config grid each one
# sweeps and the default it brackets. Keeping them in one table means the sweep,
# the printed report and the "which settings lost" summary cannot drift apart.
KNOBS: tuple[tuple[str, str, tuple, object], ...] = (
    ("cov_halflife", "covariance halflife (days)",
     SENSITIVITY_COV_HALFLIVES, COV_EWMA_HALFLIFE_DAYS),
    ("shrinkage", "shrinkage intensity",
     SENSITIVITY_SHRINKAGES, COV_SHRINKAGE),
    ("warmup_trading_days", "warm-up (trading days)",
     SENSITIVITY_WARMUPS, COV_WARMUP_TRADING_DAYS),
    ("rebalance_months", "rebalance every N months",
     SENSITIVITY_REBALANCE_MONTHS, 1),
)

BASIS_POINT = 1e-4


class SensitivityError(ValueError):
    """The sweep cannot be run, or a setting produced an incomparable result."""


# ---------------------------------------------------------------------------
# Turnover -- reconstructed, because the weights frame does not show it
# ---------------------------------------------------------------------------
def drifted_weights(
    weights: pd.DataFrame,
    sleeve_returns: pd.DataFrame,
) -> pd.DataFrame:
    """What the previous target had drifted to by each rebalance date.

    Row ``t`` is ``weights`` row ``t-1`` carried forward by each sleeve's
    compounded return over ``(t-1, t]`` and renormalised to sum to 1 -- the book
    a manager would actually be holding when they sat down to rebalance. The
    first row is all-NaN: there is no previous book, so there is no drift and
    no trade that is comparable to the others (buying the initial allocation is
    a one-off, not a rebalance).

    The returns are taken *strictly after* the previous rebalance date and up to
    and including this one, the same convention the engine accrues on: a book
    chosen at the close of day ``d`` earns from ``d+1``.
    """
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise SensitivityError("weights must be a non-empty frame of rebalance rows")
    missing = [t for t in weights.columns if t not in sleeve_returns.columns]
    if missing:
        raise SensitivityError(
            f"sleeve_returns has no column for {missing}; cannot drift the book"
        )

    sleeves = sleeve_returns[list(weights.columns)]
    rows = [np.full(weights.shape[1], np.nan)]
    for previous, current in zip(weights.index[:-1], weights.index[1:]):
        window = sleeves.loc[(sleeves.index > previous) & (sleeves.index <= current)]
        growth = (1.0 + window).prod().to_numpy()
        drifted = weights.loc[previous].to_numpy() * growth
        total = drifted.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise SensitivityError(
                f"the book drifted to a non-positive value by {current.date()}; "
                "a sleeve's return series is broken"
            )
        rows.append(drifted / total)
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)


def turnover(
    weights: pd.DataFrame,
    sleeve_returns: pd.DataFrame,
    leverage: pd.Series | None = None,
    risk_free: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """Per-rebalance turnover, split into the part each rule cannot avoid and ERC's own.

    Without ``leverage`` this is the unlevered allocation's turnover, from
    :func:`drifted_weights`. With it, the book is the levered one the overlay
    holds -- ``leverage[t] * weights[t]`` of equity in the sleeves -- and every
    column is measured on those holdings per unit of equity, drifted by the
    sleeves *and* by the loan accruing at ``risk_free`` (which must then be the
    same rate the returns were financed at). The trade that brings a drifted
    leverage back to its new decision is part of ``total``; it is not scaled on
    afterwards.

    Three columns, all as a fraction of the book, one-way (``0.5 * sum|dw|``), so
    a cost per unit traded multiplies them directly:

    ``drift_back``
        from the drifted book back to the *previous* target. What any rule pays
        simply for having held a portfolio through a month of returns. For 1/N
        this is the whole bill.
    ``reoptimisation``
        from the previous target to the new one. Zero for every fixed-weight
        rule by construction, and for ERC it is what the solver costs.
    ``total``
        from the drifted book to the new target -- the actual trade. It is **not**
        the sum of the other two: drift and re-optimisation can point the same
        way in a sleeve and partly cancel, so the real trade is at most their
        sum and usually less. That near-cancellation is a real effect worth not
        losing to a tidier definition -- ERC drifts *toward* its next solution
        about as often as away from it, because both move with volatility.

    The first rebalance is NaN in every column; see :func:`drifted_weights`.
    """
    if leverage is None:
        drifted = drifted_weights(weights, sleeve_returns)
    else:
        if not leverage.index.equals(weights.index):
            raise SensitivityError(
                "leverage must be decided on the weights' rebalance dates; a "
                "different calendar is a different book"
            )
        held = sleeve_returns.loc[
            sleeve_returns.index > weights.index[0], list(weights.columns)
        ]
        drifted = pre_trade_holdings(held, weights, leverage, risk_free)
        weights = weights.mul(leverage.astype("float64"), axis=0)
    previous = weights.shift(1)
    half_abs = lambda frame: 0.5 * frame.abs().sum(axis=1, skipna=False)  # noqa: E731
    return pd.DataFrame(
        {
            "drift_back": half_abs(previous - drifted),
            "reoptimisation": half_abs(weights - previous),
            "total": half_abs(weights - drifted),
        }
    )


def charge_costs(
    returns: pd.Series,
    traded: pd.Series,
    cost_rate: float,
) -> pd.Series:
    """``returns`` with ``cost_rate`` charged on each rebalance's traded fraction.

    The charge lands on the **first return day at or after** the rebalance date
    -- for a rebalance on a trading day, that day itself. The trade is made at
    that day's close, so its cost comes out of that day's closing equity; the
    new book it buys starts earning the *next* day, under the engine's one-day
    rule. (The cost is paid when the trade is made, not when the new book first
    earns.) A rebalance after the last return day is dropped rather than charged
    against nothing.

    ``traded`` is per unit of equity. For a levered book pass
    ``turnover(..., leverage=...)``, which already measures the trade on the
    levered holdings -- including re-gearing a leverage that drifted -- so
    nothing is scaled here.
    """
    if cost_rate < 0:
        raise SensitivityError(f"cost_rate must be non-negative, got {cost_rate}")
    if cost_rate == 0:
        return returns.copy()

    charge = traded.astype("float64").fillna(0.0) * float(cost_rate)

    index = returns.index
    positions = index.searchsorted(charge.index, side="left")
    out = returns.copy()
    for position, amount in zip(positions, charge.to_numpy()):
        if position >= len(index) or amount == 0.0:
            continue
        out.iloc[position] -= amount
    return out


def turnover_table(
    rule_results: dict,
    sleeve_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """One row per rule: what it trades per rebalance and per year.

    ``annual_turnover`` is the total traded over the sample divided by the
    sample's length in years, so a quarterly rule and a monthly one are directly
    comparable -- per-rebalance turnover alone is not, since the quarterly rule
    has a third as many of them and a bigger one each time.

    Every figure is of the **unlevered allocation**. Leverage is deliberately
    not folded in here: a rule's weights are one thing and the size the overlay
    chooses to hold them at is another, and a table that multiplied them would
    report a levered ERC book as trading more than ERC *as an allocation* does.
    ``turnover(..., leverage=...)`` is where a levered book's trades are
    measured, and the caller asks for it explicitly.
    """
    rows = []
    for name, result in rule_results.items():
        traded = turnover(result.weights, sleeve_returns)
        years = len(result.returns) / periods_per_year
        rows.append(
            {
                "name": name,
                "rebalances": int(len(result.weights)),
                "drift_back_mean": float(traded["drift_back"].mean()),
                "reoptimisation_mean": float(traded["reoptimisation"].mean()),
                "turnover_mean": float(traded["total"].mean()),
                "turnover_max": float(traded["total"].max()),
                "annual_turnover": float(traded["total"].sum() / years) if years else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("name").rename_axis(None)


# ---------------------------------------------------------------------------
# Cost sensitivity and the breakeven
# ---------------------------------------------------------------------------
def _ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Return-to-volatility at a zero risk-free rate -- this project's ratio."""
    return float(summarize(returns, periods_per_year)["sharpe_ratio"])


@dataclass(frozen=True)
class CostComparison:
    """ERC against 1/N as a cost per unit traded is dialled up.

    ``table`` is one row per cost rate. ``breakeven_cost`` is the rate at which
    ERC's ratio falls to 1/N's -- ``nan`` if ERC never led at zero cost, and
    ``inf`` if it still leads beyond ``config.MAX_BREAKEVEN_COST``, because an
    extrapolated breakeven is not a measurement.
    """

    table: pd.DataFrame
    breakeven_cost: float
    erc_label: str
    benchmark_label: str
    levered: bool


def cost_comparison(
    erc_returns: pd.Series,
    erc_traded: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_traded: pd.Series,
    cost_rates_bps: tuple[float, ...] = COST_RATES_BPS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    levered: bool = False,
    erc_label: str = ERC_UNLEVERED,
    benchmark_label: str = EQUAL_CAPITAL,
) -> CostComparison:
    """Both rules' ratios at a grid of costs, plus the breakeven between them.

    Both rules are charged the **same** rate. That is the point: the question is
    not what trading costs but whether ERC's advantage is large enough to pay for
    the extra trading it does, and that is answered by one number -- the rate at
    which the two ratios cross --     which nobody has to agree about spreads to read.

    ``levered`` records that ``erc_traded`` is a levered book's turnover, so the
    report prints the caveat that goes with it. It changes no arithmetic.
    """
    if not len(cost_rates_bps):
        raise SensitivityError("cost_rates_bps is empty; there is nothing to sweep")

    def ratios(rate: float) -> tuple[float, float]:
        return (
            _ratio(charge_costs(erc_returns, erc_traded, rate), periods_per_year),
            _ratio(charge_costs(benchmark_returns, benchmark_traded, rate), periods_per_year),
        )

    rows = []
    for bps in cost_rates_bps:
        erc_ratio, benchmark_ratio = ratios(bps * BASIS_POINT)
        rows.append(
            {
                "cost_bps": float(bps),
                erc_label: erc_ratio,
                benchmark_label: benchmark_ratio,
                "gap": erc_ratio - benchmark_ratio,
            }
        )
    table = pd.DataFrame(rows).set_index("cost_bps")

    # Bisection on a difference that is smooth and decreasing in the cost (ERC
    # trades more, so every basis point costs it more). Guarded at both ends
    # rather than assumed: if ERC does not lead at zero there is no breakeven to
    # find, and if it still leads at the cap the honest answer is "past here".
    low_gap = table["gap"].iloc[0] if table.index[0] == 0.0 else (ratios(0.0)[0] - ratios(0.0)[1])
    if low_gap <= 0:
        breakeven = float("nan")
    else:
        high = MAX_BREAKEVEN_COST
        erc_high, benchmark_high = ratios(high)
        if erc_high - benchmark_high > 0:
            breakeven = float("inf")
        else:
            low = 0.0
            for _ in range(60):
                middle = 0.5 * (low + high)
                erc_mid, benchmark_mid = ratios(middle)
                if erc_mid - benchmark_mid > 0:
                    low = middle
                else:
                    high = middle
            breakeven = 0.5 * (low + high)

    return CostComparison(
        table=table,
        breakeven_cost=breakeven,
        erc_label=erc_label,
        benchmark_label=benchmark_label,
        levered=bool(levered),
    )


# ---------------------------------------------------------------------------
# The estimator sweep
# ---------------------------------------------------------------------------
def run_setting(
    sleeve_returns: pd.DataFrame,
    risk_free: pd.Series,
    cov_halflife: float = COV_EWMA_HALFLIFE_DAYS,
    shrinkage: float = COV_SHRINKAGE,
    warmup_trading_days: int = COV_WARMUP_TRADING_DAYS,
    rebalance_months: int = 1,
    leverage_cap: tuple[float, float] = LEVERAGE_CAP,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """ERC and 1/N at one estimator setting, measured against each other.

    Three engine-grade runs: ERC (whose unlevered series is the allocation claim
    on its own), 1/N on the **same** covariance settings, warm-up and calendar,
    and ERC levered to 1/N's realised volatility -- the matched column, rebuilt
    here rather than approximated, because the headline conclusion is about that
    row and a sweep that only moved the unlevered one would not test it.

    Both rules take every setting together. A sweep that moved the engine's
    warm-up and left 1/N's alone would compare two rules over two windows.
    """
    erc = run_backtest(
        sleeve_returns,
        risk_free=risk_free,
        cov_halflife=cov_halflife,
        shrinkage=shrinkage,
        warmup_trading_days=warmup_trading_days,
        rebalance_months=rebalance_months,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
    )
    benchmark = equal_weight(
        sleeve_returns,
        cov_estimator=partial(
            estimate_covariance,
            halflife=cov_halflife,
            shrinkage=shrinkage,
            periods_per_year=periods_per_year,
        ),
        strategy=EQUAL_CAPITAL,
        warmup_trading_days=warmup_trading_days,
        rebalance_months=rebalance_months,
    )
    index = erc.returns.index
    if not benchmark.returns.index.equals(index):
        raise ComparisonError(
            "ERC and 1/N came back on different calendars at this setting "
            f"(cov_halflife={cov_halflife}, shrinkage={shrinkage}, "
            f"warmup={warmup_trading_days}, rebalance_months={rebalance_months}); "
            "a comparison across different windows is not a fair one"
        )

    erc_unlevered = unlevered_returns(sleeve_returns.loc[index], erc.weights)
    match_target = matched_target_vol(benchmark.returns, periods_per_year)
    matched = run_backtest(
        sleeve_returns,
        risk_free=risk_free,
        target_vol=match_target,
        cov_halflife=cov_halflife,
        shrinkage=shrinkage,
        warmup_trading_days=warmup_trading_days,
        rebalance_months=rebalance_months,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
    )

    erc_ratio = _ratio(erc_unlevered, periods_per_year)
    benchmark_ratio = _ratio(benchmark.returns, periods_per_year)
    matched_ratio = _ratio(matched.returns, periods_per_year)
    return {
        "erc_ratio": erc_ratio,
        "one_n_ratio": benchmark_ratio,
        "gap": erc_ratio - benchmark_ratio,
        "matched_ratio": matched_ratio,
        "matched_gap": matched_ratio - benchmark_ratio,
        "matched_realized_vol": float(
            summarize(matched.returns, periods_per_year)["annual_volatility"]
        ),
        "erc_bets": float(effective_bets(erc.risk_contributions).mean()),
        "one_n_bets": float(effective_bets(benchmark.risk_contributions).mean()),
        "rebalances": int(len(erc.weights)),
        "days": int(len(index)),
        "solver_failures": int(erc.meta.get("solver_failures", 0)),
    }


def sweep(
    sleeve_returns: pd.DataFrame,
    risk_free: pd.Series,
    knobs: tuple = KNOBS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Every knob across its grid, one row per setting, with the defaults marked.

    Each row moves exactly **one** knob and leaves the rest at their ``config.py``
    values. A full grid would be more thorough and unreadable; one-at-a-time says
    which knob the conclusion is sensitive to, which is the question.

    The default setting appears once per knob and is recomputed rather than
    copied, which is also a check: four rows that are supposed to be the same run
    and are not would mean a knob is leaking.
    """
    rows = []
    for name, label, grid, default in knobs:
        for value in grid:
            log.info("Sweeping %s = %s ...", name, value)
            measured = run_setting(
                sleeve_returns, risk_free, periods_per_year=periods_per_year, **{name: value}
            )
            rows.append(
                {
                    "knob": label,
                    "setting": value,
                    "is_default": bool(value == default),
                    **measured,
                }
            )
    return pd.DataFrame(rows)


def losing_settings(swept: pd.DataFrame, column: str = "gap") -> pd.DataFrame:
    """The rows where ERC did **not** beat 1/N on ``column``.

    Named and returned rather than left for a reader to find in a seventeen-row
    table: the whole reason this step exists is that the answer might be
    uncomfortable, and a sweep that buries its own counterexamples is decoration.

    ``column`` is ``"gap"`` (unlevered) or ``"matched_gap"`` (levered to 1/N's
    volatility with financing charged), and the two must be read apart rather
    than OR-ed together. The matched column being behind is the project's
    *headline*, true at the default setting before any sweep started; folding it
    into one "ERC lost here" list would report the known conclusion as if it
    were a fragility of the estimator and hide the thing the sweep actually
    found.
    """
    if column not in swept.columns:
        raise SensitivityError(f"{column!r} is not a column of the sweep: {list(swept.columns)}")
    return swept[swept[column] <= 0]


# ---------------------------------------------------------------------------
# The printed forms
# ---------------------------------------------------------------------------
_COST_CAVEATS: tuple[str, ...] = (
    "Turnover is RECONSTRUCTED, not read off the weights. Each rule's weights "
    "frame holds its targets, so 1/N's shows 0.0% turnover; but both books are "
    "held between month-ends and drift with returns, in the return path as well "
    "as here. Each rebalance's trade is measured from the previous target drifted "
    "by realised returns to the new target -- the same drift the returns were "
    "earned on.",
    "One-way, as a fraction of the book, so a cost per unit traded multiplies it "
    "directly. 'total' is not drift-back plus re-optimisation: the two partly "
    "cancel, because ERC drifts toward its next solution about as often as away "
    "from it.",
    "The levered rows' trades are measured on the held book's holdings per unit "
    "of equity, so they include re-gearing a leverage that drifted during the "
    "month as well as moving the weights.",
    "The breakeven is the number to quote. It is a property of the two rules' "
    "turnover and their ratios, and reading it needs no agreement about what a "
    "spread costs.",
)


def format_turnover_table(table: pd.DataFrame) -> str:
    """:func:`turnover_table`'s frame as percentages, one labelled line per rule."""
    rows = {}
    for name in table.index:
        row = table.loc[name]
        rows[name] = {
            "rebalances": f"{int(row['rebalances'])}",
            "traded per rebalance": f"{row['turnover_mean']:.2%}",
            "   of which drift-back": f"{row['drift_back_mean']:.2%}",
            "   of which re-optimising": f"{row['reoptimisation_mean']:.2%}",
            "worst single rebalance": f"{row['turnover_max']:.2%}",
            "traded per year": f"{row['annual_turnover']:.1%}",
        }
    return pd.DataFrame(rows).to_string()


def format_cost_table(comparison: CostComparison) -> str:
    """:class:`CostComparison`'s grid, plus the breakeven in words."""
    table = comparison.table.copy()
    table.index = [f"{int(bps)}bp" for bps in table.index]
    text = table.to_string(float_format=lambda v: f"{v:>8.3f}")

    if np.isnan(comparison.breakeven_cost):
        verdict = (
            f"! {comparison.erc_label} does not beat {comparison.benchmark_label} even "
            "at zero cost, so there is no breakeven to find -- cost is not what "
            "decides this row."
        )
    elif np.isinf(comparison.breakeven_cost):
        verdict = (
            f"{comparison.erc_label} still beats {comparison.benchmark_label} at "
            f"{MAX_BREAKEVEN_COST * 1e4:.0f}bp per unit traded, which is past any real "
            "market. The breakeven is beyond the range worth searching."
        )
    else:
        verdict = (
            f"Breakeven: {comparison.erc_label} stops beating "
            f"{comparison.benchmark_label} at "
            f"{comparison.breakeven_cost * 1e4:.1f}bp per unit traded, charged to both."
        )
    return f"{text}\n\n{verdict}"


def format_sweep(swept: pd.DataFrame) -> str:
    """The knob sweep, one block per knob, with losing settings called out."""
    blocks = []
    for knob in swept["knob"].unique():
        rows = swept[swept["knob"] == knob]
        rendered = pd.DataFrame(
            {
                # ":g" rather than str(): the settings column holds ints and
                # floats together, so pandas stores it as float64 and a halflife
                # of 60 would otherwise print as "60.0".
                "setting": [
                    f"{row.setting:g}{' (default)' if row.is_default else ''}"
                    for row in rows.itertuples()
                ],
                "ERC ratio": [f"{v:.3f}" for v in rows["erc_ratio"]],
                "1/N ratio": [f"{v:.3f}" for v in rows["one_n_ratio"]],
                "gap": [f"{v:+.3f}" for v in rows["gap"]],
                "ERC @1/N vol": [f"{v:.3f}" for v in rows["matched_ratio"]],
                "matched gap": [f"{v:+.3f}" for v in rows["matched_gap"]],
                "ERC bets": [f"{v:.2f}" for v in rows["erc_bets"]],
                "1/N bets": [f"{v:.2f}" for v in rows["one_n_bets"]],
                "rebals": [f"{int(v)}" for v in rows["rebalances"]],
                "days": [f"{int(v)}" for v in rows["days"]],
            }
        ).set_index("setting")
        blocks.append(f"{knob}\n{textwrap.indent(rendered.to_string(), '  ')}")
    return "\n\n".join(blocks)


def format_sensitivity_report(
    swept: pd.DataFrame,
    traded: pd.DataFrame,
    costs: CostComparison | Sequence[CostComparison],
) -> str:
    """The whole sensitivity analysis as one block of text.

    ``costs`` may be one :class:`CostComparison` or several -- the unlevered
    comparison and the matched-risk one answer different halves of the headline
    and both are worth charging a cost to.
    """
    lines = [
        "Does ERC's edge survive the covariance being estimated?",
        "  One knob moved per row, the rest at their config.py defaults, both rules",
        "  re-run together at every setting on one window and one calendar.",
        "",
        format_sweep(swept),
        "",
    ]

    # The two claims are reported apart. Unlevered, ERC is expected to win and
    # the question is at which settings it does not. At matched risk it is
    # expected to LOSE -- that is the project's headline, measured before any
    # sweep ran -- and the question is whether any setting rescues it. OR-ing
    # them into one "ERC lost here" list reads as the estimator being fragile
    # when what it is showing is the conclusion being robust.
    unlevered_losers = losing_settings(swept, "gap")
    total = len(swept)

    lines.append(f"Unlevered -- the allocation claim on its own ({total} settings):")
    if unlevered_losers.empty:
        lines.append(
            "  ERC beat 1/N on return-to-volatility at every setting swept."
        )
    else:
        lines.append(
            f"  ! ERC did NOT beat 1/N at {len(unlevered_losers)} of {total}:"
        )
        for row in unlevered_losers.itertuples():
            lines.append(
                f"      {row.knob} = {row.setting:g}: "
                f"{row.erc_ratio:.3f} against {row.one_n_ratio:.3f}"
            )

    matched_losers = losing_settings(swept, "matched_gap")
    lines.append("")
    lines.append(f"Levered to 1/N's volatility, financing charged ({total} settings):")
    if len(matched_losers) == total:
        worst = swept.loc[swept["matched_gap"].idxmin()]
        best = swept.loc[swept["matched_gap"].idxmax()]
        lines.append(
            f"  ERC was behind 1/N at EVERY setting, from {best['matched_gap']:+.3f} "
            f"({best['knob']} = {best['setting']:g}) to {worst['matched_gap']:+.3f} "
            f"({worst['knob']} = {worst['setting']:g})."
        )
        lines.append(
            "  That is this project's headline finding and the sweep makes it "
            "stronger, not weaker: no estimator setting rescues it. The gap is "
            "the financing, not the covariance."
        )
    elif matched_losers.empty:
        # Not phrased as an alarm: this is what the fixtures produce, and the
        # fixtures are shape rather than magnitude. It IS the opposite of the
        # live four-sleeve result, so the line says which question to ask.
        lines.append(
            "  ERC beat 1/N at every setting, financing and all. That is the "
            "opposite of what the live four-sleeve sample shows, where the "
            "financing costs more than the allocation gains -- so establish "
            "which data this ran on before quoting it."
        )
    else:
        lines.append(
            f"  ERC was behind at {len(matched_losers)} of {total} settings and "
            "ahead at the rest. The settings where it is ahead:"
        )
        for row in swept[swept["matched_gap"] > 0].itertuples():
            lines.append(
                f"      {row.knob} = {row.setting:g}: "
                f"{row.matched_ratio:.3f} against {row.one_n_ratio:.3f}"
            )

    lines.append("\nWhat each rule trades, and what that would cost")
    lines.append(textwrap.indent(format_turnover_table(traded), "  "))

    comparisons = [costs] if isinstance(costs, CostComparison) else list(costs)
    for comparison in comparisons:
        lines.append(
            f"\n  {comparison.erc_label} against {comparison.benchmark_label}, "
            "both charged the same rate"
        )
        lines.append(textwrap.indent(format_cost_table(comparison), "  "))

    lines.append("\nHow to read the cost block:")
    caveats = [
        caveat
        for caveat in _COST_CAVEATS
        if "levered rows" not in caveat
        or any(comparison.levered for comparison in comparisons)
    ]
    for i, caveat in enumerate(caveats, start=1):
        lines.extend(
            textwrap.wrap(f"{i}. {caveat}", width=SUMMARY_LINE_WIDTH, subsequent_indent="   ")
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    # Offline and write-free -- fixtures in, a report on stdout, nothing on disk.
    from tests.fixtures import make_fake_returns, make_fake_risk_free

    from .config import UNIVERSE

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    fake_returns = make_fake_returns(tickers=UNIVERSE)
    fake_rf = make_fake_risk_free().reindex(fake_returns.index).ffill().bfill()

    # A reduced grid: the default and the far end of each knob's range, rather
    # than the full config grids. Each setting is three engine-grade runs, and a
    # self-check that takes two minutes is a self-check nobody runs. The far end
    # rather than the near one because it is where a knob is most likely to
    # change the answer. compare_rules.py --sensitivities sweeps the real grids.
    small_knobs = tuple(
        (name, label, (default, grid[-1]), default) for name, label, grid, default in KNOBS
    )
    swept = sweep(fake_returns, fake_rf, knobs=small_knobs)

    erc = run_backtest(fake_returns, risk_free=fake_rf)
    one_n = equal_weight(
        fake_returns, cov_estimator=estimate_covariance, strategy=EQUAL_CAPITAL
    )
    index = erc.returns.index
    erc_unlevered = unlevered_returns(fake_returns.loc[index], erc.weights)
    traded = turnover_table({ERC_UNLEVERED: erc, EQUAL_CAPITAL: one_n}, fake_returns)
    erc_trades = turnover(erc.weights, fake_returns)["total"]
    one_n_trades = turnover(one_n.weights, fake_returns)["total"]
    erc_levered_trades = turnover(
        erc.weights, fake_returns, leverage=erc.leverage, risk_free=fake_rf
    )["total"]
    costs = [
        cost_comparison(erc_unlevered, erc_trades, one_n.returns, one_n_trades),
        # The levered row too: the headline conclusion is about the matched
        # column, and charging a cost only to the unlevered one would answer
        # the easier half of the question.
        cost_comparison(
            erc.returns, erc_levered_trades, one_n.returns, one_n_trades,
            levered=True,
            erc_label="ERC @ 10% target, financed",
        ),
    ]

    print(format_sensitivity_report(swept, traded, costs))
    print()

    checks = 0

    def check(label: str, condition: bool) -> None:
        global checks
        if not condition:
            raise AssertionError(f"FAILED: {label}")
        checks += 1
        print(f"  ok   {label}")

    n = len(UNIVERSE)
    erc_traded = turnover(erc.weights, fake_returns)
    one_n_traded = turnover(one_n.weights, fake_returns)

    check("the first rebalance has no turnover -- there is no previous book",
          bool(erc_traded.iloc[0].isna().all()))
    check("1/N re-optimises exactly nothing, by construction",
          float(one_n_traded["reoptimisation"].abs().max()) == 0.0)
    check("...but it still trades, because the book drifts off 25% between rebalances",
          float(one_n_traded["drift_back"].mean()) > 0.0)
    check("ERC trades more per rebalance than 1/N does",
          float(erc_traded["total"].mean()) > float(one_n_traded["total"].mean()))
    check("a trade is never more than drift-back plus re-optimisation, and usually less",
          bool((erc_traded["total"].dropna()
                <= (erc_traded["drift_back"] + erc_traded["reoptimisation"]).dropna()
                + 1e-12).all()))
    check("zero cost leaves a return series untouched, bit for bit",
          charge_costs(one_n.returns, one_n_traded["total"], 0.0).equals(one_n.returns))
    check("a cost only ever reduces a return",
          bool((charge_costs(one_n.returns, one_n_traded["total"], 0.001)
                <= one_n.returns + 1e-18).all()))
    check("the default setting is marked once per knob",
          int(swept["is_default"].sum()) == len(KNOBS))
    check("every sweep row carries its own observation count, so a shorter "
          "window cannot pass as the same sample",
          bool((swept["days"] > 0).all()))
    check("the sweep moved one knob at a time and every setting produced a result",
          len(swept) == 2 * len(KNOBS) and swept["erc_ratio"].notna().all())
    check("ERC takes N effective bets at every setting -- that is the rule, not the estimator",
          float((swept["erc_bets"] - n).abs().max()) < 1e-6)
    check("the cost grid starts at zero cost and the gap narrows as cost rises",
          costs[0].table.index[0] == 0.0
          and costs[0].table["gap"].iloc[-1] < costs[0].table["gap"].iloc[0])
    check("the levered comparison records that it was levered, so the report "
          "prints the caveat that goes with it",
          costs[1].levered and not costs[0].levered)
    check("losing_settings reads one gap column at a time, never both at once",
          len(losing_settings(swept, "gap")) == int((swept["gap"] <= 0).sum())
          and len(losing_settings(swept, "matched_gap"))
          == int((swept["matched_gap"] <= 0).sum()))
    try:
        losing_settings(swept, "no_such_column")
        raise AssertionError("FAILED: an unknown gap column did not raise")
    except SensitivityError:
        check("...and an unknown column raises rather than returning nothing", True)

    print(f"\n{checks} checks passed")
    print("\nFixture data: shape, not magnitude. Nothing was written to disk.")
