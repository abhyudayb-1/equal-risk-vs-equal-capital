"""Self-checks for ``src.riskmetrics`` -- realized risk and return metrics.

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_riskmetrics.py

Entirely offline. ``src.riskmetrics`` holds pure functions of a return series
and reaches nothing outside ``src.metrics`` and ``src.config``, so every check
below runs on a hand-built series of a dozen numbers or fewer.

**Every expected value here is derived by hand from the definition**, in plain
arithmetic written out in the test's own docstring -- not by calling the
function under test a second way, which would only prove it agrees with
itself. That is the same bar ``tests/test_riskparity.py``'s independent oracle
sets for the ERC solver, applied to arithmetic simple enough to do on paper.

Four groups:

* **Tail loss** -- the VaR quantile interpolated by hand and the CVaR mean of
  the observations at or below it, plus the guards (too few observations is
  NaN, a confidence outside (0, 1) raises).
* **Downside risk** -- downside deviation over the whole sample rather than
  only the losing periods, and the Sortino built on it, including the case
  with no downside at all, which must be NaN rather than a huge finite ratio.
* **Drawdown shape** -- the longest underwater stretch, its still-open flag,
  the peak floored at the 1.0 starting value inherited from ``metrics.py``,
  and Calmar against a drawdown computed by hand.
* **The row and the table** -- ``risk_summary`` carrying every key the table
  formatter reads, ``realized_risk_table`` indexing one row per book, and
  ``format_metric_table`` printing ``--`` for an undefined metric and
  ``(open)`` for an unrecovered drawdown. Those last two are what stop an
  undefined number from reading as a real one on a report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import riskmetrics  # noqa: E402

_checks = 0

# Every expectation below is exact arithmetic, so the only slack needed is
# floating-point noise.
METRIC_TOLERANCE = 1e-12


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


def expect_raises(label: str, exception: type[Exception], call) -> None:
    try:
        call()
    except exception:
        check(label, True)
        return
    raise AssertionError(f"FAILED: {label} -- no {exception.__name__} raised")


# ---------------------------------------------------------------------------
# Tail loss, downside risk and drawdown shape
# ---------------------------------------------------------------------------
def test_historical_var_and_cvar_hand_derived() -> None:
    """r_k = (k - 51) / 1000 for k = 1..100, so the sorted returns are
    -0.050, -0.049, ..., +0.049. The 5% quantile sits at position 0.05 * 99 =
    4.95 of the sorted array, i.e. 95% of the way from sorted[4] = -0.046 to
    sorted[5] = -0.045: VaR = -0.046 + 0.95 * 0.001 = -0.04505. The tail at or
    below it is exactly sorted[0..4] = -0.050 .. -0.046, mean -0.048.
    """
    r = pd.Series((np.arange(1, 101) - 51) / 1000.0)
    check(
        "historical VaR 95% equals the hand-interpolated quantile -0.04505",
        abs(riskmetrics.historical_var(r, 0.95) - (-0.04505)) < METRIC_TOLERANCE,
    )
    check(
        "historical CVaR 95% equals the mean of the five worst returns, -0.048",
        abs(riskmetrics.historical_cvar(r, 0.95) - (-0.048)) < METRIC_TOLERANCE,
    )
    check(
        "CVaR is never better than VaR",
        riskmetrics.historical_cvar(r, 0.95) <= riskmetrics.historical_var(r, 0.95),
    )
    check(
        "VaR on fewer than two observations is NaN, not a number from nothing",
        np.isnan(riskmetrics.historical_var(pd.Series([0.01]))),
    )
    expect_raises(
        "a confidence outside (0, 1) is rejected",
        ValueError,
        lambda: riskmetrics.historical_var(r, 1.0),
    )


def test_downside_deviation_and_sortino_hand_derived() -> None:
    """r = [+1%, -2%, +3%, -1%] at rf = 0: the shortfalls are [0, -.02, 0, -.01],
    mean square (0.0004 + 0.0001) / 4 = 0.000125 over ALL four periods, and
    the mean return is 0.0025.
    """
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    per_period_downside = math.sqrt((0.0004 + 0.0001) / 4.0)
    annual = math.sqrt(252)
    check(
        "annualised downside deviation matches sqrt(mean shortfall^2) * sqrt(252)",
        abs(riskmetrics.downside_deviation(r, 252, 0.0) - per_period_downside * annual)
        < METRIC_TOLERANCE,
    )
    check(
        "Sortino matches mean excess / downside deviation * sqrt(252)",
        abs(riskmetrics.sortino_ratio(r, 252, 0.0) - 0.0025 / per_period_downside * annual)
        < METRIC_TOLERANCE,
    )

    no_downside = pd.Series([0.001, 0.002, 0.003, 0.002])
    check(
        "a series with no losing period has zero downside deviation",
        riskmetrics.downside_deviation(no_downside, 252, 0.0) == 0.0,
    )
    check(
        "...and an UNDEFINED Sortino (NaN), not a huge finite ratio",
        np.isnan(riskmetrics.sortino_ratio(no_downside, 252, 0.0)),
    )


def test_drawdown_duration_and_calmar_hand_derived() -> None:
    """r = [+10%, -5%, -5%, +5%, +10%]: the curve is 1.1, 1.045, 0.99275,
    1.0424, 1.1466. Periods 2-4 sit below the 1.1 peak (three of them) and
    period 5 makes a new high, so the longest stretch is 3 and it recovered.
    The trough is 1.1 * 0.95^2, a drawdown of 1 - 0.9025 = 9.75% exactly.
    """
    r = pd.Series([0.10, -0.05, -0.05, 0.05, 0.10])
    days, still_open = riskmetrics.drawdown_duration(r)
    check("longest drawdown lasts 3 periods on the hand-built path", days == 3)
    check("...and it has recovered by the end of the series", still_open is False)

    days, still_open = riskmetrics.drawdown_duration(r.iloc[:4])
    check("cutting the series before the recovery keeps the 3-period stretch", days == 3)
    check("...and flags it as still open at the window end", still_open is True)

    days, still_open = riskmetrics.drawdown_duration(pd.Series([-0.05, -0.05]))
    check(
        "a series that starts below its initial value is underwater from the first "
        "period (the peak is floored at the 1.0 starting value, as in metrics.py)",
        (days, still_open) == (2, True),
    )

    # ppy = 5 makes the five periods exactly one year, so CAGR is just growth - 1.
    growth = 1.10 * 0.95 * 0.95 * 1.05 * 1.10
    expected = (growth - 1.0) / (1.0 - 0.95 * 0.95)
    check(
        "Calmar equals CAGR over the 9.75% max drawdown",
        abs(riskmetrics.calmar_ratio(r, 5) - expected) < METRIC_TOLERANCE,
    )
    check(
        "Calmar is NaN for a series that never draws down",
        np.isnan(riskmetrics.calmar_ratio(pd.Series([0.001, 0.002, 0.003]), 252)),
    )


# ---------------------------------------------------------------------------
# The summary row and the shared table
# ---------------------------------------------------------------------------
# riskmetrics.format_metric_table is the one place an undefined metric can
# still reach a reader as a number, so the table gets its own checks rather
# than being trusted because the metrics under it are tested.
_DATES = pd.date_range("2020-01-01", periods=5, freq="D")

# longest stretch underwater is 3 periods and it recovers; max drawdown is
# 1 - 0.95^2 = 9.75% exactly (the series test_drawdown_duration_and_calmar
# derives by hand above).
RECOVERING = pd.Series([0.10, -0.05, -0.05, 0.05, 0.10], index=_DATES)
# never falls below its 1.0 starting value: no drawdown, no losing period, so
# Calmar and Sortino are both undefined.
RISING = pd.Series([0.001, 0.002, 0.003, 0.002, 0.001], index=_DATES)
# underwater from the very first period and never recovers.
SINKING = pd.Series([-0.05] * 5, index=_DATES)


def test_risk_summary_carries_every_row_the_table_reads() -> None:
    """``risk_summary`` assembles a row out of the functions above, and the
    failure mode of an assembler is a mislabelled slot -- CVaR reported in the
    VaR cell reads perfectly plausibly. So each key is checked against the
    function it is supposed to have come from, and the whole key set is
    checked against what the table formatter will go looking for.
    """
    row = riskmetrics.risk_summary(RECOVERING, 5, "recovering")

    wanted = {key for _, key, _ in riskmetrics._REALIZED_ROWS}
    check(
        "risk_summary carries every key the shared table formatter reads",
        wanted <= set(row),
    )
    check(
        "...plus the open-drawdown flag the formatter needs to mark it",
        "longest_drawdown_open" in row,
    )
    check("the row keeps the name it was given", row["name"] == "recovering")

    check(
        "the var cell holds historical_var, not some other tail number",
        row["var"] == riskmetrics.historical_var(RECOVERING),
    )
    check(
        "the cvar cell holds historical_cvar",
        row["cvar"] == riskmetrics.historical_cvar(RECOVERING),
    )
    check(
        "the sortino cell holds sortino_ratio at the same periods_per_year",
        row["sortino_ratio"] == riskmetrics.sortino_ratio(RECOVERING, 5),
    )
    check(
        "the calmar cell holds calmar_ratio at the same periods_per_year",
        row["calmar_ratio"] == riskmetrics.calmar_ratio(RECOVERING, 5),
    )
    check(
        "the downside_deviation cell holds downside_deviation",
        row["downside_deviation"] == riskmetrics.downside_deviation(RECOVERING, 5),
    )
    check(
        "worst_day is the minimum observation, -5% here",
        abs(row["worst_day"] - (-0.05)) < METRIC_TOLERANCE,
    )
    check(
        "longest_drawdown_days agrees with drawdown_duration's 3 periods",
        (row["longest_drawdown_days"], row["longest_drawdown_open"])
        == riskmetrics.drawdown_duration(RECOVERING),
    )


def test_realized_risk_table_is_one_row_per_book() -> None:
    books = pd.DataFrame({"recovering": RECOVERING, "rising": RISING, "sinking": SINKING})
    table = riskmetrics.realized_risk_table(books, periods_per_year=5)

    check(
        "the table has one row per book, in the frame's column order",
        list(table.index) == ["recovering", "rising", "sinking"],
    )
    check(
        "a cell matches risk_summary called directly on that column",
        table.at["sinking", "var"] == riskmetrics.risk_summary(SINKING, 5)["var"],
    )
    check(
        "the book that never draws down has an UNDEFINED Calmar in the table",
        bool(pd.isna(table.at["rising", "calmar_ratio"])),
    )
    check(
        "the book still underwater at the end is flagged open",
        bool(table.at["sinking", "longest_drawdown_open"]),
    )


def test_format_metric_table_marks_undefined_and_open_drawdowns() -> None:
    """The formatter is the last chance to stop an undefined metric reading as
    a real one, so it is checked on a frame deliberately built to contain
    both: a book with no downside (undefined Sortino and Calmar) and a book
    that never recovers (an open drawdown).
    """
    books = pd.DataFrame({"recovering": RECOVERING, "rising": RISING, "sinking": SINKING})
    table = riskmetrics.realized_risk_table(books, periods_per_year=5)
    text = riskmetrics.format_metric_table(table)

    for label, _, _ in riskmetrics._REALIZED_ROWS:
        check(f"the table prints a '{label}' row", label in text)
    check("an undefined metric reaches the reader as -- and not as a number", "--" in text)
    check(
        "an unrecovered longest drawdown is flagged (open)",
        "(open)" in text,
    )
    check(
        "...on exactly the one book of the three that never recovered, so the "
        "flag marks a book rather than decorating the whole row",
        text.count("(open)") == 1,
    )

    check(
        "an undefined metric formats as -- not as 0",
        riskmetrics._format_metric(float("nan"), "ratio") == "--",
    )
    check(
        "an undefined change formats as -- too",
        riskmetrics._format_delta(float("nan"), "pct") == "--",
    )
    check(
        "a percentage-point change carries its sign",
        riskmetrics._format_delta(-0.0123, "pct") == "-1.23pp",
    )


if __name__ == "__main__":
    test_historical_var_and_cvar_hand_derived()
    test_downside_deviation_and_sortino_hand_derived()
    test_drawdown_duration_and_calmar_hand_derived()

    test_risk_summary_carries_every_row_the_table_reads()
    test_realized_risk_table_is_one_row_per_book()
    test_format_metric_table_marks_undefined_and_open_drawdowns()

    print(f"\n{_checks} checks passed")
