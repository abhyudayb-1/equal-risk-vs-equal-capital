"""Self-checks for the head-to-head comparison (step 44).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_compare.py

Everything here is **offline**: ``src.compare`` is a pure function of return
series, so nothing needs the network. The live wiring (the engine run, the
yfinance fetch) is ``compare_ticker.py``'s job and is exercised by running it;
the fetch's own behaviour is in ``tests/test_data.py``.

The metrics themselves -- relative metrics, the tail block, the crisis table and
both printed forms -- moved to ``src/headtohead.py`` and are checked by
``tests/test_headtohead.py``, including the hand-derived expectations that used
to live here. What is tested below is the part that is specific to a *ticker*:

* **Alignment and the guards** -- overlap is the index intersection, too little
  overlap and a foreign trading calendar both raise.
* **The two calendar tiers** -- a weekend-trading series is refused, a foreign
  weekday exchange runs and sets ``calendar_warning``.
* **The identity check can fail** -- the house technique: a candidate one day out
  of step must NOT reproduce ``beta = 0.5``, or the identity proves nothing about
  alignment.
* **The report** -- every block renders, an uncovered window says why it is
  ``n/a``, and undefined metrics never reach the reader as a number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import compare, config, metrics  # noqa: E402
from src.data import DataUnavailableError  # noqa: E402
from src.overlay import unlevered_returns  # noqa: E402
from tests.fixtures import make_fake_result, make_fake_returns  # noqa: E402

_checks = 0
TOLERANCE = 1e-12


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


def _close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# compare_to_ticker: alignment, guards, wiring
# ---------------------------------------------------------------------------
def _fixture_pair():
    result = make_fake_result()
    returns = make_fake_returns(tickers=(config.GOLD_TICKER, *config.UNIVERSE))
    book = returns[list(config.UNIVERSE)].loc[result.returns.index]
    unlevered = unlevered_returns(book, result.weights)
    return result, returns[config.GOLD_TICKER], unlevered


def test_alignment_and_wiring() -> None:
    result, gold, unlevered = _fixture_pair()
    comp = compare.compare_to_ticker(
        result.returns, gold, config.GOLD_TICKER, unlevered=unlevered, strategy_name=result.strategy
    )

    common = result.returns.index.intersection(gold.index)
    check("the overlap is the index intersection", comp.n_observations == len(common))
    check("the window runs from the first to the last shared date",
          comp.window_start == common[0] and comp.window_end == common[-1])
    check(
        "columns are the strategy, its unlevered book, then the ticker",
        list(comp.series.columns) == [result.strategy, compare.UNLEVERED_LABEL, config.GOLD_TICKER],
    )
    check("the aligned strategy column is the input, untouched",
          bool(np.array_equal(comp.series[result.strategy].to_numpy(), result.returns.loc[common].to_numpy())))
    check(
        "the summary's max drawdown agrees with metrics.max_drawdown on the same column",
        _close(comp.summary.at[config.GOLD_TICKER, "max_drawdown"],
               metrics.max_drawdown(comp.series[config.GOLD_TICKER])),
    )
    check("relative metrics are reported for both risk-parity columns",
          list(comp.relative.columns) == [result.strategy, compare.UNLEVERED_LABEL])
    check("the tail block has one row per risk-parity column",
          list(comp.tail.index) == [result.strategy, compare.UNLEVERED_LABEL])

    without = compare.compare_to_ticker(
        result.returns, gold, config.GOLD_TICKER, strategy_name=result.strategy
    )
    check("the unlevered column is optional",
          list(without.series.columns) == [result.strategy, config.GOLD_TICKER])


def test_overlap_is_the_intersection_for_a_young_ticker() -> None:
    result, gold, _ = _fixture_pair()
    young = gold.loc["2020-01-02":]
    comp = compare.compare_to_ticker(result.returns, young, "YOUNG")
    check("a ticker that starts later shrinks the window to where it starts",
          comp.window_start == young.index[0] and comp.n_observations == len(young.index.intersection(result.returns.index)))
    check("...and every crisis window before it is n/a",
          not bool(comp.crisis.loc["GFC (2008)", "covered"]))
    check("...while a window it does cover is measured",
          bool(comp.crisis.loc["Rate shock (2022)", "covered"]))


def test_guards_raise() -> None:
    result, gold, _ = _fixture_pair()
    expect_raises(
        "an overlap shorter than COMPARE_MIN_OVERLAP_DAYS raises DataUnavailableError",
        DataUnavailableError,
        lambda: compare.compare_to_ticker(result.returns, gold.iloc[-100:], "THIN"),
    )
    foreign = gold.copy()
    foreign.index = foreign.index + pd.Timedelta(days=1)  # Tue-Sat: a Saturday for every Friday
    expect_raises(
        "a ticker on a different trading calendar raises rather than misaligning",
        DataUnavailableError,
        lambda: compare.compare_to_ticker(result.returns, foreign, "FOREIGN"),
    )
    with_nan = gold.copy()
    with_nan.iloc[10] = np.nan
    expect_raises("a NaN in the ticker's returns is rejected", ValueError,
                  lambda: compare.compare_to_ticker(result.returns, with_nan, "BAD"))
    unsorted = gold.iloc[::-1]
    expect_raises("an unsorted index is rejected", ValueError,
                  lambda: compare.compare_to_ticker(result.returns, unsorted, "REV"))
    expect_raises(
        "a ticker label that collides with a strategy column is rejected",
        ValueError,
        lambda: compare.compare_to_ticker(
            result.returns, gold, config.GOLD_TICKER, strategy_name=config.GOLD_TICKER
        ),
    )


def test_identity_check_can_fail() -> None:
    """The house technique: prove the check has teeth.

    With the strategy equal to half the ticker, ``compare_to_ticker`` must report
    beta 0.5 and correlation 1. Feed it the same ticker returns one day out of
    step and it must NOT -- otherwise those identities say nothing about whether
    the two series are aligned at all.
    """
    index = pd.bdate_range("2010-01-04", periods=800)
    rng = np.random.default_rng(11)
    ticker = pd.Series(rng.normal(0.0003, 0.011, len(index)), index=index)
    strategy = 0.5 * ticker

    aligned = compare.compare_to_ticker(strategy, ticker, "TKR")
    rel = aligned.relative[strategy.name or compare.STRATEGY_LABEL]
    check("aligned: beta 0.5 and correlation 1 through compare_to_ticker",
          _close(rel["beta"], 0.5) and _close(rel["correlation"], 1.0))

    stale = ticker.shift(1).iloc[1:]  # the ticker's values, one day late
    broken = compare.compare_to_ticker(strategy.iloc[1:], stale, "TKR")
    rel_broken = broken.relative[compare.STRATEGY_LABEL]
    check("one day out of step: beta is NOT 0.5 and correlation is nowhere near 1",
          abs(rel_broken["beta"] - 0.5) > 0.2 and rel_broken["correlation"] < 0.5)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_calendar_warning_tier() -> None:
    """Two tiers: refuse what cannot be aligned, warn about what merely can be.

    A foreign weekday exchange (Tokyo, London, Toronto ... measured 96.6-98.2%
    against a US listing's 100%) is not refused, but its relative metrics are
    biased, so the comparison must say so rather than print them cleanly.
    """
    result, gold, _ = _fixture_pair()
    clean = compare.compare_to_ticker(result.returns, gold, "GLD", strategy_name=result.strategy)
    check("a ticker on the strategy's own calendar (100% match) carries no warning",
          clean.calendar_warning is None)
    check("...and its report prints none",
          "calendar warning" not in compare.format_comparison_report(clean))

    # Add ~3% extra Saturday returns: days the ticker traded and the strategy did
    # not, which is exactly what a foreign exchange's differing holidays look like.
    lo, hi = result.returns.index[0], result.returns.index[-1]
    n_in_range = int(((gold.index >= lo) & (gold.index <= hi)).sum())
    saturdays = pd.date_range(lo, hi, freq="W-SAT")[: round(0.031 * n_in_range)]
    foreign = pd.concat([gold, pd.Series(0.0005, index=saturdays)]).sort_index()

    in_range = foreign.index[(foreign.index >= lo) & (foreign.index <= hi)]
    expected = len(in_range.intersection(result.returns.index)) / len(in_range)
    check("the constructed ticker sits between the refusal and the warning thresholds",
          config.COMPARE_MIN_CALENDAR_MATCH < expected < config.COMPARE_CALENDAR_WARN_BELOW)

    warned = compare.compare_to_ticker(result.returns, foreign, "FOREIGN", strategy_name=result.strategy)
    check("it still runs (it is not refused)", warned.n_observations > 0)
    check("...the measured match agrees with the hand calculation",
          _close(warned.calendar_match, expected))
    check("...and it carries a warning that says the metrics are understated",
          isinstance(warned.calendar_warning, str) and "UNDERSTATED" in warned.calendar_warning)
    check("...which the printed report shows, marked with a leading !",
          "! calendar warning" in compare.format_comparison_report(warned))
    check("the weekend-only extra days are dropped, not paired: the overlap is unchanged",
          warned.n_observations == len(gold.index.intersection(result.returns.index)))

    silenced = compare.compare_to_ticker(
        result.returns, foreign, "FOREIGN", strategy_name=result.strategy, warn_calendar_below=0.0
    )
    check("the warning threshold is a parameter", silenced.calendar_warning is None)


def test_report_formatting() -> None:
    result, gold, unlevered = _fixture_pair()
    comp = compare.compare_to_ticker(result.returns, gold, config.GOLD_TICKER, unlevered=unlevered)
    text = compare.format_comparison_report(comp)

    for heading in (
        "head-to-head comparison",
        "Risk and return on the overlap window",
        f"Relative to {config.GOLD_TICKER}",
        f"On {config.GOLD_TICKER}'s worst 5% of days",
        "Crisis windows",
        "What this is, and is not:",
    ):
        check(f"the report has its '{heading}' section", heading in text)
    check("all five caveats are printed",
          all(f"{i}. " in text for i in range(1, len(compare._CAVEATS) + 1)) and len(compare._CAVEATS) == 5)
    check("a fully-covered comparison prints no n/a note", "n/a =" not in text)

    young = compare.compare_to_ticker(result.returns, gold.loc["2020-01-02":], "YOUNG")
    young_text = compare.format_comparison_report(young)
    check("an uncovered crisis window prints n/a and says why",
          "n/a" in young_text and "n/a = the window is not fully inside the overlap" in young_text)

    flat = pd.Series(0.0, index=gold.index)
    degenerate = compare.compare_to_ticker(result.returns, flat, "FLAT")
    degenerate_text = compare.format_comparison_report(degenerate)
    check("undefined metrics (a flat ticker's beta and Sharpe) print -- and never a number",
          "--" in degenerate_text and bool(np.isnan(degenerate.relative.loc["beta"]).all()))


if __name__ == "__main__":
    test_alignment_and_wiring()
    test_overlap_is_the_intersection_for_a_young_ticker()
    test_guards_raise()
    test_identity_check_can_fail()
    test_calendar_warning_tier()

    test_report_formatting()

    print(f"\n{_checks} checks passed")
