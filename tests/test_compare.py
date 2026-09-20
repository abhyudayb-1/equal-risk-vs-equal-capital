"""Self-checks for the head-to-head comparison (step 44).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_compare.py

Everything here is **offline**: ``src.compare`` is a pure function of return
series, so nothing needs the network. The live wiring (the engine run, the
yfinance fetch) is ``compare_ticker.py``'s job and is exercised by running it;
the fetch's own behaviour is in ``tests/test_data.py``.

Every expected value below is derived from the definition, by hand, rather than
by calling the function under test a second way:

* **Relative metrics** on pairs whose answer is known exactly: ``s = 0.5 * t``
  must give correlation 1, beta 0.5 and both captures 0.5; ``s = -t`` must give
  -1 throughout; and a four-day example worked out on paper.
* **The tail block** on a hundred returns whose worst five days are known.
* **Crisis windows**: a constant daily return compounds to a number computed
  from the day count, a huge return on the peak day itself must NOT be counted,
  and a ticker that listed after the window's start must give ``n/a`` rather
  than a partial window.
* **Alignment and the guards** -- overlap is the index intersection, too little
  overlap and a foreign trading calendar both raise.
* **The identity check can fail** -- the house technique: a candidate one day out
  of step must NOT reproduce ``beta = 0.5``, or the identity proves nothing about
  alignment.
"""

from __future__ import annotations

import math
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


def _daily(values, start: str = "2015-01-05") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def _ticker_series(n: int = 600, seed: int = 3, start: str = "2015-01-05") -> pd.Series:
    rng = np.random.default_rng(seed)
    return _daily(rng.normal(0.0004, 0.012, n), start)


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------
def test_relative_metrics_known_pairs() -> None:
    t = _ticker_series()

    half = compare.relative_metrics(0.5 * t, t)
    check("s = 0.5 t: correlation is exactly 1", _close(half["correlation"], 1.0))
    check("s = 0.5 t: beta is 0.5", _close(half["beta"], 0.5))
    check("s = 0.5 t: up and down capture are both 0.5",
          _close(half["up_capture"], 0.5) and _close(half["down_capture"], 0.5))
    check(
        "s = 0.5 t: tracking error is the annualised std of 0.5 t",
        _close(half["tracking_error"], 0.5 * t.std(ddof=1) * math.sqrt(252)),
    )

    inverse = compare.relative_metrics(-t, t)
    check("s = -t: correlation, beta and both captures are -1",
          all(_close(inverse[k], -1.0) for k in ("correlation", "beta", "up_capture", "down_capture")))

    same = compare.relative_metrics(t, t)
    check("s = t: tracking error is zero", _close(same["tracking_error"], 0.0))


def test_relative_metrics_four_day_example_by_hand() -> None:
    """t = [+2%, -1%, +4%, -3%], s = [+1%, +0.5%, +1%, -1%].

    Up days (t > 0) are days 1 and 3: mean t = 0.03, mean s = 0.01, so up
    capture = 1/3. Down days are 2 and 4: mean t = -0.02, mean s = -0.0025, so
    down capture = 0.125. Deviations from the means (t 0.005, s 0.00375) give a
    covariance sum of 0.000775 and a variance sum of 0.0029, so beta =
    0.000775 / 0.0029 (the ddof cancels).
    """
    t = _daily([0.02, -0.01, 0.04, -0.03])
    s = _daily([0.01, 0.005, 0.01, -0.01])
    result = compare.relative_metrics(s, t)
    check("up capture is 1/3", _close(result["up_capture"], 1.0 / 3.0))
    check("down capture is 0.125", _close(result["down_capture"], 0.125))
    check("beta is 0.000775 / 0.0029", _close(result["beta"], 0.000775 / 0.0029))


def test_relative_metrics_degenerate_inputs() -> None:
    flat = _daily([0.001] * 50)
    result = compare.relative_metrics(_ticker_series(50), flat)
    check("a ticker with no variance has an UNDEFINED beta and correlation (NaN)",
          math.isnan(result["beta"]) and math.isnan(result["correlation"]))
    all_up = compare.relative_metrics(_ticker_series(50), _daily([0.001] * 50))
    check("no down days means an undefined down capture, not a zero",
          math.isnan(all_up["down_capture"]))
    expect_raises(
        "series on different indexes are rejected",
        ValueError,
        lambda: compare.relative_metrics(_ticker_series(50), _ticker_series(50, start="2016-01-04")),
    )


# ---------------------------------------------------------------------------
# The tail block
# ---------------------------------------------------------------------------
def test_tail_block_hand_derived() -> None:
    """t_k = (k - 51) / 1000 for k = 1..100, so the ticker's 5% VaR is -0.04505
    and the days at or below it are k = 1..5 (-0.050 .. -0.046), mean -0.048.
    The strategy returns +1% on the first three of them and -6% on the last
    two: mean (3 * 0.01 - 2 * 0.06) / 5 = -0.018. It beats the ticker on the
    first three (+1% > -5.0%, -4.9%, -4.8%) and loses on the last two
    (-6% < -4.7%, -4.6%): 3 of 5 = 0.6.
    """
    t = _daily((np.arange(1, 101) - 51) / 1000.0)
    s = _daily(np.zeros(100))
    s.iloc[0:3] = 0.01
    s.iloc[3:5] = -0.06
    block = compare.tail_block(s, t, 0.95)
    check("the tail is the five worst days", block["n_days"] == 5)
    check("the ticker averages -0.048 on them", _close(block["ticker_mean"], -0.048))
    check("the strategy averages -0.018 on them", _close(block["strategy_mean"], -0.018))
    check("the strategy beat the ticker on 3 of the 5", _close(block["strategy_beat_share"], 0.6))
    expect_raises(
        "series on different indexes are rejected",
        ValueError,
        lambda: compare.tail_block(_daily([0.0] * 100, "2016-01-04"), t),
    )


# ---------------------------------------------------------------------------
# Crisis windows
# ---------------------------------------------------------------------------
def test_crisis_table_by_hand() -> None:
    index = pd.bdate_range("2007-01-01", "2024-12-31")
    constant = pd.Series(0.001, index=index)
    table = compare.crisis_table(constant, constant)

    start, end = pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")
    n_days = len(pd.bdate_range(start + pd.Timedelta(days=1), end))
    expected = 1.001**n_days - 1.0
    covid = table.loc["COVID (2020)"]
    check("every window is covered on a full-history series", bool(table["covered"].all()))
    check(
        "a constant 0.1% a day compounds over exactly the days AFTER the peak",
        _close(covid["strategy_return"], expected, 1e-9) and _close(covid["ticker_return"], expected, 1e-9),
    )

    spiked = constant.copy()
    spiked.loc[start] = 0.5  # the peak day's own return: it happened on the way UP
    spiked_table = compare.crisis_table(spiked, constant)
    check(
        "a huge return on the peak day itself is excluded from the window",
        _close(spiked_table.loc["COVID (2020)", "strategy_return"], expected, 1e-9),
    )

    falling = pd.Series(-0.001, index=index)
    down = compare.crisis_table(falling, falling).loc["COVID (2020)"]
    check("a steadily falling series draws down by exactly its total decline",
          _close(down["strategy_max_drawdown"], down["strategy_return"], 1e-9))

    late_index = pd.bdate_range("2023-01-02", "2024-12-31")  # after all three windows
    late = pd.Series(0.001, index=late_index)
    late_table = compare.crisis_table(late, late)
    check("a series that starts after every window has none of them covered",
          not late_table["covered"].any())
    check("...and reports NaN for them, not a partial window",
          bool(late_table[["strategy_return", "ticker_return"]].isna().all().all()))

    partial = compare.crisis_table(
        pd.Series(0.001, index=pd.bdate_range("2021-01-04", "2024-12-31")),
        pd.Series(0.001, index=pd.bdate_range("2021-01-04", "2024-12-31")),
    )
    check("a window that ended before the data starts is not covered",
          not bool(partial.loc["GFC (2008)", "covered"]))


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
    test_relative_metrics_known_pairs()
    test_relative_metrics_four_day_example_by_hand()
    test_relative_metrics_degenerate_inputs()

    test_tail_block_hand_derived()

    test_crisis_table_by_hand()

    test_alignment_and_wiring()
    test_overlap_is_the_intersection_for_a_young_ticker()
    test_guards_raise()
    test_identity_check_can_fail()

    test_report_formatting()

    print(f"\n{_checks} checks passed")
