"""Self-checks for ``src.headtohead`` -- pairwise comparison of two return series.

Runs standalone with no test framework::

    .venv/bin/python tests/test_headtohead.py

Entirely offline. ``src.headtohead`` is pure functions of return series, so
every check below runs on a hand-built series or a seeded random one; nothing
fetches anything.

**Every expected value is derived from the definition, by hand**, rather than by
calling the function under test a second way:

* **Relative metrics** on pairs whose answer is known exactly: ``s = 0.5 * t``
  must give correlation 1, beta 0.5 and both captures 0.5; ``s = -t`` must give
  -1 throughout; and a four-day example worked out on paper. Plus the degenerate
  cases -- a benchmark with no variance has an undefined beta, not a zero, and
  two series on different indexes are rejected rather than silently aligned.
* **The tail block** on a hundred returns whose worst five days are known, with
  the strategy built so that it beats the benchmark on exactly three of them.
* **Crisis windows**: a constant daily return compounds to a number computed
  from the day count; a huge return on the peak day itself must NOT be counted
  (the window is measured from the peak close); and a series starting after a
  window must report ``NaN`` rather than a partial window.
* **The printed forms**: an undefined relative metric renders ``--`` and an
  uncovered crisis window renders ``n/a`` in every cell -- never ``0.00%``,
  which would read as "came through it flat" instead of "was not trading yet".

``src.comparison`` is the consumer, and this suite covers the printed forms as well as the
arithmetic — that report's tables are only as good as these.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import headtohead  # noqa: E402

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


def _random_series(n: int = 600, seed: int = 3, start: str = "2015-01-05") -> pd.Series:
    rng = np.random.default_rng(seed)
    return _daily(rng.normal(0.0004, 0.012, n), start)


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------
def test_relative_metrics_known_pairs() -> None:
    t = _random_series()

    half = headtohead.relative_metrics(0.5 * t, t)
    check("s = 0.5 t: correlation is exactly 1", _close(half["correlation"], 1.0))
    check("s = 0.5 t: beta is 0.5", _close(half["beta"], 0.5))
    check("s = 0.5 t: up and down capture are both 0.5",
          _close(half["up_capture"], 0.5) and _close(half["down_capture"], 0.5))
    check(
        "s = 0.5 t: tracking error is the annualised std of 0.5 t",
        _close(half["tracking_error"], 0.5 * t.std(ddof=1) * math.sqrt(252)),
    )

    inverse = headtohead.relative_metrics(-t, t)
    check("s = -t: correlation, beta and both captures are -1",
          all(_close(inverse[k], -1.0) for k in ("correlation", "beta", "up_capture", "down_capture")))

    same = headtohead.relative_metrics(t, t)
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
    result = headtohead.relative_metrics(s, t)
    check("up capture is 1/3", _close(result["up_capture"], 1.0 / 3.0))
    check("down capture is 0.125", _close(result["down_capture"], 0.125))
    check("beta is 0.000775 / 0.0029", _close(result["beta"], 0.000775 / 0.0029))


def test_relative_metrics_degenerate_inputs() -> None:
    flat = _daily([0.001] * 50)
    result = headtohead.relative_metrics(_random_series(50), flat)
    check("a benchmark with no variance has an UNDEFINED beta and correlation (NaN)",
          math.isnan(result["beta"]) and math.isnan(result["correlation"]))
    all_up = headtohead.relative_metrics(_random_series(50), _daily([0.001] * 50))
    check("no down days means an undefined down capture, not a zero",
          math.isnan(all_up["down_capture"]))
    expect_raises(
        "series on different indexes are rejected",
        ValueError,
        lambda: headtohead.relative_metrics(_random_series(50), _random_series(50, start="2016-01-04")),
    )


# ---------------------------------------------------------------------------
# The tail block
# ---------------------------------------------------------------------------
def test_tail_block_hand_derived() -> None:
    """t_k = (k - 51) / 1000 for k = 1..100, so the benchmark's 5% VaR is -0.04505
    and the days at or below it are k = 1..5 (-0.050 .. -0.046), mean -0.048.
    The strategy returns +1% on the first three of them and -6% on the last
    two: mean (3 * 0.01 - 2 * 0.06) / 5 = -0.018. It beats the benchmark on the
    first three (+1% > -5.0%, -4.9%, -4.8%) and loses on the last two
    (-6% < -4.7%, -4.6%): 3 of 5 = 0.6.
    """
    t = _daily((np.arange(1, 101) - 51) / 1000.0)
    s = _daily(np.zeros(100))
    s.iloc[0:3] = 0.01
    s.iloc[3:5] = -0.06
    block = headtohead.tail_block(s, t, 0.95)
    check("the tail is the five worst days", block["n_days"] == 5)
    check("the benchmark averages -0.048 on them", _close(block["benchmark_mean"], -0.048))
    check("the strategy averages -0.018 on them", _close(block["strategy_mean"], -0.018))
    check("the strategy beat the benchmark on 3 of the 5", _close(block["strategy_beat_share"], 0.6))
    expect_raises(
        "series on different indexes are rejected",
        ValueError,
        lambda: headtohead.tail_block(_daily([0.0] * 100, "2016-01-04"), t),
    )


# ---------------------------------------------------------------------------
# Crisis windows
# ---------------------------------------------------------------------------
def test_crisis_table_by_hand() -> None:
    index = pd.bdate_range("2007-01-01", "2024-12-31")
    constant = pd.Series(0.001, index=index)
    table = headtohead.crisis_table(constant, constant)

    start, end = pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")
    n_days = len(pd.bdate_range(start + pd.Timedelta(days=1), end))
    expected = 1.001**n_days - 1.0
    covid = table.loc["COVID (2020)"]
    check("every window is covered on a full-history series", bool(table["covered"].all()))
    check(
        "a constant 0.1% a day compounds over exactly the days AFTER the peak",
        _close(covid["strategy_return"], expected, 1e-9) and _close(covid["benchmark_return"], expected, 1e-9),
    )

    spiked = constant.copy()
    spiked.loc[start] = 0.5  # the peak day's own return: it happened on the way UP
    spiked_table = headtohead.crisis_table(spiked, constant)
    check(
        "a huge return on the peak day itself is excluded from the window",
        _close(spiked_table.loc["COVID (2020)", "strategy_return"], expected, 1e-9),
    )

    falling = pd.Series(-0.001, index=index)
    down = headtohead.crisis_table(falling, falling).loc["COVID (2020)"]
    check("a steadily falling series draws down by exactly its total decline",
          _close(down["strategy_max_drawdown"], down["strategy_return"], 1e-9))

    late_index = pd.bdate_range("2023-01-02", "2024-12-31")  # after all three windows
    late = pd.Series(0.001, index=late_index)
    late_table = headtohead.crisis_table(late, late)
    check("a series that starts after every window has none of them covered",
          not late_table["covered"].any())
    check("...and reports NaN for them, not a partial window",
          bool(late_table[["strategy_return", "benchmark_return"]].isna().all().all()))

    partial = headtohead.crisis_table(
        pd.Series(0.001, index=pd.bdate_range("2021-01-04", "2024-12-31")),
        pd.Series(0.001, index=pd.bdate_range("2021-01-04", "2024-12-31")),
    )
    check("a window that ended before the data starts is not covered",
          not bool(partial.loc["GFC (2008)", "covered"]))

# ---------------------------------------------------------------------------
# The printed forms
# ---------------------------------------------------------------------------
# The printed forms are shared, so they get direct checks -- and the one that matters is the "n/a" rendering: an uncovered crisis window
# printed as 0.00% would read as "came through it flat".
def test_format_relative_table_labels_and_undefined_cells() -> None:
    relative = pd.DataFrame(
        {
            "Risk parity": {
                "correlation": 0.42,
                "beta": 0.31,
                "tracking_error": 0.1234,
                "up_capture": 0.78,
                "down_capture": -0.05,
            },
            "Risk parity (unlevered)": {
                "correlation": float("nan"),
                "beta": float("nan"),
                "tracking_error": 0.0,
                "up_capture": 0.5,
                "down_capture": 0.5,
            },
        }
    )
    text = headtohead.format_relative_table(relative, "Equal weight")

    check("the beta row names the benchmark it is a beta TO", "beta to Equal weight" in text)
    for label in ("correlation", "tracking error (annual)", "up capture", "down capture"):
        check(f"the table prints a '{label}' row", label in text)
    check("one column per series being compared", text.count("Risk parity") >= 2)
    check("tracking error prints as a two-decimal percentage", "12.34%" in text)
    check("a capture prints as a whole percentage, since it is a multiple", "78%" in text)
    check("a negative capture keeps its sign", "-5%" in text)
    check("an undefined relative metric prints -- and not a number", "--" in text)


def test_format_crisis_table_renders_uncovered_windows_as_na() -> None:
    index = pd.bdate_range("2007-01-01", "2024-12-31")
    full = pd.Series(0.001, index=index)
    covered = headtohead.crisis_table(full, full)
    text = headtohead.format_crisis_table(covered, "Equal weight")

    check(
        "the columns name the benchmark on both of its rows",
        "Equal weight return" in text and "Equal weight max DD" in text,
    )
    check("a fully-covered table prints no n/a at all", "n/a" not in text)

    late = pd.Series(0.001, index=pd.bdate_range("2023-01-02", "2024-12-31"))
    late_table = headtohead.crisis_table(late, late)
    late_text = headtohead.format_crisis_table(late_table, "Equal weight")
    check(
        "an uncovered window prints n/a in all four of its cells",
        late_text.count("n/a") == 4 * len(late_table),
    )
    check(
        "...and never 0.00%, which would read as 'came through it flat' rather "
        "than 'was not trading yet'",
        "0.00%" not in late_text,
    )


if __name__ == "__main__":
    test_relative_metrics_known_pairs()
    test_relative_metrics_four_day_example_by_hand()
    test_relative_metrics_degenerate_inputs()

    test_tail_block_hand_derived()

    test_crisis_table_by_hand()

    test_format_relative_table_labels_and_undefined_cells()
    test_format_crisis_table_renders_uncovered_windows_as_na()

    print(f"\n{_checks} checks passed")
