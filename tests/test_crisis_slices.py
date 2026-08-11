"""Self-checks for the crisis-window helpers in ``src/metrics.py`` (TASKS.md step 13).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_crisis_slices.py

Three things here are worth more than they look.

The first is that a sliced ``BacktestResult`` is still contract-valid. Cutting all
four fields to the same dates does *not* work -- the rebalance frames are monthly
and the returns are daily, so a window opening mid-month would leave returns that
precede the first allocation, which ``validate()`` rejects. The anchoring test
below is what pins that behaviour down.

The second is that every rejection path actually rejects. A mistyped window name
or a window that misses the data must raise, because the alternative is a
plausible-looking row of NaNs in the crisis table.

The third is the drawdown peak floor: crisis windows are sliced peak-to-trough,
so the first period is usually negative, and that is precisely the case the floor
in ``_drawdown_series`` was carried over from the sp500 repo to handle.

Per the "Data reality check" in TASKS.md, the fixtures are shape-accurate, not
magnitude-accurate -- so nothing here asserts on a realized magnitude from a
single simulated path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, metrics  # noqa: E402
from src.types import ContractViolation  # noqa: E402
from tests.fixtures import make_fake_result  # noqa: E402

_checks = 0


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


def test_resolve_window() -> None:
    print("window resolution:")
    for name, (start, end) in config.CRISIS_WINDOWS.items():
        check(
            f"{name} resolves to {start} -> {end}",
            metrics.resolve_window(name) == (pd.Timestamp(start), pd.Timestamp(end)),
        )

    open_start, open_end = metrics.resolve_window(config.REGIME_SHIFT_WINDOW)
    check(
        "REGIME_SHIFT_WINDOW keeps an open end rather than inventing one",
        open_start == pd.Timestamp("2021-01-01") and open_end is None,
    )
    expect_raises(
        "an unknown window name raises instead of returning something empty",
        KeyError,
        lambda: metrics.resolve_window("Taper tantrum"),
    )
    expect_raises(
        "a backwards window raises",
        ValueError,
        lambda: metrics.resolve_window(("2022-10-12", "2022-01-03")),
    )


def test_slice_dated() -> None:
    print("slicing a plain series:")
    daily = pd.Series(
        0.001, index=pd.bdate_range("2020-01-01", "2020-12-31"), name="returns"
    )

    # Both endpoints inclusive -- an exclusive end would drop the trough, which
    # is the single most important day in a peak-to-trough window.
    both = metrics.slice_dated(daily, ("2020-03-02", "2020-03-06"))
    check(
        "both endpoints are included",
        len(both) == 5
        and both.index[0] == pd.Timestamp("2020-03-02")
        and both.index[-1] == pd.Timestamp("2020-03-06"),
    )
    check(
        "an open-ended window runs to the end of the data",
        metrics.slice_dated(daily, ("2020-12-01", None)).index[-1] == daily.index[-1],
    )
    check(
        "a DataFrame slices the same way a Series does",
        metrics.slice_dated(
            pd.DataFrame({"a": daily, "b": daily}), ("2020-03-02", "2020-03-06")
        ).shape == (5, 2),
    )
    expect_raises(
        "a window that misses the data entirely raises",
        ValueError,
        lambda: metrics.slice_dated(daily, ("1995-01-01", "1995-12-31")),
    )


def test_slice_result_keeps_the_contract() -> None:
    """The reason this function is not a four-line slice."""
    print("slicing a BacktestResult:")
    full = make_fake_result()

    for name in config.CRISIS_WINDOWS:
        window_start, window_end = metrics.resolve_window(name)
        sliced = metrics.slice_to_window(full, name)

        # slice_to_window validates internally; calling it again documents that
        # the contract is the actual bar, not an incidental property.
        sliced.validate()
        check(f"{name}: the sliced result satisfies the contract", True)

        check(
            f"{name}: returns stay inside the window",
            bool(sliced.returns.index[0] >= window_start)
            and bool(sliced.returns.index[-1] <= window_end),
        )
        check(
            f"{name}: anchored to the allocation in force when the window opened",
            sliced.weights.index[0] <= window_start,
        )
        check(
            f"{name}: no return precedes its allocation",
            sliced.returns.index[0] >= sliced.weights.index[0],
        )
        check(
            f"{name}: weights, risk contributions and leverage stay aligned",
            sliced.weights.index.equals(sliced.risk_contributions.index)
            and sliced.weights.index.equals(sliced.leverage.index),
        )

    check(
        "the strategy name survives the slice",
        metrics.slice_to_window(full, "GFC (2008)").meta["sliced_from"] == full.strategy,
    )
    expect_raises(
        "a window past the end of the backtest raises",
        ValueError,
        lambda: metrics.slice_to_window(full, ("2090-01-01", "2090-12-31")),
    )


def test_naive_slice_would_break() -> None:
    """Proves the anchoring is load-bearing rather than defensive decoration.

    Cutting all four fields to the window -- the obvious implementation -- leaves
    daily returns that start before the first in-window rebalance. If this ever
    stops raising, ``slice_to_window`` can be simplified; until then it cannot.
    """
    print("the naive implementation:")
    full = make_fake_result()
    start, end = metrics.resolve_window("GFC (2008)")

    reb = full.weights.index
    in_window = reb[(reb >= start) & (reb <= end)]

    from src.types import BacktestResult

    expect_raises(
        "slicing every field to the same dates violates the contract",
        ContractViolation,
        lambda: BacktestResult(
            weights=full.weights.loc[in_window],
            risk_contributions=full.risk_contributions.loc[in_window],
            leverage=full.leverage.loc[in_window],
            returns=full.returns.loc[start:end],
            meta=dict(full.meta),
        ).validate(),
    )


def test_summarize_window() -> None:
    print("per-window summary:")
    full = make_fake_result()
    rows = [metrics.summarize_window(full, name) for name in config.CRISIS_WINDOWS]
    table = pd.DataFrame(rows).set_index("name")

    check(
        f"one row per crisis window ({len(config.CRISIS_WINDOWS)})",
        len(table) == len(config.CRISIS_WINDOWS),
    )
    check(
        "rows are keyed by the window name from config",
        list(table.index) == list(config.CRISIS_WINDOWS),
    )
    # The four statistics step 13 names, plus the period count behind them.
    for column in (
        "total_return", "annual_volatility", "max_drawdown", "sharpe_ratio", "n_periods"
    ):
        check(
            f"every row carries a finite {column}",
            bool(np.isfinite(table[column].to_numpy(dtype="float64")).all()),
        )
    check(
        "every row reports which strategy it summarises",
        bool((table["strategy"] == full.strategy).all()),
    )
    check(
        "the fixture spans all three crises, so every window is fully covered",
        bool(table["fully_covered"].all()),
    )
    check(
        "summarising an already-sliced result gives the same row",
        np.isclose(
            metrics.summarize_window(
                metrics.slice_to_window(full, "COVID (2020)")
            )["total_return"],
            metrics.summarize_window(full, "COVID (2020)")["total_return"],
        ),
    )
    check(
        "an open-ended window summarises to the end of the sample",
        metrics.summarize_window(full, config.REGIME_SHIFT_WINDOW)["covered_end"]
        == full.returns.index[-1],
    )


def test_drawdown_floor_survives_slicing() -> None:
    """The port's peak floor, asserted where it actually matters.

    A crisis window opens at a market peak, so its first period is negative. With
    the running peak floored at the 1.0 starting value the decline is measured
    from that peak; without the floor it would be measured from the first
    period's *close*, understating every crisis in the table.
    """
    print("drawdown floor under slicing:")
    index = pd.bdate_range("2020-01-01", periods=10)
    falling = pd.Series([-0.10] + [0.0] * 9, index=index, name="returns")

    check(
        "a window whose first day is -10% reports a -10% drawdown, not 0%",
        bool(np.isclose(metrics.max_drawdown(falling), -0.10)),
    )
    check(
        "the peak is None when the decline runs from the window open",
        metrics.max_drawdown_dates(falling)[0] is None,
    )


if __name__ == "__main__":
    test_resolve_window()
    test_slice_dated()
    test_slice_result_keeps_the_contract()
    test_naive_slice_would_break()
    test_summarize_window()
    test_drawdown_floor_survives_slicing()
    print(f"\n{_checks} checks passed")
