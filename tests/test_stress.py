"""Self-checks for ``src/stress.py`` and ``src/viz.py`` (TASKS.md steps 24-27).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_stress.py

What is worth asserting here, and what is not
---------------------------------------------
Per the "Data reality check" in TASKS.md, the fixtures are shape-accurate but not
magnitude-accurate, and a single simulated path's realised numbers move seed to
seed. So nothing below asserts that risk parity lost 18% in 2022. What it asserts
instead is that the *instruments* are correct:

* the diversification ratio against **closed-form values** -- for two equal-vol
  assets at correlation rho the ratio is exactly ``sqrt(2 / (1 + rho))``, which
  is a fact about the formula and not about any dataset;
* that the ratio is the diagnostic it is claimed to be, by constructing two
  covariances with the *same* correlation and different volatility dispersion and
  checking it separates them where a correlation cannot;
* that every rejection path rejects, since a NaN row in the crisis table reads as
  "the strategy was flat through the GFC";
* that the charts write real PNG bytes, which is steps 26 and 27's "done when".

The one place a realised magnitude is checked is the sign-free structural claim
that a strategy holding more capital in bonds shows more capital in bonds in the
table -- a tautology about the column being wired to the right frame, which is
exactly the bug that column could plausibly have.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, stress, viz  # noqa: E402
from src.benchmarks import equal_weight, naive_risk_parity, sixty_forty  # noqa: E402
from tests.fixtures import make_fake_result, make_fake_returns  # noqa: E402

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


def two_asset_cov(rho: float, vols: tuple[float, float] = (0.15, 0.15)) -> np.ndarray:
    """A 2x2 covariance with the given correlation and volatilities."""
    sd = np.array(vols, dtype="float64")
    corr = np.array([[1.0, rho], [rho, 1.0]])
    return corr * np.outer(sd, sd)


# ---------------------------------------------------------------------------
# Step 25 -- the diagnostic, against closed form
# ---------------------------------------------------------------------------
def test_diversification_ratio_closed_form() -> None:
    """The ratio's value is a fact about the formula, so it is checkable exactly.

    For two assets at equal volatility and weight, ``sigma_p = sigma *
    sqrt((1 + rho) / 2)`` while ``sum(w_i sigma_i) = sigma``, so the ratio is
    ``sqrt(2 / (1 + rho))`` regardless of the volatility level. Checking against
    that rather than against a stored number means this test still passes
    judgement if someone rewrites the implementation.
    """
    print("diversification ratio, closed form:")
    half = np.array([0.5, 0.5])

    for rho in (-0.9, -0.3, 0.0, 0.35, 0.9):
        expected = float(np.sqrt(2.0 / (1.0 + rho)))
        actual = stress.diversification_ratio(half, two_asset_cov(rho))
        check(
            f"rho = {rho:+.2f} -> {expected:.4f} (sqrt(2/(1+rho)))",
            abs(actual - expected) < 1e-12,
        )

    check(
        "perfectly correlated assets give exactly 1.0 -- no diversification at all",
        abs(stress.diversification_ratio(half, two_asset_cov(1.0)) - 1.0) < 1e-12,
    )
    check(
        "the ratio is scale-free: 10x the volatilities, same ratio",
        abs(
            stress.diversification_ratio(half, two_asset_cov(0.3, (0.05, 0.05)))
            - stress.diversification_ratio(half, two_asset_cov(0.3, (0.50, 0.50)))
        ) < 1e-12,
    )
    check(
        "a single asset cannot diversify against itself: ratio is 1.0",
        abs(stress.diversification_ratio(np.array([1.0]), np.array([[0.04]])) - 1.0) < 1e-12,
    )


def test_ratio_falls_as_correlation_rises() -> None:
    """Monotone in correlation, which is the property the 2022 claim rests on.

    The narrative is that correlation rising *toward zero* -- without changing
    sign -- was enough to remove the offset. That is only a coherent claim if the
    ratio responds continuously to correlation rather than to its sign, so this
    checks the whole path from -0.9 to +0.9 is strictly decreasing.
    """
    print("\nmonotonicity in correlation:")
    half = np.array([0.5, 0.5])
    rhos = np.linspace(-0.9, 0.9, 19)
    ratios = [stress.diversification_ratio(half, two_asset_cov(r)) for r in rhos]

    check(
        "strictly decreasing in rho across [-0.9, +0.9]",
        all(b < a for a, b in zip(ratios, ratios[1:])),
    )
    check(
        "moving from -0.34 to 0.00 costs real diversification with the sign unchanged",
        stress.diversification_ratio(half, two_asset_cov(-0.34))
        - stress.diversification_ratio(half, two_asset_cov(0.0)) > 0.1,
    )


def test_ratio_sees_what_correlation_cannot() -> None:
    """The diagnostic's whole justification, as an assertion.

    Two covariances with an *identical* SPY/TLT correlation but different
    volatility dispersion across the four sleeves. A pairwise correlation is by
    construction the same number in both; the ratio is not, because it aggregates
    the volatilities as well. If this ever fails, the ratio has stopped being an
    improvement on the correlation and step 25's premise is gone.
    """
    print("\nthe ratio separates what a correlation cannot:")
    corr = np.array(
        [
            [1.00, -0.30, 0.20, -0.10],
            [-0.30, 1.00, -0.05, 0.85],
            [0.20, -0.05, 1.00, 0.10],
            [-0.10, 0.85, 0.10, 1.00],
        ]
    )
    even = np.array([0.15, 0.15, 0.15, 0.15])
    dispersed = np.array([0.30, 0.06, 0.30, 0.06])
    equal = np.full(4, 0.25)

    ratio_even = stress.diversification_ratio(equal, corr * np.outer(even, even))
    ratio_dispersed = stress.diversification_ratio(equal, corr * np.outer(dispersed, dispersed))

    check(
        "same correlation matrix, different volatilities -> different ratio "
        f"({ratio_even:.3f} vs {ratio_dispersed:.3f})",
        abs(ratio_even - ratio_dispersed) > 0.01,
    )


def test_ratio_rejects_degenerate_input() -> None:
    print("\ndegenerate inputs raise rather than returning a number:")
    expect_raises(
        "a zero covariance has no portfolio volatility to divide by",
        stress.StressError,
        lambda: stress.diversification_ratio(np.array([0.5, 0.5]), np.zeros((2, 2))),
    )
    expect_raises(
        "a non-square covariance is rejected",
        stress.StressError,
        lambda: stress.diversification_ratio(np.array([0.5, 0.5]), np.zeros((2, 3))),
    )
    expect_raises(
        "a length mismatch between weights and covariance is rejected",
        stress.StressError,
        lambda: stress.diversification_ratio(np.array([0.3, 0.3, 0.4]), two_asset_cov(0.2)),
    )


def test_rolling_series() -> None:
    print("\nrolling series:")
    returns = make_fake_returns()

    correlation = stress.rolling_correlation(returns)
    check(
        f"correlation drops the incomplete leading window "
        f"({len(returns) - len(correlation)} days for a "
        f"{config.CORRELATION_WINDOW_DAYS}-day window)",
        len(returns) - len(correlation) == config.CORRELATION_WINDOW_DAYS - 1,
    )
    check(
        "every correlation lies in [-1, 1]",
        bool(correlation.between(-1.0, 1.0).all()),
    )

    ratio = stress.rolling_diversification_ratio(returns, step_days=21)
    check(
        "the diversification ratio never drops below 1.0 -- it cannot, by construction",
        bool((ratio.dropna() >= 1.0 - 1e-9).all()),
    )
    check(
        "annual_means gives one value per calendar year in the sample",
        list(stress.annual_means(ratio).index)
        == sorted(set(returns.index.year[config.DIVERSIFICATION_WINDOW_DAYS - 1 :])),
    )

    expect_raises(
        "a missing ticker in the correlation pair raises",
        stress.StressError,
        lambda: stress.rolling_correlation(returns, pair=("SPY", "NOPE")),
    )
    expect_raises(
        "a window longer than the sample raises rather than returning nothing",
        stress.StressError,
        lambda: stress.rolling_correlation(returns.iloc[:20]),
    )


def test_rolling_ratio_respects_the_lookahead_guard() -> None:
    """A future spike must not move a point that precedes it.

    The same attack ``tests/test_lookahead.py`` (step 28) makes on the engine,
    scoped to this module: the diagnostic slices through ``src.data.asof_slice``,
    so appending an enormous return *after* a date cannot change that date's
    value. If someone reimplements the windowing with a centred ``rolling``, this
    is what fails.
    """
    print("\nlook-ahead guard:")
    returns = make_fake_returns(end="2010-12-31")
    cut = returns.index[len(returns) // 2]

    baseline = stress.rolling_diversification_ratio(returns.loc[:cut], step_days=21)

    spiked = returns.copy()
    spiked.iloc[-5:] = spiked.iloc[-5:] + 0.75  # a violent, unmissable future move
    contaminated = stress.rolling_diversification_ratio(spiked, step_days=21).loc[:cut]

    common = baseline.index.intersection(contaminated.index)
    check(
        f"a +75% spike in the last week leaves all {len(common)} earlier points untouched",
        bool(np.allclose(baseline.loc[common], contaminated.loc[common], equal_nan=True)),
    )


# ---------------------------------------------------------------------------
# Step 24 -- the crisis table
# ---------------------------------------------------------------------------
def test_crisis_decomposition() -> None:
    print("\ncrisis decomposition:")
    returns = make_fake_returns()
    strategies = {
        "Risk parity (fixture)": make_fake_result(strategy="Risk parity (fixture)"),
        "Equal weight": equal_weight(returns),
        "60/40": sixty_forty(returns),
        "Naive 1/vol": naive_risk_parity(returns),
    }

    table = stress.crisis_decomposition(strategies)
    check(
        f"one row per (window, strategy): "
        f"{len(config.CRISIS_WINDOWS)} x {len(strategies)} = {len(table)}",
        len(table) == len(config.CRISIS_WINDOWS) * len(strategies),
    )
    check(
        "the bond-weight column is present and finite -- it is what explains the ranking",
        bool(np.isfinite(table["capital_weight_bonds"]).all()),
    )
    check(
        "every window is fully covered by the fixture's span",
        bool(table["fully_covered"].all()),
    )
    check(
        "drawdowns are non-positive everywhere",
        bool((table["max_drawdown"] <= 0.0).all()),
    )
    check(
        "unlevered benchmarks report a leverage of exactly 1.0",
        bool(
            np.allclose(
                table.loc[table["strategy"] != "Risk parity (fixture)", "mean_leverage"], 1.0
            )
        ),
    )

    # 60/40 holds 40% of its capital in TLT by definition, and every other
    # strategy here holds less. This is the column being wired to the right
    # frame, not a claim about any dataset.
    bonds = table.groupby("strategy")["capital_weight_bonds"].mean()
    check(
        f"60/40's bond weight is exactly 0.40 ({bonds['60/40']:.4f})",
        abs(bonds["60/40"] - 0.40) < 1e-9,
    )
    check(
        "equal weight's bond weight is exactly 0.25",
        abs(bonds["Equal weight"] - 0.25) < 1e-9,
    )
    check(
        "risk weight and capital weight differ for equal weight -- the whole thesis",
        abs(
            table.loc[table["strategy"] == "Equal weight", "risk_weight_bonds"].mean() - 0.25
        ) > 0.01,
    )

    pivot = stress.drawdown_table(strategies)
    check(
        "the pivot keeps the windows chronological rather than alphabetical",
        list(pivot.index) == list(config.CRISIS_WINDOWS),
    )
    check(
        "the pivot keeps the strategies in the order they were passed",
        list(pivot.columns) == list(strategies),
    )

    with tempfile.TemporaryDirectory() as directory:
        destination = stress.write_crisis_table(table, Path(directory) / "crisis.csv")
        check("write_crisis_table emits a non-empty CSV", destination.stat().st_size > 0)
        check(
            "the CSV round-trips with the same shape",
            pd.read_csv(destination).shape == table.shape,
        )


def test_crisis_decomposition_rejections() -> None:
    print("\ncrisis table rejection paths:")
    result = make_fake_result()

    expect_raises(
        "an unknown window name raises rather than emitting an empty row",
        KeyError,
        lambda: stress.crisis_decomposition(result, ["Taper tantrum"]),
    )
    expect_raises(
        "two results sharing a strategy name raise rather than overwriting each other",
        stress.StressError,
        lambda: stress.crisis_decomposition([result, make_fake_result(seed=1)]),
    )
    expect_raises(
        "an empty result set raises",
        stress.StressError,
        lambda: stress.crisis_decomposition({}),
    )
    expect_raises(
        "pivoting on a column that is not in the decomposition raises",
        stress.StressError,
        lambda: stress.drawdown_table(result, value="alpha"),
    )

    # A window entirely outside the data is a skipped row, and with only one
    # strategy that empties the table -- which must be an error, not an empty
    # frame that a downstream chart would render as a blank panel.
    expect_raises(
        "a window the data never reaches raises rather than returning nothing",
        stress.StressError,
        lambda: stress.crisis_decomposition(result, [("1975-01-01", "1975-12-31")]),
    )


def test_ad_hoc_windows() -> None:
    """``REGIME_SHIFT_WINDOW`` has an open end, and the table has to carry it."""
    print("\nad-hoc and open-ended windows:")
    result = make_fake_result()
    table = stress.crisis_decomposition(result, [config.REGIME_SHIFT_WINDOW])

    check("an open-ended window produces exactly one row", len(table) == 1)
    check(
        "and it is labelled by its dates rather than colliding with the strategy name",
        table.loc[0, "window"].startswith("2021-01-01 to open"),
    )
    check(
        "its coverage runs to the end of the sample",
        table.loc[0, "covered_end"] == result.returns.index[-1],
    )


def test_regime_summary() -> None:
    """The two diagnostics side by side -- the comparison step 25 exists to make."""
    print("\nregime summary:")
    returns = make_fake_returns()
    summary = stress.regime_summary(returns)

    check(
        "one row per crisis window",
        list(summary.index) == list(config.CRISIS_WINDOWS),
    )
    check(
        "fraction_positive is a proportion",
        bool(summary["fraction_positive"].between(0.0, 1.0).all()),
    )
    check(
        "the 2022 window carries both diagnostics, so they can be compared at all",
        bool(
            np.isfinite(summary.loc["Rate shock (2022)", "mean_correlation"])
            and np.isfinite(summary.loc["Rate shock (2022)", "mean_diversification_ratio"])
        ),
    )

    yearly = stress.correlation_regime_table(returns)
    check(
        "the per-year correlation table counts real trading days",
        bool((yearly["n_days"] > 0).all()),
    )


# ---------------------------------------------------------------------------
# Steps 26-27 -- the charts
# ---------------------------------------------------------------------------
def _is_png(path: Path) -> bool:
    """True if the file starts with the PNG magic number.

    Checked rather than trusting the extension: a matplotlib call that fails
    halfway can leave a truncated or empty file behind, and "the path exists" is
    not the same claim as "a chart was drawn".
    """
    return path.exists() and path.stat().st_size > 1024 and path.read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )


def test_charts_render() -> None:
    print("\ncharts:")
    returns = make_fake_returns()
    result = make_fake_result(strategy="Risk parity (fixture)")

    with tempfile.TemporaryDirectory() as directory:
        written = viz.render_all(result, returns, reports_dir=directory)

        check("render_all writes four charts", len(written) == 4)
        for name, path in written.items():
            check(f"{name}.png is real PNG data ({path.stat().st_size // 1024} KB)",
                  _is_png(path))

        # An unlevered benchmark is a legitimate thing to chart, and the leverage
        # panel must not refuse it -- a flat 1.0 line is the sanity check that
        # says the overlay is genuinely off.
        flat = viz.plot_leverage(
            make_fake_result(levered=False), Path(directory) / "flat.png"
        )
        check("the leverage chart accepts an unlevered result", _is_png(flat))

        # Re-rendering must overwrite in place rather than accumulate: step 30's
        # CLI is run repeatedly and reports/ is read by a human.
        again = viz.render_all(result, returns, reports_dir=directory)
        check(
            "re-rendering overwrites the same four paths",
            set(again.values()) == set(written.values()),
        )

    check(
        "colours are assigned per ticker, so a sleeve keeps its colour across figures",
        viz._colour_map(["SPY", "TLT", "DBC", "TIP"])["TLT"]
        == viz._colour_map(["SPY", "TLT"])["TLT"],
    )
    check(
        "and a strategy missing a sleeve does not inherit its colour",
        viz._colour_map(["SPY", "TLT"])["TLT"] != viz._colour_map(["TLT", "SPY"])["TLT"],
    )


def test_chart_rejections() -> None:
    print("\nchart rejection paths:")
    empty = pd.DataFrame(
        columns=list(config.UNIVERSE), index=pd.DatetimeIndex([], name="date")
    )
    result = make_fake_result()
    hollow = type(result)(
        weights=empty, risk_contributions=empty,
        leverage=pd.Series(dtype="float64", index=empty.index),
        returns=result.returns, meta={"strategy": "empty"},
    )

    expect_raises(
        "an empty risk-contribution frame raises rather than drawing a blank panel",
        ValueError,
        lambda: viz.plot_risk_contributions(hollow, "/dev/null"),
    )
    expect_raises(
        "an empty leverage series raises",
        ValueError,
        lambda: viz.plot_leverage(hollow, "/dev/null"),
    )


if __name__ == "__main__":
    test_diversification_ratio_closed_form()
    test_ratio_falls_as_correlation_rises()
    test_ratio_sees_what_correlation_cannot()
    test_ratio_rejects_degenerate_input()
    test_rolling_series()
    test_rolling_ratio_respects_the_lookahead_guard()
    test_crisis_decomposition()
    test_crisis_decomposition_rejections()
    test_ad_hoc_windows()
    test_regime_summary()
    test_charts_render()
    test_chart_rejections()
    print(f"\n{_checks} checks passed")
