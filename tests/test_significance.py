"""Self-checks for ``src.significance`` -- the interval on a gap between two ratios.

Runs standalone with no test framework::

    .venv/bin/python tests/test_significance.py

Entirely offline. ``src.significance`` holds pure functions of return series and
a seed, so every series below is built here from a fixed-seed generator.

**Nothing here asserts that one simulated series beat another.** A resampling
test is tempting to check by "the interval contained the truth", which is a
statement about one random path and would pass or fail by luck. The checks are
of two other kinds instead:

* **Exact identities** -- things that must hold on *every* resample whatever the
  data: a series against itself has a gap of exactly zero; a series against a
  multiple of itself has the same ratio on every draw, so the interval has no
  width at all (the direct test that the resampling is *paired*); swapping the
  two series negates the interval; a wider confidence level contains a narrower
  one.
* **Parameters of the method** -- the share of steps at which the stationary
  bootstrap starts a new block is ``1/mean_block``; the closed-form standard
  error matches arithmetic written out by hand; at a block length of one the
  bootstrap's standard error agrees with the closed form on independent data,
  and on autocorrelated data a long block gives a wider one, which is the whole
  reason for using blocks.

The pairing check is mutation-tested in place: the resampler is replaced by one
that draws each series' dates independently, and the check must then fail.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import significance  # noqa: E402
from src.config import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_MEAN_BLOCK_DAYS,
    RATIO_TEST_CONFIDENCE,
    TRADING_DAYS_PER_YEAR,
)
from src.metrics import sharpe_ratio  # noqa: E402
from src.significance import (  # noqa: E402
    SignificanceError,
    format_ratio_difference_table,
    memmel_test,
    ratio_difference_table,
    ratio_difference_test,
    stationary_bootstrap_indices,
)

_checks = 0

# Draws for the checks that only need the machinery to run. The identities hold
# on every resample, so they do not need many; the two standard-error
# comparisons say what they need where they are made.
_DRAWS = 2_000


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


def expect_raises(label: str, call) -> None:
    try:
        call()
    except SignificanceError:
        check(label, True)
        return
    raise AssertionError(f"FAILED: {label} -- no SignificanceError raised")


def _correlated_pair(n: int = 1500, rho: float = 0.8, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Two independent-over-time daily series with correlation ``rho`` between them."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, 0.01, n)
    own = rng.normal(0.0, 0.01, n)
    index = pd.bdate_range("2010-01-04", periods=n)
    a = pd.Series(0.0004 + shared, index=index)
    b = pd.Series(0.0002 + rho * shared + math.sqrt(1.0 - rho**2) * own, index=index)
    return a, b


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    shocks = rng.normal(0.0, 0.01, n)
    out = np.empty(n)
    out[0] = shocks[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


# ---------------------------------------------------------------------------
# The resampling scheme
# ---------------------------------------------------------------------------
def test_stationary_bootstrap_indices() -> None:
    print("stationary bootstrap indices:")
    n, draws = 2000, 200
    for mean_block in (1.0, 5.0, 20.0, 63.0):
        indices = stationary_bootstrap_indices(n, draws, mean_block, np.random.default_rng(1))
        if mean_block == 1.0:
            check("the index matrix is draws x observations", indices.shape == (draws, n))
            check("every index is a valid row", int(indices.min()) >= 0 and int(indices.max()) < n)

        # A step either continues to the next observation (wrapping at the end)
        # or starts a new block. A new block lands on "the next observation" by
        # chance 1/n of the time, so the share of visible jumps is
        # (1/L) * (1 - 1/n) -- a parameter of the scheme, not of the data.
        continued = (indices[:, 1:] - indices[:, :-1]) % n == 1
        jump_share = 1.0 - float(continued.mean())
        expected = (1.0 / mean_block) * (1.0 - 1.0 / n)
        # 398,000 steps: the binomial standard error is under 0.001 at every
        # block length here, so 0.004 is four of them.
        check(f"mean block {mean_block:g}: a new block starts on {jump_share:.4f} of steps "
              f"(expected {expected:.4f})",
              abs(jump_share - expected) < 0.004)

    first = stationary_bootstrap_indices(n, 4000, 20.0, np.random.default_rng(2))[:, 0]
    # A uniform start has mean (n-1)/2 and sd n/sqrt(12) ~ 577; over 4000 draws
    # the mean's standard error is ~9, so 40 is more than four of them.
    check("blocks start uniformly over the sample, not at its beginning",
          abs(float(first.mean()) - (n - 1) / 2.0) < 40.0)
    wrapped = stationary_bootstrap_indices(50, 200, 30.0, np.random.default_rng(3))
    check("a block that runs off the end wraps to the start",
          bool(((wrapped[:, :-1] == 49) & (wrapped[:, 1:] == 0)).any()))

    same = stationary_bootstrap_indices(n, 5, 20.0, np.random.default_rng(9))
    again = stationary_bootstrap_indices(n, 5, 20.0, np.random.default_rng(9))
    check("the same generator state gives the same resamples", np.array_equal(same, again))

    expect_raises("a mean block below one observation raises",
                  lambda: stationary_bootstrap_indices(n, 5, 0.5, np.random.default_rng(0)))
    expect_raises("zero draws raises",
                  lambda: stationary_bootstrap_indices(n, 0, 5.0, np.random.default_rng(0)))


# ---------------------------------------------------------------------------
# The closed-form test
# ---------------------------------------------------------------------------
def test_memmel_matches_arithmetic_done_by_hand() -> None:
    """Five observations, every moment written out.

    a = [0.01, 0.03, -0.01, 0.02, 0.00]   mean 0.01, sum of squared deviations
        0 + 0.0004 + 0.0004 + 0.0001 + 0.0001 = 0.0010, variance 0.00025
    b = [0.02, 0.01, 0.00, 0.03, -0.01]   mean 0.01, the same five values in
        another order, so variance 0.00025 too
    cross deviations: 0*0.01 + 0.02*0 + (-0.02)(-0.01) + 0.01*0.02 + (-0.01)(-0.02)
        = 0.0002 + 0.0002 + 0.0002 = 0.0006, covariance 0.00015, rho = 0.6
    s1 = s2 = 0.01 / sqrt(0.00025) = 0.632456
    V = (1/5) * [2 - 1.2 + 0.5 * (0.4 + 0.4 - 2 * 0.4 * 0.36)] = (1/5) * 1.056
    """
    print("Memmel / Jobson-Korkie, by hand:")
    index = pd.bdate_range("2020-01-06", periods=5)
    a = pd.Series([0.01, 0.03, -0.01, 0.02, 0.00], index=index)
    b = pd.Series([0.02, 0.01, 0.00, 0.03, -0.01], index=index)
    out = memmel_test(a, b, periods_per_year=1)

    s = 0.01 / math.sqrt(0.00025)
    variance = (2.0 - 2.0 * 0.6 + 0.5 * (s * s + s * s - 2.0 * s * s * 0.36)) / 5.0
    check("equal ratios give a difference of zero", abs(out["difference"]) < 1e-12)
    check("the standard error is sqrt(V) from the formula",
          abs(out["std_error"] - math.sqrt(variance)) < 1e-12)
    check("...which is sqrt(1.056 / 5)", abs(out["std_error"] - math.sqrt(1.056 / 5.0)) < 1e-12)
    check("a zero difference has z = 0 and p = 1",
          abs(out["z"]) < 1e-9 and abs(out["p_value"] - 1.0) < 1e-9)
    check("the interval is symmetric about the difference",
          abs((out["ci_low"] + out["ci_high"]) / 2.0 - out["difference"]) < 1e-12)

    annual = memmel_test(a, b, periods_per_year=4)
    check("annualising scales the standard error by sqrt(periods per year)",
          abs(annual["std_error"] - 2.0 * out["std_error"]) < 1e-12)

    a, b = _correlated_pair()
    forward, backward = memmel_test(a, b), memmel_test(b, a)
    check("swapping the series negates the difference and keeps its error",
          abs(forward["difference"] + backward["difference"]) < 1e-12
          and abs(forward["std_error"] - backward["std_error"]) < 1e-12)
    check("the difference is the difference of the project's own ratios",
          abs(forward["difference"]
              - (sharpe_ratio(a, TRADING_DAYS_PER_YEAR) - sharpe_ratio(b, TRADING_DAYS_PER_YEAR)))
          < 1e-12)

    # Four copies of the sample end to end: the same moments (to the ddof
    # correction), four times the observations, so half the standard error.
    def tiled(x: pd.Series) -> pd.Series:
        return pd.Series(np.tile(x.to_numpy(), 4), index=pd.bdate_range("2000-01-03", periods=4 * len(x)))

    longer = memmel_test(tiled(a), tiled(b))
    check("four times the observations halves the standard error",
          abs(longer["std_error"] / forward["std_error"] - 0.5) < 0.005)

    same = memmel_test(a, a)
    check("a series against itself: zero difference, zero error, no p-value to quote",
          same["difference"] == 0.0 and same["std_error"] == 0.0 and math.isnan(same["p_value"]))


# ---------------------------------------------------------------------------
# The bootstrap -- identities that hold on every resample
# ---------------------------------------------------------------------------
def _independent_columns(excess, periods_per_year, n_draws, mean_block, seed):
    """The mutation: each column resampled on its OWN dates. Pairing destroyed."""
    columns = [
        _REAL_BOOTSTRAP(excess[:, [k, k]], periods_per_year, n_draws, mean_block, seed + 1000 * k)[:, 0]
        for k in range(excess.shape[1])
    ]
    return np.column_stack(columns)


_REAL_BOOTSTRAP = significance._bootstrap_ratios


def test_bootstrap_identities() -> None:
    print("bootstrap, identities:")
    a, b = _correlated_pair()
    out = ratio_difference_test(a, b, n_draws=_DRAWS)

    point = sharpe_ratio(a, TRADING_DAYS_PER_YEAR) - sharpe_ratio(b, TRADING_DAYS_PER_YEAR)
    check("the point estimate is exactly the difference of the two reported ratios",
          abs(out["difference"] - point) < 1e-12
          and abs(out["ratio"] - sharpe_ratio(a, TRADING_DAYS_PER_YEAR)) < 1e-12)
    check("the interval is ordered and has width", out["ci_low"] < out["ci_high"])
    check("the row records the settings it was computed at",
          out["n_draws"] == _DRAWS and out["mean_block_days"] == BOOTSTRAP_MEAN_BLOCK_DAYS
          and out["confidence"] == RATIO_TEST_CONFIDENCE and out["n_observations"] == len(a))

    again = ratio_difference_test(a, b, n_draws=_DRAWS)
    check("a fixed seed reproduces the interval exactly",
          again["ci_low"] == out["ci_low"] and again["ci_high"] == out["ci_high"])
    other = ratio_difference_test(a, b, n_draws=_DRAWS, seed=12345)
    check("...and a different seed gives a different one, so the seed is really used",
          other["ci_low"] != out["ci_low"])

    itself = ratio_difference_test(a, a.copy(), n_draws=_DRAWS)
    check("a series against itself has a gap of exactly zero and an interval of [0, 0]",
          itself["difference"] == 0.0 and itself["ci_low"] == 0.0 and itself["ci_high"] == 0.0)

    # THE pairing check. 3*a has a's ratio on any set of dates, so if both
    # series are resampled on the same dates the gap is zero on every draw.
    scaled = ratio_difference_test(a, 3.0 * a, n_draws=_DRAWS)
    width = scaled["ci_high"] - scaled["ci_low"]
    check(f"a series against a multiple of itself: the interval has no width ({width:.1e}), "
          "so both series are resampled on the same dates",
          width < 1e-9)
    significance._bootstrap_ratios = _independent_columns
    try:
        broken = ratio_difference_test(a, 3.0 * a, n_draws=_DRAWS)
    finally:
        significance._bootstrap_ratios = _REAL_BOOTSTRAP
    broken_width = broken["ci_high"] - broken["ci_low"]
    check(f"...and resampling the two independently fails that check (width {broken_width:.2f}), "
          "so it can tell",
          broken_width > 0.1)

    rescaled = ratio_difference_test(5.0 * a, 0.2 * b, n_draws=_DRAWS)
    check("the ratio has no units: rescaling either series leaves the interval alone",
          abs(rescaled["ci_low"] - out["ci_low"]) < 1e-9
          and abs(rescaled["ci_high"] - out["ci_high"]) < 1e-9)

    swapped = ratio_difference_test(b, a, n_draws=_DRAWS)
    check("swapping the series negates the interval end for end",
          abs(swapped["ci_low"] + out["ci_high"]) < 1e-9
          and abs(swapped["ci_high"] + out["ci_low"]) < 1e-9)

    narrow = ratio_difference_test(a, b, n_draws=_DRAWS, confidence=0.80)
    wide = ratio_difference_test(a, b, n_draws=_DRAWS, confidence=0.99)
    check("a higher confidence level contains a lower one",
          wide["ci_low"] <= out["ci_low"] <= narrow["ci_low"]
          and narrow["ci_high"] <= out["ci_high"] <= wide["ci_high"])
    check("the p-value is a probability", 0.0 < out["p_value"] <= 1.0)

    # The p-value is the interval inverted, so they cannot disagree about zero.
    shifted = ratio_difference_test(a + 0.002, b, n_draws=_DRAWS)
    for label, row in (("a gap near zero", out), ("a large gap", shifted)):
        spans = row["ci_low"] <= 0.0 <= row["ci_high"]
        check(f"{label}: zero is {'inside' if spans else 'outside'} the 95% interval and "
              f"p is {'above' if spans else 'below'} 0.05, consistently",
              spans == (row["p_value"] > 1.0 - RATIO_TEST_CONFIDENCE))

    with_rate = ratio_difference_test(a, b, n_draws=_DRAWS, annual_risk_free_rate=0.03)
    check("a risk-free rate reaches the point estimate the way sharpe_ratio applies it",
          abs(with_rate["difference"]
              - (sharpe_ratio(a, TRADING_DAYS_PER_YEAR, 0.03)
                 - sharpe_ratio(b, TRADING_DAYS_PER_YEAR, 0.03))) < 1e-12)


def test_the_table_is_the_pair_test_row_by_row() -> None:
    print("the table:")
    a, b = _correlated_pair()
    rng = np.random.default_rng(11)
    c = pd.Series(0.0003 + rng.normal(0.0, 0.012, len(a)), index=a.index)
    frame = pd.DataFrame({"A": a, "bench": b, "C": c})
    table = ratio_difference_table(frame, "bench", n_draws=_DRAWS)

    check("one row per column other than the benchmark, in column order",
          list(table.index) == ["A", "C"])
    for name in ("A", "C"):
        pair = ratio_difference_test(frame[name], frame["bench"], n_draws=_DRAWS)
        check(f"row {name} equals the pair test at the same seed -- all rows share one "
              "set of resampled dates",
              all(abs(table.at[name, key] - pair[key]) < 1e-9
                  for key in ("difference", "ci_low", "ci_high", "std_error", "p_value")))
    check("every row carries the benchmark's own ratio",
          bool((table["benchmark_ratio"] == sharpe_ratio(b, TRADING_DAYS_PER_YEAR)).all()))
    check("the default number of draws is the configured one",
          int(ratio_difference_table(frame, "bench").iloc[0]["n_draws"]) == BOOTSTRAP_DRAWS)

    text = format_ratio_difference_table(table, "bench")
    row = table.loc["A"]
    check("the printed table puts the interval beside the point estimate",
          f"{row['difference']:+.3f}" in text
          and f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f}" in text)
    check("...names the confidence level and the benchmark in its headers",
          "95% interval" in text and "gap to bench" in text)

    # The spans-zero flag, on rows built by hand so each case is certain.
    crafted = table.copy()
    crafted.loc["A", ["difference", "ci_low", "ci_high"]] = [0.30, 0.10, 0.50]
    crafted.loc["C", ["difference", "ci_low", "ci_high"]] = [0.04, -0.12, 0.21]
    lines = format_ratio_difference_table(crafted, "bench").splitlines()
    line_a = next(line for line in lines if line.startswith("A"))
    line_c = next(line for line in lines if line.startswith("C"))
    check("an interval clear of zero is flagged NO", " NO " in line_a and " yes " not in line_a)
    check("an interval containing zero is flagged yes", " yes " in line_c and " NO " not in line_c)


# ---------------------------------------------------------------------------
# The bootstrap -- parameters of the method
# ---------------------------------------------------------------------------
def test_the_bootstrap_agrees_with_the_closed_form_where_it_should() -> None:
    print("bootstrap against the closed form:")
    # Independent normal returns: exactly the closed form's assumptions. At a
    # block length of one the bootstrap is the i.i.d. bootstrap, so the two are
    # estimating the same standard error two different ways. 4,000 draws puts
    # the bootstrap's own Monte Carlo error near 1%; 10% is the bar.
    a, b = _correlated_pair(n=3000, rho=0.8, seed=21)
    closed = memmel_test(a, b)["std_error"]
    iid = ratio_difference_test(a, b, n_draws=4000, mean_block_days=1.0)["std_error"]
    check(f"independent data, block of one: bootstrap error {iid:.4f} against closed form "
          f"{closed:.4f}",
          abs(iid / closed - 1.0) < 0.10)

    # The correlation between the books is what makes the gap estimable at all:
    # the closed form's leading term is 2(1 - rho)/T.
    loose_a, loose_b = _correlated_pair(n=3000, rho=0.0, seed=21)
    loose = ratio_difference_test(loose_a, loose_b, n_draws=_DRAWS, mean_block_days=1.0)["std_error"]
    check(f"two uncorrelated books have a much wider gap than two correlated ones "
          f"({loose:.3f} against {iid:.3f})",
          loose > 1.5 * iid)


def test_blocks_carry_serial_dependence() -> None:
    print("why blocks:")
    # Two independent AR(1) series at phi = 0.6. Autocorrelation inflates the
    # variance of a mean by (1 + phi)/(1 - phi) = 4, so the true standard error
    # of each ratio is about twice what an i.i.d. resample reports. A block
    # bootstrap has to see that; a block of one cannot.
    rng = np.random.default_rng(31)
    n = 4000
    index = pd.bdate_range("2005-01-03", periods=n)
    a = pd.Series(0.0003 + _ar1(n, 0.6, rng), index=index)
    b = pd.Series(0.0003 + _ar1(n, 0.6, rng), index=index)

    one = ratio_difference_test(a, b, n_draws=_DRAWS, mean_block_days=1.0)["std_error"]
    blocked = ratio_difference_test(a, b, n_draws=_DRAWS, mean_block_days=40.0)["std_error"]
    check(f"on autocorrelated returns a 40-day block gives a wider gap than a 1-day one "
          f"({blocked:.3f} against {one:.3f}, theory ~2x)",
          blocked > 1.4 * one)

    # And on data with no dependence, blocks should change little: they are a
    # correction for dependence, not a way of widening every interval.
    c, d = _correlated_pair(n=3000, rho=0.8, seed=41)
    flat_one = ratio_difference_test(c, d, n_draws=4000, mean_block_days=1.0)["std_error"]
    flat_blocked = ratio_difference_test(c, d, n_draws=4000, mean_block_days=40.0)["std_error"]
    check(f"on independent returns the block length barely matters "
          f"({flat_blocked:.4f} against {flat_one:.4f})",
          abs(flat_blocked / flat_one - 1.0) < 0.15)


# ---------------------------------------------------------------------------
# Inputs that cannot be tested
# ---------------------------------------------------------------------------
def test_rejections() -> None:
    print("rejections:")
    a, b = _correlated_pair(n=300)
    expect_raises("series on different dates raise -- a paired test needs one index",
                  lambda: ratio_difference_test(a, b.iloc[1:], n_draws=_DRAWS))
    holed = b.copy()
    holed.iloc[5] = np.nan
    expect_raises("a NaN raises rather than being dropped from one series only",
                  lambda: ratio_difference_test(a, holed, n_draws=_DRAWS))
    flat = pd.Series(0.001, index=a.index)
    expect_raises("a constant series has no ratio to compare",
                  lambda: ratio_difference_test(a, flat, n_draws=_DRAWS))
    expect_raises("...for the closed form too", lambda: memmel_test(a, flat))
    expect_raises("too few draws to resolve a tail raises",
                  lambda: ratio_difference_test(a, b, n_draws=50))
    expect_raises("a confidence of 1.0 raises",
                  lambda: ratio_difference_test(a, b, n_draws=_DRAWS, confidence=1.0))
    expect_raises("a mean block below one day raises",
                  lambda: ratio_difference_test(a, b, n_draws=_DRAWS, mean_block_days=0.5))
    expect_raises("two observations are not enough",
                  lambda: ratio_difference_test(a.iloc[:2], b.iloc[:2], n_draws=_DRAWS))
    expect_raises("a benchmark that is not a column raises",
                  lambda: ratio_difference_table(pd.DataFrame({"A": a, "B": b}), "missing",
                                                 n_draws=_DRAWS))
    expect_raises("bare arrays raise -- the index is what proves the series are paired",
                  lambda: ratio_difference_test(a.to_numpy(), b.to_numpy(), n_draws=_DRAWS))


if __name__ == "__main__":
    test_stationary_bootstrap_indices()
    test_memmel_matches_arithmetic_done_by_hand()
    test_bootstrap_identities()
    test_the_table_is_the_pair_test_row_by_row()
    test_the_bootstrap_agrees_with_the_closed_form_where_it_should()
    test_blocks_carry_serial_dependence()
    test_rejections()
    print(f"\n{_checks} checks passed")
