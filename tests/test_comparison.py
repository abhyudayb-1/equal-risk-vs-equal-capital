"""Self-checks for ``src.comparison`` -- equal risk against equal capital.

Runs standalone with no test framework::

    .venv/bin/python tests/test_comparison.py

Entirely offline. The comparison calls ``backtest.run_backtest`` twice and four
benchmarks once each, so this suite is the slowest in the repo -- it builds the
whole comparison three times (once on fixtures, once on a hand-built two-asset
case, once mutated) and caches the fixture one rather than rebuilding it per
test. Nothing fetches anything; ``tests/fixtures.py`` is the only data source.

**What is checked, and why each one is here:**

* **The effective-bets formula, at its limits and nowhere near its average.**
  Exactly ``N`` when every risk share is ``1/N`` and exactly ``1`` when one
  sleeve carries all of it, both to float precision rather than approximately.
  And deliberately **not** bounded inside ``[1, N]``: a sleeve that hedges the
  rest carries a *negative* risk share, the shares still sum to 1, and
  ``sum(RC**2)`` then exceeds 1. This happens on live data -- 60/40
  bottoms at 0.78 effective bets and max Sharpe at 0.88 on the live sample -- so
  a check asserting ``>= 1`` would be asserting something about the *rule*
  rather than about the formula, and would fail on real data while looking
  principled.

* **The matched column's construction, against a case whose answer is known
  before the engine runs.** Two assets, the second exactly ``k`` times the
  first. Two facts make the answer exact and path-independent: with two assets
  equal risk contribution means ``w_1 sigma_1 == w_2 sigma_2`` *whatever* the
  correlation (the cross terms cancel), and when ``B == k*A`` every portfolio of
  the two is a scalar multiple of one underlying series. So ERC must hold
  ``(k/(1+k), 1/(1+k))`` and the leverage that puts it at 1/N's volatility must
  be ``(1+k)**2 / (4k)`` -- 1.125 at ``k = 2`` -- on any return path, with no
  tolerance needed beyond float noise. If the matched target ever stops being
  that ratio, the construction is wrong rather than merely different.

* **The mutation.** The matched column is rebuilt with the second engine run
  swapped for the naive ``leverage = target / realised vol`` rescale someone
  would write instead, and the column has to *move*. That is the whole point of
  paying for a second engine run: ``backtest.py`` owns the one-day lag between a
  leverage decision and the return it earns, and a trailing estimator cannot hit
  its target, so the honest column overshoots while the naive one lands on the
  target exactly. A matched column that could not tell those two apart would not
  be testing the overlay at all. Its positive control is that the *other*
  columns are bit-identical across the mutation -- a mutation that moved
  everything would prove nothing.

* **The guards**, because every one of them exists to stop a comparison that
  looks fine: two rules measured on different calendars, a financing series with
  a gap in it, a benchmark with no volatility to match, and -- the one the whole
  construction rests on -- a second engine run that came back with *different
  ERC weights*, which would mean the target had leaked into the solve.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import comparison  # noqa: E402
from src.comparison import (  # noqa: E402
    ALLOCATION_RULES,
    COLUMN_ORDER,
    COUNTERFACTUAL,
    EQUAL_CAPITAL,
    ERC_MATCHED_FINANCED,
    ERC_MATCHED_FREE,
    ERC_TARGET_FINANCED,
    ERC_TARGET_FREE,
    ERC_UNLEVERED,
    MAX_SHARPE,
    NAIVE_INV_VOL,
    SIXTY_FORTY,
    ComparisonError,
    allocation_table,
    build_series,
    compare_rules,
    effective_bets,
    format_comparison_report,
    matched_target_vol,
)
from src.config import (  # noqa: E402
    COV_WARMUP_TRADING_DAYS,
    MIN_MATCHABLE_VOL,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from src.covariance import estimate_covariance  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.overlay import unlevered_returns  # noqa: E402
from tests.fixtures import make_fake_returns, make_fake_risk_free  # noqa: E402

_checks = 0
TOLERANCE = 1e-12

N = len(UNIVERSE)


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
    return abs(float(a) - float(b)) < tol


def _shares(*rows: list[float], columns: list[str] | None = None) -> pd.DataFrame:
    """A risk-contribution frame from literal rows -- no engine, no estimator."""
    cols = columns or list(UNIVERSE)[: len(rows[0])]
    return pd.DataFrame(list(rows), columns=cols)


def _annual_vol(series: pd.Series) -> float:
    return float(summarize(series, TRADING_DAYS_PER_YEAR)["annual_volatility"])


# ---------------------------------------------------------------------------
# The fixture comparison, built once
# ---------------------------------------------------------------------------
# Two engine runs and four benchmarks over nineteen fixture years is a few
# seconds; five tests want the same object. Cached rather than passed around as
# a parameter so each test still reads as a standalone function.
_FIXTURE_CACHE: dict[str, object] = {}


def fixture_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """The sleeve returns and financing rate every comparison here is built from."""
    if "returns" not in _FIXTURE_CACHE:
        returns = make_fake_returns(tickers=UNIVERSE)
        _FIXTURE_CACHE["returns"] = returns
        _FIXTURE_CACHE["risk_free"] = (
            make_fake_risk_free().reindex(returns.index).ffill().bfill()
        )
    return _FIXTURE_CACHE["returns"], _FIXTURE_CACHE["risk_free"]


def fixture_comparison():
    if "comp" not in _FIXTURE_CACHE:
        _FIXTURE_CACHE["comp"] = compare_rules(*fixture_inputs())
    return _FIXTURE_CACHE["comp"]


# ---------------------------------------------------------------------------
# effective_bets -- the formula, at its limits
# ---------------------------------------------------------------------------
def test_effective_bets_limiting_cases() -> None:
    equal = _shares([1.0 / N] * N)
    check(
        f"effective bets is exactly {N} when every risk share is 1/{N}",
        _close(effective_bets(equal).iloc[0], N),
    )

    one_sleeve = _shares([1.0] + [0.0] * (N - 1))
    check(
        "effective bets is exactly 1 when one sleeve carries all the risk",
        _close(effective_bets(one_sleeve).iloc[0], 1.0),
    )

    # Two equal halves out of four sleeves: 1 / (2 * 0.25) = 2 exactly.
    halves = _shares([0.5, 0.5, 0.0, 0.0])
    check(
        "two sleeves at half the risk each reads as exactly 2 bets",
        _close(effective_bets(halves).iloc[0], 2.0),
    )

    # The named case: a hedging sleeve's share is negative, the row
    # still sums to 1, and sum(RC**2) = 1.5**2 + 0.5**2 = 2.5 > 1.
    hedged = _shares([1.5, -0.5, 0.0, 0.0])
    check("a hedging sleeve's risk share is negative and the row still sums to 1",
          _close(hedged.iloc[0].sum(), 1.0))
    check(
        "...and effective bets then falls BELOW 1 -- the [1, N] bound is not "
        "guaranteed, so no check in this suite asserts it",
        _close(effective_bets(hedged).iloc[0], 1.0 / 2.5),
    )

    # Order cannot matter: the index is a sum over sleeves.
    scrambled = _shares([0.0, -0.5, 0.0, 1.5])
    check(
        "effective bets is invariant to which sleeve holds which share",
        _close(effective_bets(scrambled).iloc[0], effective_bets(hedged).iloc[0]),
    )

    # One value per rebalance, in order, named for the frame it joins.
    stacked = _shares([1.0 / N] * N, [1.0] + [0.0] * (N - 1))
    bets = effective_bets(stacked)
    check("one effective-bets value per rebalance row, in order",
          len(bets) == 2 and _close(bets.iloc[0], N) and _close(bets.iloc[1], 1.0))
    check("the series is named so it can be joined into a frame",
          bets.name == "effective_bets")

    expect_raises(
        "an empty risk-contribution frame raises rather than returning an empty answer",
        ComparisonError,
        lambda: effective_bets(pd.DataFrame()),
    )
    expect_raises(
        "a rebalance whose shares are all zero raises rather than dividing by it",
        ComparisonError,
        lambda: effective_bets(_shares([0.0, 0.0, 0.0, 0.0])),
    )
    expect_raises(
        "a non-frame raises rather than being coerced",
        ComparisonError,
        lambda: effective_bets(pd.Series([0.25, 0.25, 0.25, 0.25])),
    )


def test_effective_bets_matches_the_definition_on_random_rows() -> None:
    """``1 / sum(RC**2)`` computed a second way, on rows that sum to 1.

    Random shares rather than hand-built ones, because the hand-built cases
    above are all symmetric and a bug in the axis of the sum would survive every
    one of them.
    """
    rng = np.random.default_rng(11)
    rows = rng.normal(0.25, 0.3, size=(40, N))
    rows = rows / rows.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(rows, columns=list(UNIVERSE))

    check("random rows were normalised to sum to 1",
          float(np.abs(frame.sum(axis=1) - 1.0).max()) < 1e-12)

    expected = np.array([1.0 / float(np.sum(row**2)) for row in rows])
    check(
        "effective bets agrees with a row-by-row evaluation of the definition",
        float(np.abs(effective_bets(frame).to_numpy() - expected).max()) < 1e-12,
    )
    check(
        "...and at least one random row breached the usual [1, N] bound, which "
        "is why this suite never asserts it",
        bool((expected < 1.0).any() or (expected > N).any()),
    )


# ---------------------------------------------------------------------------
# matched_target_vol -- the number the second engine run is given
# ---------------------------------------------------------------------------
def test_matched_target_vol_is_the_benchmarks_own_volatility() -> None:
    index = pd.bdate_range("2015-01-05", periods=TRADING_DAYS_PER_YEAR * 2)
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0.0003, 0.008, len(index)), index=index)

    check(
        "the matched target is just the benchmark's own annualised volatility",
        _close(matched_target_vol(series), _annual_vol(series)),
    )

    # Scaling a book scales the target it would be matched to, exactly. This is
    # the property the two-asset case below turns into an exact answer.
    check(
        "doubling the benchmark doubles the target an ERC book is levered to",
        _close(matched_target_vol(2.0 * series), 2.0 * matched_target_vol(series), 1e-14),
    )

    expect_raises(
        "a benchmark that never moves has no target to match, and raises",
        ComparisonError,
        lambda: matched_target_vol(pd.Series(0.0, index=index)),
    )

    # The regression that found the guard. A *constant* series is just as
    # degenerate as a zero one, but the constant is not exactly representable in
    # binary, so its measured volatility is float noise rather than zero -- and
    # a bare `vol <= 0` test lets 1.7e-18 through, hands the overlay that as a
    # target, and gets a column back with the leverage pinned to its floor.
    nearly_flat = pd.Series(0.0004, index=index)
    measured = _annual_vol(nearly_flat)
    check("a constant return series measures a positive, finite volatility, "
          "not zero -- which is why the floor is not 0.0",
          0.0 < measured < MIN_MATCHABLE_VOL)
    expect_raises(
        "...and it is rejected against config.MIN_MATCHABLE_VOL rather than "
        "silently levered to 1.7e-18 volatility",
        ComparisonError,
        lambda: matched_target_vol(nearly_flat),
    )
    expect_raises(
        "a frame instead of a series raises rather than being summed over",
        ComparisonError,
        lambda: matched_target_vol(pd.DataFrame({"a": series})),
    )


# ---------------------------------------------------------------------------
# The identities, on a comparison the engine actually produced
# ---------------------------------------------------------------------------
def test_contract_identities_hold_across_every_rule() -> None:
    comp = fixture_comparison()

    check("every comparison column is present, in the report's order",
          list(comp.series.columns) == list(COLUMN_ORDER))
    check("the summary has exactly one row per column",
          list(comp.summary.index) == list(COLUMN_ORDER))
    check("the allocation block covers exactly the rules with distinct weights",
          sorted({rule for rule, _ in comp.allocation.columns}) == sorted(ALLOCATION_RULES))

    for rule in ALLOCATION_RULES:
        check(f"{rule}: mean capital weights sum to 1",
              _close(comp.allocation[(rule, "capital")].sum(), 1.0, 1e-9))
        check(f"{rule}: mean risk shares sum to 1",
              _close(comp.allocation[(rule, "risk")].sum(), 1.0, 1e-9))

    check("equal CAPITAL is flat for 1/N, to float precision",
          float((comp.allocation[(EQUAL_CAPITAL, "capital")] - 1.0 / N).abs().max()) < 1e-12)
    check("...and its RISK is not. That divergence is the whole argument",
          float((comp.allocation[(EQUAL_CAPITAL, "risk")] - 1.0 / N).abs().max()) > 0.01)
    check("equal RISK is flat for ERC, to the solver's own tolerance",
          float((comp.allocation[(ERC_UNLEVERED, "risk")] - 1.0 / N).abs().max()) < 1e-6)
    check("...and its CAPITAL is not",
          float((comp.allocation[(ERC_UNLEVERED, "capital")] - 1.0 / N).abs().max()) > 0.01)

    check(f"ERC takes {N}.00 effective bets at every single rebalance, by construction",
          float((comp.effective_bets[ERC_UNLEVERED] - N).abs().max()) < 1e-6)
    check("1/N takes fewer, which is the compactest form of the thesis",
          float(comp.effective_bets[EQUAL_CAPITAL].mean()) < N)

    check(
        "the concentration table reports mean(1 / sum RC^2), NOT "
        "1 / mean(sum RC^2) -- 1/x is convex and the second understates it",
        _close(
            comp.concentration.at[EQUAL_CAPITAL, "effective_bets_mean"],
            comp.effective_bets[EQUAL_CAPITAL].mean(),
        ),
    )
    squared = (comp.allocation[(EQUAL_CAPITAL, "risk")] ** 2).sum()
    check(
        "...and the two aggregations really do differ on this data, so the "
        "distinction is not academic",
        not _close(comp.concentration.at[EQUAL_CAPITAL, "effective_bets_mean"],
                   1.0 / squared, 1e-3),
    )

    check("the worst single-sleeve risk share 1/N reaches exceeds its 1/N capital",
          comp.concentration.at[EQUAL_CAPITAL, "worst_sleeve_risk_share"] > 1.0 / N)
    check("...and ERC's does not",
          comp.concentration.at[ERC_UNLEVERED, "worst_sleeve_risk_share"] < 1.0 / N + 1e-6)

    check("both financing-free rows are marked counterfactual, and only those",
          set(COUNTERFACTUAL) == {ERC_TARGET_FREE, ERC_MATCHED_FREE})
    check("financing is a cost, never a gain, at the published target",
          comp.summary.at[ERC_TARGET_FINANCED, "cagr"]
          <= comp.summary.at[ERC_TARGET_FREE, "cagr"])
    check("...and at 1/N's volatility",
          comp.summary.at[ERC_MATCHED_FINANCED, "cagr"]
          <= comp.summary.at[ERC_MATCHED_FREE, "cagr"])

    check("1/N is never measured against itself in the relative block",
          EQUAL_CAPITAL not in comp.relative.columns
          and len(comp.relative.columns) == len(COLUMN_ORDER) - 1)
    check("the matched column's realised volatility is reported beside its "
          "target rather than instead of it",
          comp.matched_realized_vol != comp.matched_target_vol)
    check("the target did not change the ERC weights -- the construction rests on this",
          comp.weights_unchanged)


def test_allocation_table_reads_both_frames_off_the_same_result() -> None:
    """``allocation_table`` must not recompute anything the rules already decided.

    And the per-rebalance identity the mean in that table would hide: every row
    of ``risk_contributions`` sums to 1, not merely every column mean. That is
    ``BacktestResult.validate()``'s job and ``compare_rules`` calls it on every
    result it builds, so this checks the thing the comparison relies on rather
    than re-deriving it.
    """
    comp = fixture_comparison()
    returns, _ = fixture_inputs()

    from src.benchmarks import equal_weight  # local: this is the only user

    rebuilt = equal_weight(returns, cov_estimator=estimate_covariance, strategy=EQUAL_CAPITAL)
    table = allocation_table({EQUAL_CAPITAL: rebuilt})
    check(
        "the capital column is the mean of the rule's own weights frame, "
        "not an estimate of it",
        float((table[(EQUAL_CAPITAL, "capital")] - rebuilt.weights.mean()).abs().max()) < 1e-15,
    )
    check(
        "the risk column is the mean of the rule's own risk_contributions frame",
        float(
            (table[(EQUAL_CAPITAL, "risk")] - rebuilt.risk_contributions.mean()).abs().max()
        ) < 1e-15,
    )
    check(
        "1/N's risk shares sum to 1 at EVERY rebalance, not just on average -- "
        "a column mean hides a row that does not",
        float((rebuilt.risk_contributions.sum(axis=1) - 1.0).abs().max()) < 1e-9,
    )
    check(
        "...and so do its capital weights, which is what makes the two columns "
        "of the allocation table comparable at all",
        float((rebuilt.weights.sum(axis=1) - 1.0).abs().max()) < 1e-12,
    )
    check("the comparison's own 1/N block is indexed by sleeve, in universe order",
          list(comp.allocation[(EQUAL_CAPITAL, "capital")].index) == list(UNIVERSE))


# ---------------------------------------------------------------------------
# The matched column, against an answer known before the engine runs
# ---------------------------------------------------------------------------
# Two assets, the second exactly K times the first. Both facts that make this
# exact are stated in the module docstring; the arithmetic:
#
#   ERC          w = (K/(1+K), 1/(1+K))      -> book = 2K/(1+K) * r
#   1/N          w = (1/2, 1/2)              -> book = (1+K)/2 * r
#   ratio of their volatilities = (1+K)^2 / (4K)   = 1.125 at K = 2
#
# Named SPY/TLT because ``compare_rules`` runs 60/40 as a reference row and
# ``benchmarks.sixty_forty`` rejects a universe without those two tickers. The
# identity under test involves neither of them.
K = 2.0
EXPECTED_MATCH_RATIO = (1.0 + K) ** 2 / (4.0 * K)


def two_asset_case():
    if "two_asset" not in _FIXTURE_CACHE:
        rng = np.random.default_rng(7)
        index = pd.bdate_range("2010-01-04", periods=1800, name="date")
        base = pd.Series(rng.normal(0.0002, 0.009, len(index)), index=index)
        frame = pd.DataFrame({"SPY": base, "TLT": K * base})
        risk_free = pd.Series(0.00002, index=index, name="risk_free")
        _FIXTURE_CACHE["two_asset"] = (frame, risk_free, compare_rules(frame, risk_free))
    return _FIXTURE_CACHE["two_asset"]


def test_matched_column_on_a_case_whose_answer_is_known_in_advance() -> None:
    frame, _, comp = two_asset_case()

    erc_capital = comp.allocation[(ERC_UNLEVERED, "capital")]
    check(
        f"with two assets, equal risk means w proportional to 1/sigma whatever the "
        f"correlation: ERC holds {K / (1 + K):.4f} / {1 / (1 + K):.4f}",
        _close(erc_capital["SPY"], K / (1.0 + K), 1e-9)
        and _close(erc_capital["TLT"], 1.0 / (1.0 + K), 1e-9),
    )
    check("...at every rebalance, not merely on average",
          float((comp.allocation[(ERC_UNLEVERED, "risk")] - 0.5).abs().max()) < 1e-6)

    equal_vol = _annual_vol(comp.series[EQUAL_CAPITAL])
    # Both books are HELD between rebalances, so the hand-derived ratio is exact
    # only where both sit at their targets: the first return day after each
    # rebalance. There, ERC earns 2K/(1+K)*r and 1/N earns (1+K)/2*r, so 1/N's
    # return is exactly (1+K)^2/(4K) times ERC's on every such day, whatever r is.
    days = comp.series.index
    rebalances = comp.rule_results[ERC_UNLEVERED].weights.index
    first_held = days[days.searchsorted(rebalances, side="right")
                      .clip(max=len(days) - 1)].unique()
    first_held = first_held[first_held > rebalances[0]]
    on_trade_days = (
        comp.series.loc[first_held, EQUAL_CAPITAL] / comp.series.loc[first_held, ERC_UNLEVERED]
    )
    check(
        f"on each of the {len(first_held)} days the books are fresh, 1/N's return is "
        f"exactly (1+k)^2/(4k) = {EXPECTED_MATCH_RATIO} times ERC's -- the "
        f"hand-derived answer",
        float((on_trade_days - EXPECTED_MATCH_RATIO).abs().max()) < 1e-9,
    )
    # Between rebalances the books drift, so the full-sample volatility ratio is
    # near 1.125 but not pinned to it; how near is a property of this path and
    # is deliberately not asserted.
    check("the matched target is 1/N's own realised volatility, not the published target",
          _close(comp.matched_target_vol, equal_vol) and comp.matched_target_vol != TARGET_VOL,
    )

    # The leverage the overlay chooses is target / trailing vol, and this series
    # is homoscedastic by construction, so the trailing estimate tracks the
    # full-sample one and the realised leverage lands near the hand-derived
    # ratio. Loose on purpose: a trailing estimator is never exact, and pinning
    # this tighter would be asserting a property of the random draw.
    mean_leverage = float(comp.meta["matched"]["mean_leverage"])
    check(
        f"the overlay levers the matched book to about {EXPECTED_MATCH_RATIO:.3f}x "
        f"(measured {mean_leverage:.4f}x) on a homoscedastic path",
        abs(mean_leverage - EXPECTED_MATCH_RATIO) < 0.05,
    )
    check("the published-target run levers to something different, so the two "
          "runs are genuinely distinct",
          not _close(float(comp.meta["engine"]["mean_leverage"]), mean_leverage, 1e-3))

    check("the same weights came back from both runs on this case too",
          comp.weights_unchanged)
    check("every sleeve of the two-asset book is in the series frame",
          list(frame.columns) == ["SPY", "TLT"])


# ---------------------------------------------------------------------------
# The mutation: swap the second engine run for a naive rescale
# ---------------------------------------------------------------------------
def _naive_rescale(result, sleeve_returns, risk_free, target_vol):
    """The matched column someone writes instead of running the engine again.

    One constant multiplier -- ``target / the book's own realised volatility`` --
    applied to the whole sample, with financing charged at that constant
    leverage. It is wrong in two ways the engine is not: it uses the realised
    volatility of the very window it is sizing (the look-ahead this repo has a
    whole suite about), and it has no decision lag at all.
    """
    index = result.returns.index
    unlevered = unlevered_returns(sleeve_returns.loc[index], result.weights)
    multiplier = target_vol / _annual_vol(unlevered)
    rate = risk_free.reindex(index)
    naive = multiplier * unlevered - (multiplier - 1.0) * rate
    return dataclasses.replace(
        result,
        leverage=pd.Series(multiplier, index=result.leverage.index, name="leverage"),
        returns=naive.rename("returns"),
    )


def test_the_matched_column_can_tell_the_engine_from_a_naive_rescale() -> None:
    honest = fixture_comparison()
    returns, risk_free = fixture_inputs()

    real_run_backtest = comparison.run_backtest

    def mutated_run_backtest(frame, **kwargs):
        result = real_run_backtest(frame, **kwargs)
        # Only the matched run is swapped: it is the one whose target is not the
        # published one. The published run stays honest so the comparison still
        # builds and the untouched columns act as the positive control.
        if not _close(kwargs["target_vol"], TARGET_VOL, 1e-15):
            result = _naive_rescale(result, frame, kwargs["risk_free"], kwargs["target_vol"])
        return result

    comparison.run_backtest = mutated_run_backtest
    try:
        mutant = compare_rules(returns, risk_free)
    finally:
        comparison.run_backtest = real_run_backtest

    check("the mutation left run_backtest restored",
          comparison.run_backtest is real_run_backtest)

    # The teeth. If these two columns agreed, the second engine run would be
    # buying nothing and this comparison would not be testing the overlay.
    moved = (mutant.series[ERC_MATCHED_FINANCED] - honest.series[ERC_MATCHED_FINANCED]).abs()
    check(
        "swapping the second engine run for a naive rescale MOVES the matched "
        "column -- the column is sensitive to how it was built",
        float(moved.max()) > 1e-4,
    )
    check("...on essentially every day, not at one boundary",
          float((moved > 1e-9).mean()) > 0.95)
    check(
        "...and it moves the headline the column exists to produce",
        abs(
            float(mutant.summary.at[ERC_MATCHED_FINANCED, "sharpe_ratio"])
            - float(honest.summary.at[ERC_MATCHED_FINANCED, "sharpe_ratio"])
        ) > 1e-4,
    )

    # The positive control. A mutation that moved every column would show only
    # that compare_rules is non-deterministic.
    for column in (ERC_UNLEVERED, EQUAL_CAPITAL, NAIVE_INV_VOL, SIXTY_FORTY,
                   MAX_SHARPE, ERC_TARGET_FINANCED):
        check(
            f"positive control: {column} is bit-identical across the mutation",
            float((mutant.series[column] - honest.series[column]).abs().max()) == 0.0,
        )

    # *How* they differ is the finding, not an artifact: the naive rescale is
    # sized on the volatility of the window it is sizing, so it hits the target
    # almost exactly. The engine cannot, because a trailing estimator is late.
    naive_gap = abs(mutant.matched_realized_vol / mutant.matched_target_vol - 1.0)
    honest_gap = abs(honest.matched_realized_vol / honest.matched_target_vol - 1.0)
    check(
        f"the naive rescale lands on its target ({naive_gap:.2%} off) because it "
        f"was sized on the window it is sizing",
        naive_gap < 0.01,
    )
    check(
        f"the engine's own overlay overshoots it ({honest_gap:.1%}) because a "
        f"trailing estimator is late -- which is why the report prints the "
        f"realised volatility and not the target",
        honest_gap > 5 * max(naive_gap, 1e-4),
    )


# ---------------------------------------------------------------------------
# The guards -- every one stops a comparison that would have looked fine
# ---------------------------------------------------------------------------
def test_build_series_rejects_inputs_that_cannot_be_compared() -> None:
    returns, risk_free = fixture_inputs()

    from src.backtest import run_backtest  # local: only this test builds one

    short = returns.iloc[: int(TRADING_DAYS_PER_YEAR * 4)]
    engine = run_backtest(short, risk_free=risk_free.reindex(short.index))
    other = run_backtest(
        short.iloc[:-30], risk_free=risk_free.reindex(short.index[:-30])
    )

    expect_raises(
        "two rules on different return calendars are rejected, not aligned -- a "
        "comparison across different windows is not a fair one",
        ComparisonError,
        lambda: build_series(short, engine, other, {}, risk_free),
    )

    gappy = risk_free.reindex(short.index).copy()
    gappy.iloc[-5] = np.nan
    expect_raises(
        "a financing series with a hole in it is rejected rather than "
        "forward-filled behind the caller's back",
        ComparisonError,
        lambda: build_series(short, engine, engine, {}, gappy),
    )


def test_compare_rules_rejects_bad_inputs() -> None:
    risk_free = pd.Series(0.0, index=pd.bdate_range("2015-01-05", periods=10))

    expect_raises(
        "a series instead of a frame of sleeve returns raises",
        ComparisonError,
        lambda: compare_rules(pd.Series([0.01, 0.02]), risk_free),
    )
    expect_raises(
        "an empty frame raises rather than producing an empty comparison",
        ComparisonError,
        lambda: compare_rules(pd.DataFrame(columns=list(UNIVERSE)), risk_free),
    )
    expect_raises(
        "a frame with a RangeIndex raises -- every window here is dated",
        ComparisonError,
        lambda: compare_rules(
            pd.DataFrame(np.zeros((10, N)), columns=list(UNIVERSE)), risk_free
        ),
    )


def test_a_non_default_warmup_moves_every_rule_together() -> None:
    # Regression: compare_rules handed warmup_trading_days to the engine and
    # built the four benchmarks at the config default, so any other value put
    # them on a different calendar and the run died with a bare KeyError.
    print("a non-default warm-up:")
    returns, risk_free = fixture_inputs()
    warmup = COV_WARMUP_TRADING_DAYS // 2
    comp = compare_rules(returns, risk_free, warmup_trading_days=warmup)

    check("the comparison builds at half the default warm-up",
          list(comp.series.columns) == list(COLUMN_ORDER))
    check("...and it really is a longer window than the default run's",
          comp.n_observations > fixture_comparison().n_observations)
    first = comp.rule_results[ERC_UNLEVERED].weights.index[0]
    check("every rule with its own weights starts on the engine's first rebalance",
          all(r.weights.index[0] == first for r in comp.rule_results.values()))
    check("no column carries a NaN from a rule that started late",
          not comp.series.isna().to_numpy().any())

    # Break it: a benchmark that ignores the warm-up it is given, which is what
    # every one of them effectively did. It must surface as a ComparisonError
    # naming the calendar, not as a KeyError from an index lookup.
    real = comparison.sixty_forty
    comparison.sixty_forty = lambda *args, warmup_trading_days=None, **kwargs: real(
        *args, **kwargs
    )
    try:
        expect_raises(
            "a rule left on the default warm-up is a ComparisonError, not a KeyError",
            ComparisonError,
            lambda: compare_rules(returns, risk_free, warmup_trading_days=warmup),
        )
    finally:
        comparison.sixty_forty = real


def test_every_ratio_gap_carries_its_interval() -> None:
    # The headline is a difference of two ratios. The comparison must carry an
    # interval on every such difference, computed on the same series as the
    # table of point estimates, and the report must print the two together.
    print("ratio gaps and their intervals:")
    comp = fixture_comparison()
    gaps = comp.ratio_difference

    check("one row per rule other than 1/N, in the report's order",
          list(gaps.index) == [c for c in COLUMN_ORDER if c != EQUAL_CAPITAL])
    one_n = comp.summary.at[EQUAL_CAPITAL, "sharpe_ratio"]
    check("each gap is exactly the difference of two cells of the summary table",
          all(abs(gaps.at[name, "difference"]
                  - (comp.summary.at[name, "sharpe_ratio"] - one_n)) < TOLERANCE
              for name in gaps.index))
    check("every interval contains its own point estimate's neighbourhood: low < high",
          bool((gaps["ci_low"] < gaps["ci_high"]).all()))
    check("the interval was measured on the comparison's own window",
          bool((gaps["n_observations"] == comp.n_observations).all()))

    text = format_comparison_report(comp)
    row = gaps.loc[ERC_UNLEVERED]
    interval = f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f}"
    check("the report prints ERC's interval beside its gap to 1/N",
          interval in text and f"{row['difference']:+.3f}" in text)
    check("...says how it was made", "block bootstrap" in text and "resamples" in text)

    spanning = int(((gaps["ci_low"] <= 0.0) & (gaps["ci_high"] >= 0.0)).sum())
    said = (
        "EVERY interval spans zero" if spanning == len(gaps)
        else "No interval spans zero" if spanning == 0
        else f"{spanning} of {len(gaps)} intervals span zero"
    )
    check(f"...and states in words how many intervals span zero ({spanning} of {len(gaps)})",
          said in text)

    # The words must follow the numbers. Push every interval clear of zero and
    # the sentence has to change with it.
    cleared = gaps.copy()
    cleared["ci_low"], cleared["ci_high"] = 0.01, 0.02
    check("with every interval clear of zero the report says so instead",
          "No interval spans zero" in format_comparison_report(
              dataclasses.replace(comp, ratio_difference=cleared)))
    straddling = gaps.copy()
    straddling["ci_low"], straddling["ci_high"] = -0.1, 0.1
    check("with every interval across zero it warns that no ratio is distinguishable",
          "EVERY interval spans zero" in format_comparison_report(
              dataclasses.replace(comp, ratio_difference=straddling)))


# ---------------------------------------------------------------------------
# The printed report -- what a reader actually sees
# ---------------------------------------------------------------------------
def test_report_states_what_the_table_cannot_say_for_itself() -> None:
    comp = fixture_comparison()
    text = format_comparison_report(comp)

    check("the report names the window and the data date it was measured on",
          comp.data_through in text and comp.start in text)
    check("it prints the matched column's REALISED volatility, not only the target",
          f"{comp.matched_realized_vol:.2%}" in text)
    check("...next to the overshoot, as a signed percentage",
          "overshoot" in text)
    check("the matched column is labelled a hindsight diagnostic wherever it appears, "
          "because its target is 1/N's full-window volatility",
          "(hindsight)" in ERC_MATCHED_FINANCED and "(hindsight)" in ERC_MATCHED_FREE
          and "HINDSIGHT diagnostic" in text)
    check("effective bets is described as concentration, not as independent bets",
          "not how many independent" in text)
    check("it names both counterfactual rows where a reader meets them",
          "counterfactual" in text
          and ERC_MATCHED_FREE in text and ERC_TARGET_FREE in text)
    check("it says costs are not modelled and which rule that favours",
          "transaction costs" in text and "ERC" in text)
    check("it says the ratio is return-to-volatility, not a true Sharpe",
          "not a true Sharpe" in text)
    check("it reports the engine's run health, so a bad run cannot read as a good one",
          "engine health" in text)
    check("every comparison row appears in the printed tables",
          all(column in text for column in COLUMN_ORDER))
    check("the crisis block names the rule it compares against",
          "Crisis windows" in text and EQUAL_CAPITAL in text)


if __name__ == "__main__":
    # The benchmarks log a line per rebalance where no tangency portfolio
    # exists, and the two-asset case is perfectly correlated by construction so
    # it hits that on half of them. Nothing here is testing max Sharpe.
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    test_effective_bets_limiting_cases()
    test_effective_bets_matches_the_definition_on_random_rows()

    test_matched_target_vol_is_the_benchmarks_own_volatility()

    test_contract_identities_hold_across_every_rule()
    test_allocation_table_reads_both_frames_off_the_same_result()

    test_matched_column_on_a_case_whose_answer_is_known_in_advance()
    test_the_matched_column_can_tell_the_engine_from_a_naive_rescale()

    test_build_series_rejects_inputs_that_cannot_be_compared()
    test_compare_rules_rejects_bad_inputs()
    test_a_non_default_warmup_moves_every_rule_together()
    test_every_ratio_gap_carries_its_interval()

    test_report_states_what_the_table_cannot_say_for_itself()

    print(f"\n{_checks} checks passed")
