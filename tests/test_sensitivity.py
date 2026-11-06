"""Self-checks for ``src.sensitivity`` -- turnover, cost and the estimator sweep.

Runs standalone with no test framework::

    .venv/bin/python tests/test_sensitivity.py

Entirely offline. The sweep calls the engine, so this suite builds its own small
synthetic universe rather than the nineteen-year fixture: four sleeves and about
five years, which is enough for a 252-day warm-up plus forty-odd rebalances and
fast enough that the whole file runs in seconds.

**What is checked, and why each one is here:**

* **Turnover, against arithmetic done by hand.** Two sleeves at 50/50, one of
  them up 10% over the month: the book has drifted to 0.55/1.05 and 0.50/1.05,
  so trading it back to 50/50 moves 2.38% of it. Every drift and turnover figure
  in the report is that calculation repeated, and it is the one number here that
  nothing else in the repo can cross-check -- ``benchmarks.fixed_weight`` holds
  its weights constant, so a 1/N book's *weights frame* says it trades nothing,
  which is exactly the wrong answer and exactly why this is reconstructed.

* **That the first rebalance is excluded.** Buying the initial allocation moves
  100% of a book and is not a rebalance; counted, it would put a one-off at the
  top of every turnover average and make a long sample look cheaper than a short
  one for no reason.

* **Costs land on the right day and only ever subtract.** The charge for a trade
  at the close of ``d`` hits the first return day at or after ``d`` -- the first
  period the new book is held -- which is the same alignment the engine accrues
  on. Checked by placing one trade and finding the single day that moved.

* **The breakeven is a crossing, not a report of one.** It is verified from both
  sides: ERC leads just below it and does not just above. Plus the two degenerate
  answers, which have to stay distinguishable -- ``nan`` when ERC never led (so
  there is nothing for a cost to take away) and ``inf`` when it still leads past
  the search cap (so the honest answer is "further than we looked"), because
  collapsing either into a number would invent a measurement.

* **The sweep moves one knob and keeps both rules together.** A sweep that
  changed the engine's warm-up and left 1/N's alone would compare two rules over
  two different windows and report the difference as a sensitivity. The default
  row is also re-run rather than copied, so a knob that leaked into another
  would show up as two "default" rows that disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src import sensitivity  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.benchmarks import equal_weight  # noqa: E402
from src.config import (  # noqa: E402
    COV_EWMA_HALFLIFE_DAYS,
    COV_SHRINKAGE,
    COV_WARMUP_TRADING_DAYS,
    MAX_BREAKEVEN_COST,
    TRADING_DAYS_PER_YEAR,
)
from src.sensitivity import (  # noqa: E402
    KNOBS,
    CostComparison,
    SensitivityError,
    charge_costs,
    cost_comparison,
    drifted_weights,
    format_sensitivity_report,
    losing_settings,
    run_setting,
    sweep,
    turnover,
    turnover_table,
)

_checks = 0
TOLERANCE = 1e-12

TICKERS = ("AAA", "BBB", "CCC", "DDD")


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


# ---------------------------------------------------------------------------
# A small universe, built once
# ---------------------------------------------------------------------------
_CACHE: dict[str, object] = {}


def small_universe() -> tuple[pd.DataFrame, pd.Series]:
    """Four sleeves of differing volatility over about five years, seeded.

    Correlated through one common factor so the covariance has structure for the
    ERC solver to find; otherwise equal risk and inverse volatility would
    coincide and the sweep would have nothing to measure.
    """
    if "returns" not in _CACHE:
        rng = np.random.default_rng(19)
        index = pd.bdate_range("2012-01-02", periods=1300, name="date")
        factor = rng.normal(0.0, 0.007, len(index))
        vols = (0.010, 0.006, 0.013, 0.003)
        loadings = (0.9, -0.3, 0.5, 0.2)
        columns = {
            ticker: loading * factor + rng.normal(0.0002, vol, len(index))
            for ticker, vol, loading in zip(TICKERS, vols, loadings)
        }
        _CACHE["returns"] = pd.DataFrame(columns, index=index)
        _CACHE["risk_free"] = pd.Series(0.00006, index=index, name="risk_free")
    return _CACHE["returns"], _CACHE["risk_free"]


def small_results():
    """One ERC run and one 1/N run over that universe, cached."""
    if "erc" not in _CACHE:
        returns, risk_free = small_universe()
        _CACHE["erc"] = run_backtest(returns, risk_free=risk_free)
        _CACHE["one_n"] = equal_weight(returns, strategy="1/N")
    return _CACHE["erc"], _CACHE["one_n"]


# ---------------------------------------------------------------------------
# Turnover, by hand
# ---------------------------------------------------------------------------
def test_drift_and_turnover_by_hand() -> None:
    # Two rebalance dates a month apart. AAA compounds to +10% over the second
    # month, BBB to 0%. Both are held at 50/50 at both dates.
    index = pd.bdate_range("2020-01-01", periods=44, name="date")
    rebalances = pd.DatetimeIndex([index[0], index[22]])
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]], index=rebalances, columns=["AAA", "BBB"]
    )
    sleeves = pd.DataFrame(0.0, index=index, columns=["AAA", "BBB"])
    # One +10% day, strictly after the first rebalance and on or before the
    # second, so the whole move belongs to the drift window.
    sleeves.loc[index[10], "AAA"] = 0.10

    drifted = drifted_weights(weights, sleeves)
    check("the first rebalance has no drifted book -- there is no previous one",
          bool(drifted.iloc[0].isna().all()))
    expected_aaa = 0.55 / 1.05
    check(f"a 50/50 book with one sleeve up 10% has drifted to "
          f"{expected_aaa:.4f} / {1 - expected_aaa:.4f}",
          _close(drifted.iloc[1]["AAA"], expected_aaa)
          and _close(drifted.iloc[1]["BBB"], 1.0 - expected_aaa))
    check("the drifted book still sums to 1",
          _close(drifted.iloc[1].sum(), 1.0))

    traded = turnover(weights, sleeves)
    expected_trade = expected_aaa - 0.5  # one-way: half the sum of |dw|
    check(f"trading it back to 50/50 moves {expected_trade:.4%} of the book",
          _close(traded["total"].iloc[1], expected_trade))
    check("all of which is drift-back, because the target never moved",
          _close(traded["drift_back"].iloc[1], expected_trade))
    check("and none of which is re-optimisation, for the same reason",
          _close(traded["reoptimisation"].iloc[1], 0.0))
    check("the first rebalance is NaN in every column, not zero -- buying the "
          "initial book is a one-off, not a rebalance",
          bool(traded.iloc[0].isna().all()))

    # A return that lands ON the previous rebalance date must not count: a book
    # chosen at that close cannot have earned that day's move.
    on_the_date = pd.DataFrame(0.0, index=index, columns=["AAA", "BBB"])
    on_the_date.loc[index[0], "AAA"] = 0.10
    check("a move on the previous rebalance date itself does not drift the book",
          _close(drifted_weights(weights, on_the_date).iloc[1]["AAA"], 0.5))

    # And one landing exactly on the new rebalance date must count: the book is
    # traded at that close, after the day's move.
    on_the_new_date = pd.DataFrame(0.0, index=index, columns=["AAA", "BBB"])
    on_the_new_date.loc[index[22], "AAA"] = 0.10
    check("a move on the new rebalance date does drift it, because the trade "
          "happens at that close",
          _close(drifted_weights(weights, on_the_new_date).iloc[1]["AAA"], expected_aaa))


def test_turnover_rejects_what_it_cannot_measure() -> None:
    index = pd.bdate_range("2020-01-01", periods=10)
    weights = pd.DataFrame([[0.5, 0.5]], index=index[:1], columns=["AAA", "BBB"])

    expect_raises(
        "an empty weights frame raises rather than returning no turnover",
        SensitivityError,
        lambda: drifted_weights(pd.DataFrame(), pd.DataFrame()),
    )
    expect_raises(
        "a sleeve with no return column raises rather than being dropped",
        SensitivityError,
        lambda: drifted_weights(
            weights, pd.DataFrame(0.0, index=index, columns=["AAA"])
        ),
    )

    # A sleeve that loses everything takes the book to zero, and renormalising
    # by zero would hand back NaN weights that look like a missing estimate.
    wiped = pd.DataFrame(0.0, index=index, columns=["AAA", "BBB"])
    wiped.loc[index[5]] = -1.0
    two = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]], index=index[[0, 9]], columns=["AAA", "BBB"]
    )
    expect_raises(
        "a book that drifts to zero raises rather than renormalising by zero",
        SensitivityError,
        lambda: drifted_weights(two, wiped),
    )


def test_turnover_on_the_real_rules() -> None:
    returns, _ = small_universe()
    erc, one_n = small_results()

    erc_traded = turnover(erc.weights, returns)
    one_n_traded = turnover(one_n.weights, returns)

    check("1/N re-optimises exactly nothing, to the bit",
          float(one_n_traded["reoptimisation"].abs().max()) == 0.0)
    check("...and yet it trades, because the book drifts off 1/N between "
          "rebalances -- the claim 'equal capital barely trades' is about the "
          "rule, not about zero",
          float(one_n_traded["drift_back"].mean()) > 0.0)
    check("ERC re-optimises, so it trades more",
          float(erc_traded["total"].mean()) > float(one_n_traded["total"].mean()))
    check(
        "a trade is never MORE than drift-back plus re-optimisation: the two "
        "partly cancel, so the real trade is at most their sum",
        bool(
            (
                erc_traded["total"].dropna()
                <= (erc_traded["drift_back"] + erc_traded["reoptimisation"]).dropna() + 1e-12
            ).all()
        ),
    )
    check("...and they really do cancel sometimes, rather than the bound being "
          "attained everywhere",
          bool(
              (
                  erc_traded["total"].dropna()
                  < (erc_traded["drift_back"] + erc_traded["reoptimisation"]).dropna() - 1e-9
              ).any()
          ))

    table = turnover_table({"ERC": erc, "1/N": one_n}, returns)
    years = len(erc.returns) / TRADING_DAYS_PER_YEAR
    check("annual turnover is the total traded divided by the sample's years, "
          "so two rebalance frequencies are comparable",
          _close(table.at["ERC", "annual_turnover"],
                 erc_traded["total"].sum() / years, 1e-9))
    check("the table reports the allocation's turnover and does not fold "
          "leverage into it",
          "mean_leverage" not in table.columns)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def test_costs_land_on_the_right_day_and_only_subtract() -> None:
    index = pd.bdate_range("2020-01-01", periods=20, name="date")
    returns = pd.Series(0.001, index=index)
    traded = pd.Series([0.0, 0.25], index=index[[0, 9]])

    check("a zero cost returns the series bit for bit",
          charge_costs(returns, traded, 0.0).equals(returns))

    charged = charge_costs(returns, traded, 0.004)
    moved = (charged - returns).abs() > 0
    check("one trade moves exactly one day", int(moved.sum()) == 1)
    check("...the rebalance day itself, which is the first period the new "
          "book is held",
          charged.index[moved.to_numpy()][0] == index[9])
    check("and it subtracts rate * traded fraction, exactly",
          _close(returns.iloc[9] - charged.iloc[9], 0.25 * 0.004))

    levered = charge_costs(returns, traded, 0.004, leverage=pd.Series(2.0, index=traded.index))
    check("leverage scales the charge: a 2x book trades twice the notional",
          _close(returns.iloc[9] - levered.iloc[9], 2.0 * 0.25 * 0.004))

    # A rebalance date between two return days charges the next one; a rebalance
    # after the last return day has nothing to charge and is dropped.
    off_calendar = pd.Series([0.25], index=[index[9] + pd.Timedelta(hours=5)])
    check("a trade between two return days is charged to the next one",
          _close(
              returns.iloc[10] - charge_costs(returns, off_calendar, 0.004).iloc[10],
              0.25 * 0.004,
          ))
    after_the_end = pd.Series([0.25], index=[index[-1] + pd.Timedelta(days=40)])
    check("a trade after the last return day is dropped, not charged to the last one",
          charge_costs(returns, after_the_end, 0.004).equals(returns))

    expect_raises(
        "a negative cost raises -- a trade is never a rebate",
        SensitivityError,
        lambda: charge_costs(returns, traded, -0.001),
    )


def test_breakeven_is_a_crossing_verified_from_both_sides() -> None:
    returns, _ = small_universe()
    erc, one_n = small_results()
    index = erc.returns.index

    from src.overlay import unlevered_returns

    erc_unlevered = unlevered_returns(returns.loc[index], erc.weights)
    erc_traded = turnover(erc.weights, returns)["total"]
    one_n_traded = turnover(one_n.weights, returns)["total"]

    comparison = cost_comparison(
        erc_unlevered, erc_traded, one_n.returns, one_n_traded
    )
    check("the cost grid starts at zero cost, which is the headline",
          comparison.table.index[0] == 0.0)
    check("the gap narrows as the cost rises, because ERC trades more",
          comparison.table["gap"].iloc[-1] < comparison.table["gap"].iloc[0])
    check("an unlevered comparison records that it was not levered",
          comparison.levered is False)

    breakeven = comparison.breakeven_cost
    if np.isfinite(breakeven):
        below = cost_comparison(
            erc_unlevered, erc_traded, one_n.returns, one_n_traded,
            cost_rates_bps=(breakeven * 1e4 * 0.98,),
        )
        above = cost_comparison(
            erc_unlevered, erc_traded, one_n.returns, one_n_traded,
            cost_rates_bps=(breakeven * 1e4 * 1.02,),
        )
        check(f"ERC still leads just below the {breakeven * 1e4:.1f}bp breakeven",
              float(below.table["gap"].iloc[0]) > 0)
        check("...and does not, just above it",
              float(above.table["gap"].iloc[0]) <= 0)
    else:
        check("a breakeven past the search cap is reported as inf, not as the cap",
              np.isinf(breakeven))

    # The two degenerate answers have to stay distinguishable from each other
    # and from a number. Built by hand, so neither depends on the sample.
    shocks = np.random.default_rng(3).normal(0.0, 0.004, len(erc_unlevered))
    worse = pd.Series(0.0001 + shocks, index=erc_unlevered.index)
    better = pd.Series(0.0008 + shocks, index=erc_unlevered.index)
    never_leads = cost_comparison(worse, erc_traded, better, one_n_traded)
    check("two series differing only in drift put the lower one behind at zero cost",
          float(never_leads.table["gap"].iloc[0]) < 0)
    check("...so there is NO breakeven to find, and it is reported as nan "
          "rather than as zero, which would read as 'any cost at all'",
          np.isnan(never_leads.breakeven_cost))

    no_trading = pd.Series(0.0, index=erc_traded.index)
    always_leads = cost_comparison(
        erc_unlevered, no_trading, one_n.returns, no_trading,
    )
    check(
        f"a rule that leads with no trading at all still leads at "
        f"{MAX_BREAKEVEN_COST * 1e4:.0f}bp, reported as inf rather than as a "
        "number nobody measured",
        np.isinf(always_leads.breakeven_cost),
    )

    expect_raises(
        "an empty cost grid raises rather than sweeping nothing",
        SensitivityError,
        lambda: cost_comparison(
            erc_unlevered, erc_traded, one_n.returns, one_n_traded, cost_rates_bps=()
        ),
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def test_a_setting_runs_both_rules_on_one_window() -> None:
    returns, risk_free = small_universe()

    default = run_setting(returns, risk_free)
    check("a setting reports both rules' ratios and the gap between them",
          _close(default["gap"], default["erc_ratio"] - default["one_n_ratio"]))
    check("...and the matched column's gap against the same benchmark",
          _close(default["matched_gap"], default["matched_ratio"] - default["one_n_ratio"]))
    check(f"ERC takes exactly {len(TICKERS)}.00 effective bets -- that is the "
          "rule, not the estimator",
          _close(default["erc_bets"], len(TICKERS), 1e-6))
    check("1/N takes fewer, which is what the whole comparison is about",
          default["one_n_bets"] < len(TICKERS))
    check("the row carries its own observation count",
          default["days"] > 0 and default["rebalances"] > 0)

    # Passing the defaults explicitly must reproduce the defaults exactly: if it
    # did not, a knob would be leaking through run_backtest's signature.
    explicit = run_setting(
        returns,
        risk_free,
        cov_halflife=COV_EWMA_HALFLIFE_DAYS,
        shrinkage=COV_SHRINKAGE,
        warmup_trading_days=COV_WARMUP_TRADING_DAYS,
        rebalance_months=1,
    )
    check("passing every knob at its config.py value reproduces the default run",
          all(_close(explicit[key], default[key], 1e-12)
              for key in ("erc_ratio", "one_n_ratio", "matched_ratio", "erc_bets")))

    # A longer warm-up has to move BOTH rules, or the sweep would be comparing
    # two rules over two windows and calling the difference a sensitivity.
    longer = run_setting(returns, risk_free, warmup_trading_days=400)
    check("a longer warm-up shortens the sample",
          longer["days"] < default["days"])
    check("...and moves 1/N too, so both rules are measured on the same window",
          longer["one_n_ratio"] != default["one_n_ratio"])

    sparser = run_setting(returns, risk_free, rebalance_months=6)
    check("a sparser calendar means fewer rebalances on the same sample",
          sparser["rebalances"] < default["rebalances"]
          and sparser["days"] == default["days"])


def test_sweep_moves_one_knob_at_a_time() -> None:
    returns, risk_free = small_universe()
    knobs = (
        ("cov_halflife", "covariance halflife (days)", (30, 60), COV_EWMA_HALFLIFE_DAYS),
        ("rebalance_months", "rebalance every N months", (1, 6), 1),
    )
    swept = sweep(returns, risk_free, knobs=knobs)

    check("one row per setting, across every knob",
          len(swept) == sum(len(grid) for _, _, grid, _ in knobs))
    check("each knob is named once per block",
          sorted(swept["knob"].unique()) == sorted({label for _, label, _, _ in knobs}))
    check("the default is marked exactly once per knob",
          int(swept["is_default"].sum()) == len(knobs))

    defaults = swept[swept["is_default"]]
    check(
        "every knob's default row is the SAME run -- two that disagreed would "
        "mean a knob is leaking into another",
        float(defaults["erc_ratio"].std()) < 1e-12
        and float(defaults["one_n_ratio"].std()) < 1e-12,
    )

    check("losing_settings reads the unlevered gap by default",
          losing_settings(swept).equals(swept[swept["gap"] <= 0]))
    check("...and the matched gap when asked",
          losing_settings(swept, "matched_gap").equals(swept[swept["matched_gap"] <= 0]))
    expect_raises(
        "an unknown gap column raises rather than returning an empty frame",
        SensitivityError,
        lambda: losing_settings(swept, "no_such_column"),
    )

    erc, one_n = small_results()
    traded = turnover_table({"ERC": erc, "1/N": one_n}, returns)
    costs = cost_comparison(
        erc.returns,
        turnover(erc.weights, returns)["total"],
        one_n.returns,
        turnover(one_n.weights, returns)["total"],
    )
    text = format_sensitivity_report(swept, traded, costs)
    check("the report reads the two gap columns apart, so the known headline "
          "does not drown the thing the sweep found",
          "Unlevered" in text and "Levered to 1/N's volatility" in text)
    check("it names every knob it swept",
          all(label in text for _, label, _, _ in knobs))
    check("it marks the default setting where a reader meets it",
          "(default)" in text)
    check("it says turnover is reconstructed rather than read off the weights",
          "RECONSTRUCTED" in text)
    check("a single CostComparison is accepted as well as a list of them",
          isinstance(costs, CostComparison)
          and "both charged the same rate" in text)


def test_the_shipped_knob_table_matches_the_engine() -> None:
    """Every knob in ``KNOBS`` must be an argument ``run_setting`` accepts."""
    import inspect

    accepted = set(inspect.signature(run_setting).parameters)
    for name, label, grid, default in KNOBS:
        check(f"{label!r} sweeps a real engine argument ({name})", name in accepted)
        check(f"...and its grid brackets the default ({default})",
              min(grid) <= default <= max(grid) and default in grid)


if __name__ == "__main__":
    test_drift_and_turnover_by_hand()
    test_turnover_rejects_what_it_cannot_measure()
    test_turnover_on_the_real_rules()

    test_costs_land_on_the_right_day_and_only_subtract()
    test_breakeven_is_a_crossing_verified_from_both_sides()

    test_a_setting_runs_both_rules_on_one_window()
    test_sweep_moves_one_knob_at_a_time()
    test_the_shipped_knob_table_matches_the_engine()

    print(f"\n{_checks} checks passed")
