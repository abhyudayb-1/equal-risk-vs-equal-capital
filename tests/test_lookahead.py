"""Look-ahead: the invariant everything else rests on.

Runs standalone, no framework, no network::

    .\\.venv\\Scripts\\python.exe tests\\test_lookahead.py

At rebalance date ``t`` only data timestamped ``<= t`` may be used -- covariance,
the trailing-vol estimate and the financing rate alike. ``src.data.asof_slice``
is the single chokepoint for that rule, which is what makes it attackable: this
file has one function to break rather than a guard re-derived at every call site.

**Every check here comes in a pair, and the second half is the point.** "The
future did not change the past" is satisfied perfectly by an engine that ignores
its input entirely, so each test also proves the perturbation *did* land
somewhere -- the spike must be invisible before it and visible after it. A test
without that positive control would still pass if ``run_backtest`` returned
constants.

Three attacks, in increasing strength:

1. **A future spike** (``test_a_future_spike_cannot_move_a_past_rebalance``).
   Multiply the last 200 days by 12 and require every earlier rebalance to be
   *bit-identical* -- not close, identical, because a leak of any size is a leak.
2. **Truncation equivalence** (``test_truncating_the_future_changes_nothing``).
   The stronger form, and it needs no perturbation to be chosen well: if a
   decision at ``t`` depends only on data ``<= t``, then deleting everything
   after some cutoff cannot change a single rebalance before it. This catches
   leaks the spike test would miss -- a leak through a quantity the spike happens
   not to move.
3. **Removing the guard** (``test_the_guard_is_what_makes_the_others_pass``).
   These tests must *fail* if the ``<= t`` guard is
   removed, and asserting that requires actually removing it. ``asof_slice`` is
   monkeypatched to a leaky identity in every module that imported it, and the
   checks above are re-run expecting failure. Same for the one-day rule: patch
   ``overlay._in_force_before`` back to the naive ``ffill`` and the day-of check
   must break.

The one-day rule is tested here too, because it is look-ahead wearing a
different hat: a decision taken at the close of ``t`` cannot have earned ``t``'s
return, and crediting it is worth +0.053%/yr of unearned CAGR on the real run.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data, overlay  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.covariance import estimate_covariance  # noqa: E402
from src.data import asof_slice  # noqa: E402
from src.overlay import leverage_at  # noqa: E402

_checks = 0

# Long enough for the 252-day warmup plus ~35 monthly rebalances, short enough
# that the ~8 backtests this file runs stay quick.
_N_DAYS = 1000

# The spike covers the last 200 trading days, which leaves roughly a quarter of
# the rebalances after it -- enough for the positive control to be convincing
# rather than resting on a single date.
_SPIKE_DAYS = 200

# A per-day rate, not an annualised one: ~1.01%/yr, matching what
# src.data.fetch_risk_free returns after de-compounding.
_DAILY_RATE = 4e-5


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# Synthetic data -- the same four-sleeve structure src.backtest's self-check uses
# ---------------------------------------------------------------------------
def _returns(n_days: int = _N_DAYS, seed: int = 20260816) -> pd.DataFrame:
    """Four correlated sleeves, in ``config.UNIVERSE`` order.

    Synthetic rather than live on purpose: this suite has to pass offline, and a
    look-ahead test needs a known perturbation date more than it needs realistic
    prices. The correlation structure is realistic anyway, because a degenerate
    one could let the ERC solver land on the same weights regardless of the
    window and hand the test a false pass.
    """
    rng = np.random.default_rng(seed)
    correlation = np.array(
        [
            [1.00, -0.30, 0.20, -0.10],
            [-0.30, 1.00, -0.05, 0.85],
            [0.20, -0.05, 1.00, 0.10],
            [-0.10, 0.85, 0.10, 1.00],
        ]
    )
    vols = np.array([0.170, 0.140, 0.200, 0.055]) / np.sqrt(config.TRADING_DAYS_PER_YEAR)
    shocks = rng.normal(size=(n_days, 4)) @ np.linalg.cholesky(correlation).T * vols
    index = pd.bdate_range(config.DATA_START_DATE, periods=n_days, name="date")
    return pd.DataFrame(shocks, index=index, columns=list(config.UNIVERSE))


def _rate(index: pd.DatetimeIndex, level: float = _DAILY_RATE) -> pd.Series:
    return pd.Series(level, index=index, name="risk_free")


def _poison_from(frame: pd.DataFrame, multiple: float = 12.0) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Multiply the last ``_SPIKE_DAYS`` rows by ``multiple``.

    Returns the poisoned frame and the first date it touches. Nothing before that
    date differs by a single bit, which is what lets the comparisons below be
    exact rather than tolerant.
    """
    poisoned = frame.copy()
    poisoned.iloc[-_SPIKE_DAYS:] *= multiple
    return poisoned, frame.index[-_SPIKE_DAYS]


_CACHE: dict[str, object] = {}


def _clean_run():
    """The unpoisoned backtest, computed once and reused across tests."""
    if "clean" not in _CACHE:
        frame = _returns()
        _CACHE["clean"] = run_backtest(frame, risk_free=_rate(frame.index))
    return _CACHE["clean"]


def _rows_before(result, date: pd.Timestamp) -> np.ndarray:
    return result.weights.index < date


def _leverage_in_force(result, index: pd.DatetimeIndex) -> pd.Series:
    """The leverage that earned each date: the last decision STRICTLY before it.

    Derived here rather than imported from ``overlay._in_force_before`` on
    purpose. A test that reconstructs the engine's alignment by calling the
    engine's alignment agrees with it by construction, including when both are
    wrong; ``searchsorted(side="left") - 1`` is the same rule written out
    independently.
    """
    positions = result.leverage.index.searchsorted(index, side="left") - 1
    if np.any(positions < 0):
        raise AssertionError("a return date precedes every leverage decision")
    return pd.Series(result.leverage.to_numpy()[positions], index=index)


# ---------------------------------------------------------------------------
# Removing the guard -- used by the last test, defined here because it explains
# what "the guard" actually is
# ---------------------------------------------------------------------------
@contextmanager
def _guard_removed():
    """Replace ``asof_slice`` with a leaky identity everywhere it was imported.

    ``src.backtest`` and ``src.overlay`` both do ``from .data import asof_slice``,
    which *binds the function into their own namespace*. Patching
    ``src.data.asof_slice`` alone would therefore change nothing at all, and the
    mutation test would pass while proving nothing -- the exact failure mode this
    file exists to catch. So every module holding a reference to the original is
    found and patched, which also means a newly added module is covered
    without anyone remembering to update this list.
    """
    original = data.asof_slice

    def leaky(frame, date):  # the guard, deleted
        return frame

    patched = []
    for module in list(sys.modules.values()):
        if getattr(module, "asof_slice", None) is original:
            setattr(module, "asof_slice", leaky)
            patched.append(module)
    try:
        yield patched
    finally:
        for module in patched:
            setattr(module, "asof_slice", original)


@contextmanager
def _one_day_rule_removed():
    """Put ``overlay._in_force_before`` back to the naive ``reindex().ffill()``.

    That is the bug ``overlay`` is shaped to prevent, and it is one character
    from the correct version: ``ffill`` takes the last decision at or *before*
    each date, so on a rebalance date it credits the allocation chosen at that
    day's close with that day's move.
    """
    original = overlay._in_force_before

    def naive(source, target_index, label):
        combined = source.index.union(target_index)
        return source.reindex(combined).ffill().reindex(target_index)

    overlay._in_force_before = naive
    try:
        yield original
    finally:
        overlay._in_force_before = original


# ---------------------------------------------------------------------------
# 1. The chokepoint itself
# ---------------------------------------------------------------------------
def test_asof_slice_is_the_chokepoint() -> None:
    print("asof_slice, the guard itself:")
    frame = _returns()
    date = frame.index[500]

    sliced = asof_slice(frame, date)
    check("the boundary date is included (<=, not <)", sliced.index[-1] == date)
    check("nothing after the boundary survives", sliced.index.max() <= date)
    check("nothing before it is dropped", len(sliced) == 501)
    check(
        "identical to an explicit boolean mask",
        sliced.equals(frame.loc[frame.index <= date]),
    )

    # A rebalance date that is not itself an observation -- a month-end on a
    # holiday. The guard must still admit everything up to it and nothing after.
    weekend = pd.Timestamp("2007-01-06")  # a Saturday
    check(
        "a non-trading boundary keeps the last prior trading day",
        asof_slice(frame, weekend).index[-1] == pd.Timestamp("2007-01-05"),
    )
    check("a string date is accepted", len(asof_slice(frame, str(date.date()))) == 501)
    check("works on a Series (the risk-free path)",
          asof_slice(frame[config.EQUITY_TICKER], date).index[-1] == date)

    check("a date before the history yields nothing",
          asof_slice(frame, pd.Timestamp("1999-01-01")).empty)
    check("a date after it yields everything",
          len(asof_slice(frame, pd.Timestamp("2099-01-01"))) == len(frame))

    # Value-based, not positional. An .iloc-based implementation would pass every
    # check above and silently return future rows on any frame not already
    # sorted -- and src.overlay's own docstring records that a mis-ordered window
    # understates volatility ~3.6x, in the flattering direction.
    shuffled = frame.sample(frac=1.0, random_state=0)
    check(
        "filters by date value, not row position",
        bool((asof_slice(shuffled, date).index <= date).all())
        and len(asof_slice(shuffled, date)) == 501,
    )


# ---------------------------------------------------------------------------
# 2. Each estimator, at its own entry point
# ---------------------------------------------------------------------------
def test_the_estimators_cannot_see_a_future_spike() -> None:
    print("covariance and the vol estimate, at their entry points:")
    frame = _returns()
    poisoned, spike = _poison_from(frame)
    before = frame.index[frame.index < spike][-1]
    after = frame.index[frame.index > spike][20]
    weights = np.repeat(0.25, 4)

    clean_cov = estimate_covariance(asof_slice(frame, before))
    dirty_cov = estimate_covariance(asof_slice(poisoned, before))
    check("covariance before the spike is bit-identical",
          np.array_equal(clean_cov, dirty_cov))

    # The positive control. Without it, an estimator that returned a constant
    # matrix would pass the check above with room to spare.
    moved_cov = estimate_covariance(asof_slice(poisoned, after))
    check("covariance after the spike does move",
          not np.allclose(estimate_covariance(asof_slice(frame, after)), moved_cov))

    clean_lev = leverage_at(frame, weights, before)
    dirty_lev = leverage_at(poisoned, weights, before)
    check("trailing vol before the spike is bit-identical",
          clean_lev.trailing_vol == dirty_lev.trailing_vol)
    check("leverage before the spike is bit-identical",
          clean_lev.leverage == dirty_lev.leverage)
    check(
        "leverage after it collapses toward the floor (the spike is real)",
        leverage_at(poisoned, weights, after).leverage
        < leverage_at(frame, weights, after).leverage,
    )


# ---------------------------------------------------------------------------
# 3. The whole engine, against a deliberate future spike
# ---------------------------------------------------------------------------
def test_a_future_spike_cannot_move_a_past_rebalance() -> None:
    print("run_backtest against a 12x spike in the last 200 days:")
    frame = _returns()
    poisoned, spike = _poison_from(frame)

    clean = _clean_run()
    dirty = run_backtest(poisoned, risk_free=_rate(poisoned.index))

    check("the two runs share a rebalance calendar",
          clean.weights.index.equals(dirty.weights.index))

    before = _rows_before(clean, spike)
    check(f"{int(before.sum())} rebalances precede the spike (not a vacuous test)",
          int(before.sum()) >= 10)

    # Exact, not np.allclose. A leak has no minimum size, and every input these
    # rows were computed from is bit-identical, so anything but equality is a
    # leak or a nondeterministic solver -- both worth failing on.
    for label, field in (
        ("weights", "weights"),
        ("risk contributions", "risk_contributions"),
    ):
        check(
            f"{label} before the spike are bit-identical",
            np.array_equal(
                getattr(clean, field).to_numpy()[before],
                getattr(dirty, field).to_numpy()[before],
            ),
        )
    check(
        "leverage before the spike is bit-identical",
        np.array_equal(clean.leverage.to_numpy()[before], dirty.leverage.to_numpy()[before]),
    )

    earned_before = clean.returns.index < spike
    check(
        "realised returns before the spike are bit-identical",
        np.array_equal(
            clean.returns.to_numpy()[earned_before],
            dirty.returns.to_numpy()[earned_before],
        ),
    )

    # The positive controls: the spike has to be visible somewhere, or none of
    # the above means anything.
    after = ~before
    check("weights after the spike DO move",
          not np.array_equal(clean.weights.to_numpy()[after], dirty.weights.to_numpy()[after]))
    check(
        "the overlay de-levers after it (mean leverage falls)",
        float(dirty.leverage.to_numpy()[after].mean())
        < float(clean.leverage.to_numpy()[after].mean()),
    )


# ---------------------------------------------------------------------------
# 4. Truncation equivalence -- the strongest form
# ---------------------------------------------------------------------------
def test_truncating_the_future_changes_nothing() -> None:
    print("truncation equivalence (no perturbation to choose badly):")
    frame = _returns()
    cutoff = frame.index[len(frame) // 2]

    full = _clean_run()
    truncated_frame = frame.loc[frame.index <= cutoff]
    truncated = run_backtest(truncated_frame, risk_free=_rate(truncated_frame.index))

    common = full.weights.index.intersection(truncated.weights.index)
    check(f"{len(common)} rebalances are shared with the full run", len(common) >= 5)
    check("the truncated run invents no rebalance the full one lacks",
          truncated.weights.index.difference(full.weights.index).empty)

    # If a decision at t used only data <= t, then having 500 more days of
    # history available cannot change it. This is a stronger statement than the
    # spike test: it holds for every leak, not just one the spike happens to move.
    check(
        "shared weights are bit-identical",
        np.array_equal(full.weights.loc[common].to_numpy(),
                       truncated.weights.loc[common].to_numpy()),
    )
    check(
        "shared leverage is bit-identical",
        np.array_equal(full.leverage.loc[common].to_numpy(),
                       truncated.leverage.loc[common].to_numpy()),
    )

    shared_days = full.returns.index.intersection(truncated.returns.index)
    check(f"shared return days ({len(shared_days)}) are bit-identical",
          np.array_equal(full.returns.loc[shared_days].to_numpy(),
                         truncated.returns.loc[shared_days].to_numpy()))

    # The diagnostics too: a leak into the covariance would show up here even if
    # it were too small to move the weights past the solver's tolerance.
    check(
        "per-rebalance condition numbers are bit-identical",
        np.array_equal(
            full.meta["diagnostics"].loc[common, "condition_number"].to_numpy(),
            truncated.meta["diagnostics"].loc[common, "condition_number"].to_numpy(),
        ),
    )


# ---------------------------------------------------------------------------
# 5. The financing rate is data too
# ---------------------------------------------------------------------------
def test_a_future_rate_change_cannot_move_a_past_return() -> None:
    print("the risk-free rate obeys the same rule:")
    frame = _returns()
    _, spike = _poison_from(frame)

    flat = _rate(frame.index)
    stepped = flat.copy()
    stepped.loc[stepped.index >= spike] = _DAILY_RATE * 10.0

    clean = _clean_run()
    hiked = run_backtest(frame, risk_free=stepped)

    check("a future rate change leaves the weights untouched",
          np.array_equal(clean.weights.to_numpy(), hiked.weights.to_numpy()))
    check("and the leverage untouched (the overlay never sees the rate)",
          np.array_equal(clean.leverage.to_numpy(), hiked.leverage.to_numpy()))

    earned_before = clean.returns.index < spike
    check(
        "returns earned before the hike are bit-identical",
        np.array_equal(clean.returns.to_numpy()[earned_before],
                       hiked.returns.to_numpy()[earned_before]),
    )
    # The positive control, as an identity rather than a comparison of means.
    # Weights and leverage are identical between the two runs, so the whole
    # difference after the hike must be the financing leg and nothing else:
    # (L-1) * (r_f_hiked - r_f_clean), with L the decision in force that day.
    # Asserting the exact quantity beats asserting "returns fell", which would
    # also hold for a bug that scaled the returns.
    after = clean.returns.index[~earned_before]
    lev = _leverage_in_force(clean, after)
    predicted = (lev - 1.0) * (stepped.loc[after] - flat.loc[after])
    observed = clean.returns.loc[after] - hiked.returns.loc[after]
    worst = float((observed - predicted).abs().max())
    # Not exact equality: `observed` is a difference of two ~1e-2 returns, so it
    # carries a few ulps of cancellation error (~2e-18 measured). The bar is set
    # a couple of decades above that and still twelve below the ~2e-4 quantity
    # being predicted, which is what makes it an identity rather than a fit.
    check(f"the hike costs exactly (L-1)*delta_rate per day ({worst:.1e})", worst < 1e-16)
    check("and it is charged on every day after it, not just on average",
          bool((observed.abs() > 0).all()))

    # align_risk_free forward-fills only. Back-filling would hand a past day a
    # rate published after it -- the same leak in the one series nobody thinks of
    # as a forecast.
    published = pd.Series(
        [0.01 / 252, 0.05 / 252],
        index=pd.DatetimeIndex(["2010-01-04", "2010-06-01"]),
        name="risk_free",
    )
    aligned = data.align_risk_free(published, pd.bdate_range("2010-01-04", "2010-12-31"))
    check("the old rate persists until the new one prints",
          float(aligned.loc["2010-05-28"]) == 0.01 / 252)
    check("the new rate applies from its print date, not before",
          float(aligned.loc["2010-06-01"]) == 0.05 / 252)


# ---------------------------------------------------------------------------
# 6. The one-day rule -- look-ahead wearing a different hat
# ---------------------------------------------------------------------------
def _implied_return(result, frame: pd.DataFrame, rate: pd.Series, *, same_day: bool) -> pd.Series:
    """Rebuild each rebalance date's return from a chosen decision.

    ``same_day=False`` uses the decision taken at the previous rebalance, which is
    the correct rule; ``same_day=True`` uses the one taken at that day's close,
    which is the flattering bug. Both are computed the same way otherwise, so the
    comparison isolates the alignment and nothing else.
    """
    values = {}
    for date in result.weights.index:
        if date not in result.returns.index:
            continue
        earlier = result.weights.index[result.weights.index < date]
        if not same_day and len(earlier) == 0:
            continue
        source = date if same_day else earlier[-1]
        w = result.weights.loc[source].to_numpy(dtype="float64")
        lev = float(result.leverage.loc[source])
        unlevered = float((frame.loc[date].to_numpy(dtype="float64") * w).sum())
        values[date] = lev * unlevered - (lev - 1.0) * float(rate.loc[date])
    return pd.Series(values)


def test_a_decision_cannot_earn_the_day_it_was_made() -> None:
    print("the one-day rule, on rebalance dates:")
    frame = _returns()
    rate = _rate(frame.index)
    result = _clean_run()

    previous = _implied_return(result, frame, rate, same_day=False)
    same_day = _implied_return(result, frame, rate, same_day=True)
    check(f"{len(previous)} rebalance dates are also return days", len(previous) >= 10)

    actual = result.returns.loc[previous.index]
    worst_previous = float((actual - previous).abs().max())
    typical_same_day = float((actual - same_day.loc[previous.index]).abs().median())

    check(
        f"the realised return matches the PREVIOUS decision ({worst_previous:.2e})",
        worst_previous < 1e-15,
    )
    # The two rules have to actually differ, or the check above is satisfied by
    # both and proves nothing about which one the engine used.
    check(
        f"and not the same-day one, which differs by ~{typical_same_day:.2e}",
        typical_same_day > 1e-6,
    )

    check(
        "returns start strictly after the first rebalance",
        result.returns.index[0] > result.weights.index[0],
    )


# ---------------------------------------------------------------------------
# 7. Remove the guard, and these tests must fail
# ---------------------------------------------------------------------------
def test_the_guard_is_what_makes_the_others_pass() -> None:
    print("removing the guard breaks the tests above :")
    frame = _returns()
    poisoned, spike = _poison_from(frame)
    clean = _clean_run()

    with _guard_removed() as patched:
        names = sorted(m.__name__ for m in patched)
        check(f"asof_slice was patched in {len(patched)} module(s): {names}",
              {"src.data", "src.backtest", "src.overlay"} <= set(names))

        leaky_dirty = run_backtest(poisoned, risk_free=_rate(poisoned.index))
        before = _rows_before(clean, spike)
        check(
            "WITHOUT the guard the spike reaches rebalances that precede it",
            not np.array_equal(
                clean.weights.to_numpy()[before], leaky_dirty.weights.to_numpy()[before]
            ),
        )

        truncated_frame = frame.loc[frame.index <= frame.index[len(frame) // 2]]
        leaky_truncated = run_backtest(
            truncated_frame, risk_free=_rate(truncated_frame.index)
        )
        common = clean.weights.index.intersection(leaky_truncated.weights.index)
        check(
            "WITHOUT the guard truncation equivalence breaks too",
            not np.array_equal(
                clean.weights.loc[common].to_numpy(),
                leaky_truncated.weights.loc[common].to_numpy(),
            ),
        )

    # Restored, and checked twice over -- by identity, and by behaviour. A
    # context manager that leaked its patch would turn every later run of this
    # file into a false pass, which is the one bug a mutation test cannot
    # survive having.
    check("every patched module holds the real asof_slice again",
          all(getattr(m, "asof_slice", None) is data.asof_slice for m in patched))
    restored = run_backtest(poisoned, risk_free=_rate(poisoned.index))
    before = _rows_before(clean, spike)
    check(
        "and the spike is invisible again in a fresh run",
        np.array_equal(restored.weights.to_numpy()[before],
                       clean.weights.to_numpy()[before]),
    )

    # The same treatment for the one-day rule. It is enforced by a different
    # mechanism (_in_force_before, not asof_slice), so it needs its own mutation.
    rate = _rate(frame.index)
    with _one_day_rule_removed() as real_rule:
        hindsight = run_backtest(frame, risk_free=rate)
        previous = _implied_return(hindsight, frame, rate, same_day=False)
        same_day = _implied_return(hindsight, frame, rate, same_day=True)
        actual = hindsight.returns.loc[previous.index]
        check(
            "WITHOUT the one-day rule the return matches the SAME-DAY decision",
            float((actual - same_day.loc[previous.index]).abs().max()) < 1e-15,
        )
        check(
            "and no longer matches the previous one",
            float((actual - previous).abs().median()) > 1e-6,
        )
        # Where the leak lands, exactly. The two alignments can only disagree on
        # a date that is itself a rebalance -- everywhere else the last decision
        # at-or-before and the last one strictly-before are the same decision --
        # so the set of days that moved must be precisely that set. Deterministic,
        # and it pins the mechanism rather than its size.
        moved = hindsight.returns.index[
            hindsight.returns.to_numpy() != clean.returns.to_numpy()
        ]
        rebalance_days = hindsight.returns.index.intersection(hindsight.weights.index)
        check(
            f"the leak lands on exactly the {len(rebalance_days)} rebalance days, "
            f"and no others",
            moved.equals(rebalance_days),
        )

        # NOT asserted: that the hindsight version earns *more*. It is worth
        # +0.053%/yr on the real four-sleeve run, and asserting the same sign here
        # was the first version of this check -- it failed on the very first seed
        # tried. Measured across 12 seeds the drift runs -0.148%/yr to +0.135%/yr
        # and is positive on 4 of them, so the sign is noise over 34 rebalance
        # days of one simulated path. That is this repo's standing rule ("never
        # assert on a single simulated path") in its most literal form. The
        # mechanism below is what makes the case; the *direction* is a property of
        # the live data and belongs in the write-up, not in a check that flips
        # with a seed.
        drift = config.TRADING_DAYS_PER_YEAR * float(
            hindsight.returns.mean() - clean.returns.mean()
        )
        print(f"       (hindsight is worth {drift:+.3%}/yr on this path; "
              f"sign is path-dependent, magnitude is not)")

    check("the one-day rule is restored on exit", overlay._in_force_before is real_rule)


if __name__ == "__main__":
    test_asof_slice_is_the_chokepoint()
    test_the_estimators_cannot_see_a_future_spike()
    test_a_future_spike_cannot_move_a_past_rebalance()
    test_truncating_the_future_changes_nothing()
    test_a_future_rate_change_cannot_move_a_past_return()
    test_a_decision_cannot_earn_the_day_it_was_made()
    test_the_guard_is_what_makes_the_others_pass()
    print(f"\n{_checks} checks passed")
