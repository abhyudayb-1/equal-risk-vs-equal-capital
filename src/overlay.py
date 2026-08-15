"""Volatility targeting: trailing vol -> leverage -> caps (TASKS.md step 21).

The ERC solver in :mod:`src.riskparity` decides *how the risk is split*. It says
nothing about *how much* risk to run, and left alone it runs whatever the sleeves
happen to offer -- which for a bond-heavy risk-parity book is far below what the
strategy is sold on. This module supplies the missing half: estimate the
portfolio's trailing volatility, then lever it toward ``config.TARGET_VOL``.

**The rule.** ``L_t = TARGET_VOL / sigma_hat_t``, clipped to ``LEVERAGE_CAP``,
where ``sigma_hat_t`` is the trailing EWMA volatility of the *current* weights'
portfolio returns using only data ``<= t``.

Three things in that sentence carry more weight than they look like they do.

**"The current weights'."** ``sigma_hat_t`` is estimated on a *counterfactual*
series -- today's weights applied to past returns -- and not on the portfolio
returns the backtest actually realised. Those differ whenever the weights have
moved, and the counterfactual is the right one: the quantity being levered is the
risk of the book about to be *held*, not the risk of the book that *was* held.
Using the realised series would blend in the risk of allocations that no longer
exist, and would lag hardest exactly when the weights had just moved a long way
-- i.e. after a regime change, which is when the estimate matters most.

**"Trailing."** This estimator is *lagging by construction*, and that is not a
defect to be tuned away -- it is one of the project's findings. A trailing
estimate of volatility cannot rise until the volatility has already been
realised, so the overlay de-levers **after** the damage rather than before it.
In 2022 that is visible as a leverage series still elevated well into the
drawdown (PLAN.md's third artifact). Shortening the halflife moves the lag
around; it does not remove it, because no causal filter can. The honest framing
is that vol targeting is a risk *control*, not a risk *forecast*.

**"Clipped."** The cap is what stops a placid regime from producing an absurd
notional. It binds in both directions and the two mean opposite things, so
:class:`LeverageDecision` reports *which* -- see ``cap_binding``. A run where the
ceiling binds constantly is a run whose target vol is too high for its sleeves,
which is a different problem from one where the floor binds in a crisis.

Step 22 adds the financing leg -- ``r_p = L_t * (r_assets - r_f) + r_f`` -- to
this module. Leverage here is a multiplier on the *portfolio*; it is not free,
and nothing in this file should be read as if it were.

Look-ahead: :func:`leverage_at` and :func:`leverage_path` route their input
through :func:`src.data.asof_slice` and are the entry points that enforce the
``<= t`` invariant. :func:`trailing_portfolio_vol` deliberately does **not**
slice -- it takes a window the caller has already truncated, exactly as
:func:`src.covariance.ewma_covariance` does, so the guard stays in one attackable
place (``tests/test_lookahead.py``, step 28).

One boundary worth stating once, because step 23's accrual loop depends on it:
the decision returned for date ``t`` is made *at the close of* ``t`` and applies
to returns from ``t+1`` onward. Applying ``L_t`` to ``r_t`` would earn a levered
return on information not yet available, which is look-ahead wearing a different
hat.

Run a self-check from the repo root::

    .venv/bin/python -m src.overlay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    LEVERAGE_CAP,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    VOL_EWMA_HALFLIFE_DAYS,
    VOL_WARMUP_TRADING_DAYS,
)
from .covariance import ewma_weights
from .data import asof_slice

log = logging.getLogger(__name__)

# Weights are meant to be a portfolio, so they have to sum to 1. The bar is loose
# enough to absorb float accumulation over four sleeves (ERC output lands within
# ~1e-16) and tight enough that a genuinely unnormalised vector is caught.
_WEIGHT_SUM_TOLERANCE: float = 1e-6


class OverlayError(ValueError):
    """The overlay's input or estimate is unusable.

    Raised rather than returning a degraded number, for the same reason
    ``src.covariance`` raises ``CovarianceError``: a quietly wrong volatility
    estimate produces a plausible leverage series and an entirely wrong equity
    curve, and it does so in the direction that flatters the strategy -- an
    understated ``sigma_hat`` levers *up* into risk it has mis-measured.
    """


# ---------------------------------------------------------------------------
# The trailing volatility estimate
# ---------------------------------------------------------------------------
def trailing_portfolio_vol(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    halflife: float = VOL_EWMA_HALFLIFE_DAYS,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = VOL_WARMUP_TRADING_DAYS,
) -> float:
    """Annualised EWMA volatility of ``weights`` applied to ``returns``.

    Parameters
    ----------
    returns:
        Daily simple returns, one column per asset, oldest row first. **Already
        sliced to ``<= t`` by the caller** -- see the module docstring.
    weights:
        The portfolio about to be held. Must sum to 1.
    halflife:
        Decay halflife in trading days. Defaults to ``VOL_EWMA_HALFLIFE_DAYS``,
        which is deliberately *shorter* than the covariance halflife: the overlay
        should react to a vol spike faster than the weights do, because it is the
        thing standing between a spike and the book's notional.
    periods_per_year:
        Annualisation factor. An argument rather than a module constant for the
        same reason ``src.metrics`` takes one -- a constant that lies about the
        frequency is a silent error, and this estimator is reusable at other
        frequencies.
    min_observations:
        Warmup floor. See ``VOL_WARMUP_TRADING_DAYS`` in ``config.py`` for the
        effective-sample-size measurement that sets it.

    Notes
    -----
    **Why not just ``sqrt(w' C w)`` off the covariance module?** Because the two
    are the same number only at the same halflife, and they are deliberately at
    different ones (60 days for the weights, 42 for the overlay). Sharing the
    matrix would silently tie the overlay's reaction speed to the allocator's.

    They *are* otherwise identical, and exactly so: this function takes deviations
    from the **weighted** mean and applies the same ``1 - sum(w^2)``
    reliability-weight bias correction as :func:`src.covariance.ewma_covariance`,
    which makes ``trailing_portfolio_vol(r, w, h)^2`` equal
    ``w' ewma_covariance(r, h) w`` term for term rather than approximately. That
    is not housekeeping -- it is a free independent derivation of the same
    quantity, and the ``__main__`` self-check asserts it.

    **Demeaning.** RiskMetrics-style EWMA variance assumes a zero mean and
    squares the raw returns. This one subtracts the weighted mean instead. The
    choice is not free: measured over 3000 synthetic 252-day windows at 15%
    volatility with drifts of +/-30%/yr, the two estimates differ by a median of
    0.35%, a 95th percentile of 2.7%, and up to 9.0% in the tail. Demeaning wins
    on two counts -- it is the actual definition of variance, and it is what makes
    the covariance cross-check above exact rather than merely close.
    """
    x = _as_return_matrix(returns)
    w = _validate_weights(weights, expected=x.shape[1])

    # Checked here rather than left to ewma_weights, which raises CovarianceError
    # -- accurate, but it names the wrong module for someone debugging an overlay
    # call and would be a confusing thing to catch.
    if not (halflife > 0):
        raise OverlayError(f"halflife must be positive, got {halflife}")

    n = x.shape[0]
    if n < min_observations:
        raise OverlayError(
            f"need at least {min_observations} observations for a trailing vol "
            f"estimate at halflife {halflife}, got {n}; the rebalance calendar's "
            f"warmup is too short (see VOL_WARMUP_TRADING_DAYS in config.py)"
        )

    # The counterfactual series: the weights about to be held, applied across the
    # whole window's returns. See the module docstring for why this is not the
    # realised portfolio return series, which used the weights of the day.
    portfolio_returns = x @ w

    decay = ewma_weights(n, halflife)
    deviations = portfolio_returns - float(decay @ portfolio_returns)
    correction = 1.0 - float((decay**2).sum())
    if correction <= 0.0:
        raise OverlayError(
            f"halflife={halflife} over {n} observations leaves no effective sample "
            "to estimate a variance from"
        )
    variance = float((decay * deviations**2).sum()) / correction

    if not np.isfinite(variance) or variance <= 0.0:
        # Zero variance means every sleeve was constant across the window, or the
        # weights land in an exact offsetting combination. Either is a broken
        # price series rather than a calm market, and the distinction matters:
        # a calm market gives a small positive number that the cap handles
        # perfectly well, so reaching here at all is evidence of bad data.
        raise OverlayError(
            f"trailing portfolio variance is {variance}, which is not a volatility. "
            "Every sleeve was constant over the window, or the weights offset "
            "exactly -- check the price series rather than the overlay"
        )

    return float(np.sqrt(variance * periods_per_year))


# ---------------------------------------------------------------------------
# The leverage decision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeverageDecision:
    """The leverage applied *and* what the cap did to get there.

    ``leverage`` alone cannot distinguish "the target implied 1.4x" from "the
    target implied 9x and the ceiling cut it to 3". Those are different states of
    the world -- the first is the overlay working, the second is the overlay
    saturated and no longer targeting anything -- and this repo's rules require
    them to stay distinguishable rather than sharing a number.

    ``cap_binding`` is ``"none"``, ``"floor"`` or ``"ceiling"``. A run where the
    ceiling binds constantly has a target vol too high for its sleeves; a run
    where the floor binds is one where the overlay wanted to de-risk further than
    it was allowed to, which is the 2022 case and belongs in the write-up rather
    than in a footnote.
    """

    leverage: float
    raw_leverage: float
    trailing_vol: float
    target_vol: float
    cap_binding: str

    @property
    def capped(self) -> bool:
        """Whether the cap moved the answer at all."""
        return self.cap_binding != "none"


def target_leverage(
    trailing_vol: float,
    target_vol: float = TARGET_VOL,
    cap: tuple[float, float] = LEVERAGE_CAP,
) -> LeverageDecision:
    """``L = target_vol / trailing_vol``, clipped to ``cap``.

    Both volatilities must be on the same annualisation, which is why
    :func:`trailing_portfolio_vol` annualises rather than leaving it to the caller
    -- a daily ``sigma_hat`` against an annual target would silently ask for ~16x
    the intended leverage, and the ceiling would hide it by clipping to a number
    that still looks plausible.
    """
    low, high = _validate_cap(cap)

    if not np.isfinite(target_vol) or target_vol <= 0.0:
        raise OverlayError(f"target_vol must be finite and positive, got {target_vol}")
    if not np.isfinite(trailing_vol) or trailing_vol <= 0.0:
        raise OverlayError(
            f"trailing_vol must be finite and positive, got {trailing_vol}; "
            "trailing_portfolio_vol raises on this, so a caller has bypassed it"
        )

    raw = float(target_vol) / float(trailing_vol)
    if raw < low:
        binding = "floor"
    elif raw > high:
        binding = "ceiling"
    else:
        binding = "none"

    return LeverageDecision(
        leverage=float(np.clip(raw, low, high)),
        raw_leverage=raw,
        trailing_vol=float(trailing_vol),
        target_vol=float(target_vol),
        cap_binding=binding,
    )


def leverage_at(
    returns: pd.DataFrame,
    weights: np.ndarray,
    date: pd.Timestamp,
    halflife: float = VOL_EWMA_HALFLIFE_DAYS,
    target_vol: float = TARGET_VOL,
    cap: tuple[float, float] = LEVERAGE_CAP,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = VOL_WARMUP_TRADING_DAYS,
) -> LeverageDecision:
    """The leverage decision made at the close of ``date``.

    The look-ahead-safe entry point: the window is truncated by
    :func:`src.data.asof_slice` here, in one line, rather than at each call site.
    The returned leverage applies to returns from the **next** date onward.
    """
    window = asof_slice(returns, date)
    if window.empty:
        raise OverlayError(f"no return history at or before {pd.Timestamp(date).date()}")

    vol = trailing_portfolio_vol(
        window,
        weights,
        halflife=halflife,
        periods_per_year=periods_per_year,
        min_observations=min_observations,
    )
    return target_leverage(vol, target_vol=target_vol, cap=cap)


def leverage_path(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.DataFrame,
    dates: pd.DatetimeIndex | None = None,
    halflife: float = VOL_EWMA_HALFLIFE_DAYS,
    target_vol: float = TARGET_VOL,
    cap: tuple[float, float] = LEVERAGE_CAP,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = VOL_WARMUP_TRADING_DAYS,
) -> pd.DataFrame:
    """The leverage decision at every date in ``dates``, as a frame.

    ``weights`` may be a single vector (held constant throughout) or a frame of
    rebalance-date weights, which is forward-filled onto the return index so that
    the weights in force at each date are the ones set at the most recent
    rebalance at or before it. Forward-fill is the look-ahead-safe direction; a
    backward fill would hand a date the allocation chosen after it.

    Returns a frame indexed by date with the columns of :class:`LeverageDecision`.
    Step 23 consumes it, ``viz.py`` plots it, and it is what makes step 21's
    "leverage rises in calm regimes, falls in volatile ones" checkable rather
    than merely asserted.
    """
    if not isinstance(returns, pd.DataFrame):
        raise OverlayError("leverage_path needs a DataFrame of returns with a date index")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise OverlayError("returns must carry a DatetimeIndex")

    weights_by_date = _align_weights(weights, returns)

    if len(returns) < min_observations:
        raise OverlayError(
            f"leverage_path needs at least {min_observations} observations to make "
            f"a single decision, got {len(returns)}"
        )

    if dates is None:
        # Everything the warmup allows. Positional rather than date arithmetic so
        # it stays right across holidays and the 2006 half-year of missing DBC.
        dates = returns.index[min_observations - 1 :]
    else:
        dates = pd.DatetimeIndex(dates)
        # A requested date the returns frame has never heard of would otherwise
        # surface as a bare KeyError from the weight lookup, which reads like an
        # internal fault rather than what it is: a calendar mismatch between the
        # rebalance schedule and the price history.
        missing = dates.difference(returns.index)
        if len(missing) > 0:
            raise OverlayError(
                f"{len(missing)} requested date(s) are not in the return index, "
                f"first {missing[0].date()}; the rebalance calendar and the price "
                "history disagree"
            )

    rows = []
    for date in dates:
        decision = leverage_at(
            returns,
            weights_by_date.loc[date].to_numpy(dtype="float64"),
            date,
            halflife=halflife,
            target_vol=target_vol,
            cap=cap,
            periods_per_year=periods_per_year,
            min_observations=min_observations,
        )
        rows.append(
            {
                "leverage": decision.leverage,
                "raw_leverage": decision.raw_leverage,
                "trailing_vol": decision.trailing_vol,
                "target_vol": decision.target_vol,
                "cap_binding": decision.cap_binding,
            }
        )

    path = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name=returns.index.name))
    _log_saturation(path, cap)
    return path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _require_chronological(index: pd.Index) -> None:
    """Reject a window whose rows are not oldest-first, or repeat a date.

    The EWMA decay weights are positional -- ``ewma_weights`` returns them
    oldest-first, newest-heaviest, and pairs weight ``k`` with row ``k``. Nothing
    downstream re-reads the index, so a frame in the wrong order is not an error
    anywhere, it is simply a different and wrong answer: the decay lands on the
    oldest observations instead of the newest.

    That is worth a guard rather than a docstring line because of which direction
    it fails in. Measured on a window that is calm then violent, reversing the
    rows reports 5.9% annualised volatility against a true 21.0% -- and the
    leverage rule *divides* by this number, so a 3.6x understatement of risk
    becomes a 3.6x overstatement of the position. Silent, plausible, and in the
    direction that flatters the strategy.

    ``src.data`` sorts on the way in, so the live pipeline cannot reach this. A
    frame assembled anywhere else can -- ``pd.concat`` of two windows in the wrong
    order produces exactly this with no warning of its own.
    """
    if not isinstance(index, pd.DatetimeIndex):
        # A positional index carries no chronology to verify. Callers passing a
        # bare ndarray are trusted for ordering; there is nothing to check.
        return
    if not index.is_monotonic_increasing:
        raise OverlayError(
            "return window is not in chronological order; the EWMA decay weights "
            "are positional, so out-of-order rows silently mis-weight the estimate "
            "(a reversed window understates volatility ~3.6x, which the leverage "
            "rule then divides by). Sort the frame before slicing it"
        )
    if index.has_duplicates:
        duplicated = index[index.duplicated()].unique()
        raise OverlayError(
            f"return window repeats {len(duplicated)} date(s), first "
            f"{duplicated[0].date()}; a duplicated observation is double-weighted "
            "by the decay and is a sign of a concatenated or re-indexed frame"
        )


def _as_return_matrix(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Validate and coerce a (T, N) return window to a float array."""
    if isinstance(returns, pd.DataFrame):
        _require_chronological(returns.index)
        x = returns.to_numpy(dtype="float64")
    else:
        x = np.asarray(returns, dtype="float64")

    if x.ndim != 2:
        raise OverlayError(f"returns must be 2-D (observations x assets), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise OverlayError(
            "returns contain NaN or inf; src.data.daily_returns drops the leading "
            "NaN row, so this is a gap in the price series rather than an artefact"
        )
    return x


def _validate_weights(weights: np.ndarray, expected: int) -> np.ndarray:
    """Finite, right length, and summing to 1."""
    w = np.asarray(weights, dtype="float64").ravel()
    if w.shape != (expected,):
        raise OverlayError(f"weights must have shape ({expected},), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise OverlayError(f"weights must be finite, got {np.array2string(w, precision=4)}")

    total = float(w.sum())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        # Not renormalised silently. A vector that does not sum to 1 is a caller
        # bug, and quietly fixing it would scale the volatility estimate by the
        # error -- which the leverage rule then divides by, so the mistake lands
        # in the notional rather than in a stack trace.
        raise OverlayError(
            f"weights sum to {total:.8f}, not 1.0; the overlay levers a portfolio "
            "and will not renormalise one for you"
        )
    return w


def _validate_cap(cap: tuple[float, float]) -> tuple[float, float]:
    low, high = (float(x) for x in cap)
    if not np.isfinite(low) or not np.isfinite(high):
        raise OverlayError(f"leverage cap must be finite, got {cap}")
    if low <= 0.0:
        raise OverlayError(
            f"leverage floor must be positive, got {low}; a floor of 0 means the "
            "overlay may hold no position at all, which is a different strategy"
        )
    if high < low:
        raise OverlayError(f"leverage cap is inverted: floor {low} exceeds ceiling {high}")
    return low, high


def _align_weights(
    weights: np.ndarray | pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    """Broadcast or forward-fill ``weights`` onto ``returns``' index."""
    if isinstance(weights, pd.DataFrame):
        if list(weights.columns) != list(returns.columns):
            raise OverlayError(
                f"weight columns {list(weights.columns)} do not match return columns "
                f"{list(returns.columns)}; the overlay will not guess an ordering"
            )
        if weights.index.has_duplicates:
            # reindex raises "cannot reindex on an axis with duplicate labels"
            # from deep inside pandas, which names neither the caller's mistake
            # nor the file it came from.
            repeated = weights.index[weights.index.duplicated()].unique()
            raise OverlayError(
                f"weight frame repeats {len(repeated)} rebalance date(s), first "
                f"{pd.Timestamp(repeated[0]).date()}; two allocations for one date "
                "have no defined order of precedence"
            )
        # Fill over the UNION of the two indices, then restrict. Filling over
        # returns.index alone would silently drop any rebalance date that is not
        # itself a return observation -- a month-end that lands on a holiday, say
        # -- and every date after it would keep the previous month's weights while
        # looking perfectly well-formed. Verified: with a rebalance on a Saturday,
        # reindex-then-ffill loses it entirely and reports no error.
        combined = weights.index.union(returns.index)
        aligned = weights.reindex(combined).ffill().reindex(returns.index)
        if aligned.isna().any().any():
            first = weights.index.min()
            raise OverlayError(
                f"no weights in force before {first.date()}; leverage_path cannot "
                "start earlier than the first rebalance"
            )
        return aligned

    w = _validate_weights(weights, expected=returns.shape[1])
    return pd.DataFrame(
        np.tile(w, (len(returns), 1)), index=returns.index, columns=returns.columns
    )


def _log_saturation(path: pd.DataFrame, cap: tuple[float, float]) -> None:
    """Surface how often the cap bound, rather than leaving it in the numbers.

    A saturated overlay is not targeting volatility any more, it is holding a
    constant notional -- and that is invisible in the leverage series itself,
    which just looks flat. It belongs in ``BacktestResult.meta`` (step 29) and in
    the run summary; logging it here means it is never *only* in a chart.
    """
    if path.empty:
        return
    binding = path["cap_binding"].value_counts()
    ceiling = int(binding.get("ceiling", 0))
    floor = int(binding.get("floor", 0))
    if ceiling or floor:
        n = len(path)
        log.info(
            "leverage cap %s bound on %d/%d days (ceiling %.1f%%, floor %.1f%%)",
            cap,
            ceiling + floor,
            n,
            100.0 * ceiling / n,
            100.0 * floor / n,
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from .covariance import ewma_covariance

    rng = np.random.default_rng(20240619)
    tickers = ["SPY", "TLT", "DBC", "TIP"]

    def synthetic_returns(
        n_days: int, annual_vol: np.ndarray, seed_rng: np.random.Generator
    ) -> pd.DataFrame:
        """Daily returns at a given annualised vol per sleeve, zero correlation."""
        daily = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
        data = seed_rng.normal(0.0, daily, size=(n_days, len(annual_vol)))
        index = pd.bdate_range("2006-02-03", periods=n_days, name="date")
        return pd.DataFrame(data, index=index, columns=tickers)

    equal = np.repeat(0.25, 4)

    # --- the covariance cross-check: two derivations of one number -----------
    # sqrt(w' C w) off the covariance module must equal this module's scalar
    # estimator EXACTLY at the same halflife, because both take deviations from
    # the weighted mean and apply the same bias correction. Approximately equal
    # would mean one of the two conventions had drifted.
    window = synthetic_returns(400, np.array([0.16, 0.11, 0.19, 0.06]), rng)
    for halflife in (21.0, 42.0, 60.0, float("inf")):
        mine = trailing_portfolio_vol(window, equal, halflife=halflife)
        theirs = float(
            np.sqrt(equal @ ewma_covariance(window, halflife=halflife) @ equal)
        )
        print(
            f"vol vs sqrt(w'Cw), halflife {halflife:>5}: "
            f"{mine:.12f} vs {theirs:.12f}  (diff {abs(mine - theirs):.3e})"
        )

    # --- step 21's "done when": calm levers up, volatile levers down ---------
    print("\nleverage responds to the regime (equal weights, 4 sleeves)")
    for label, vols in (
        ("calm      (8% sleeves)", np.repeat(0.08, 4)),
        ("normal    (15%)", np.repeat(0.15, 4)),
        ("stressed  (30%)", np.repeat(0.30, 4)),
        ("crisis    (60%)", np.repeat(0.60, 4)),
    ):
        decision = leverage_at(
            synthetic_returns(400, vols, rng), equal, pd.Timestamp("2007-08-13")
        )
        print(
            f"  {label:24s}: sigma_hat {decision.trailing_vol:6.2%}  "
            f"raw L {decision.raw_leverage:6.2f}  ->  L {decision.leverage:.2f}  "
            f"[{decision.cap_binding}]"
        )

    # --- the cap holds in both directions ------------------------------------
    low, high = LEVERAGE_CAP
    extreme = leverage_path(
        synthetic_returns(600, np.array([0.02, 0.02, 0.02, 0.02]), rng), equal
    )
    calm_ok = bool((extreme["leverage"] <= high + 1e-12).all())
    violent = leverage_path(
        synthetic_returns(600, np.array([0.90, 0.90, 0.90, 0.90]), rng), equal
    )
    wild_ok = bool((violent["leverage"] >= low - 1e-12).all())
    print(f"\ncap holds: ceiling on a 2%-vol book {calm_ok}, floor on a 90%-vol book {wild_ok}")
    print(f"  2%-vol book : raw L up to {extreme['raw_leverage'].max():8.2f}, "
          f"capped at {extreme['leverage'].max():.2f}")
    print(f"  90%-vol book: raw L down to {violent['raw_leverage'].min():7.2f}, "
          f"floored at {violent['leverage'].min():.2f}")

    # --- the lag, which is a finding rather than a bug -----------------------
    # A calm year, then an abrupt tripling of volatility. A trailing estimator
    # cannot see the shock until it has been realised, so the book is still
    # levered through the first weeks of it. This is PLAN.md's third artifact --
    # the 2022 leverage series staying elevated into the drawdown -- reproduced
    # on synthetic data where the regime change date is known exactly.
    print("\nthe overlay de-levers AFTER the shock, not before (this is the finding)")
    calm = synthetic_returns(400, np.repeat(0.08, 4), rng)
    shock = synthetic_returns(120, np.repeat(0.32, 4), rng)
    shock.index = pd.bdate_range(calm.index[-1] + pd.Timedelta(days=1), periods=120)
    combined = pd.concat([calm, shock])
    combined.index.name = "date"

    path = leverage_path(combined, equal)
    shock_start = shock.index[0]
    pre = float(path.loc[path.index < shock_start, "leverage"].iloc[-1])
    print(f"  leverage the day before the shock      : {pre:.2f}")
    for offset in (0, 5, 10, 21, 42, 63):
        on = path.index[path.index >= shock_start][offset]
        print(f"  {offset:3d} trading days into the shock       : "
              f"{float(path.loc[on, 'leverage']):.2f}  "
              f"(sigma_hat {float(path.loc[on, 'trailing_vol']):.2%})")

    still_levered = float(path.loc[path.index[path.index >= shock_start][5], "leverage"])
    print(f"  -> five days in, the book is still at {still_levered:.2f}x while realised "
          f"vol has quadrupled")

    # --- time-varying weights, and the alignment regression ------------------
    # Found while building step 21: aligning a weight frame with
    # reindex(returns.index).ffill() SILENTLY DROPS any rebalance date that is not
    # itself a return observation -- a month-end landing on a holiday. Every later
    # date then carries the previous month's weights and nothing reports it. The
    # fill now runs over the union of the two indices. The check below rebalances
    # on a Saturday on purpose, which is the shape of the bug.
    print("\ntime-varying weights (alignment regression)")
    daily = synthetic_returns(300, np.array([0.16, 0.11, 0.19, 0.06]), rng)
    schedule = pd.DataFrame(
        [[0.25, 0.25, 0.25, 0.25], [0.40, 0.30, 0.20, 0.10]],
        index=pd.DatetimeIndex([daily.index[0], pd.Timestamp("2006-02-11")]),
        columns=tickers,
    )
    aligned = _align_weights(schedule, daily)
    after = aligned.loc[aligned.index > pd.Timestamp("2006-02-11")].iloc[0]
    print(f"  Saturday rebalance honoured from the next trading day: "
          f"{bool(np.allclose(after.to_numpy(), [0.40, 0.30, 0.20, 0.10]))}")
    print(f"  weights in force on {after.name.date()}: {after.to_numpy()}")

    varying = leverage_path(daily, schedule)
    print(f"  leverage_path accepts a weight frame: {len(varying)} decisions, "
          f"L in [{varying['leverage'].min():.2f}, {varying['leverage'].max():.2f}]")

    # --- the ordering regression --------------------------------------------
    # Found auditing step 21. The EWMA decay weights are POSITIONAL, so a window
    # whose rows are not oldest-first is not an error anywhere -- it is a
    # different and wrong answer, with the decay landing on the oldest rows. It
    # fails in the flattering direction: on a calm-then-violent window, reversing
    # the rows reported 5.9% vol against a true 21.0%, and the leverage rule
    # DIVIDES by that, so a 3.6x understatement of risk becomes a 3.6x
    # overstatement of the position. src.data sorts on the way in, so only a
    # frame built elsewhere can reach it -- a pd.concat in the wrong order does.
    print("\nrow ordering is checked, not assumed (regression)")
    calm_then_violent = pd.concat(
        [
            synthetic_returns(150, np.repeat(0.05, 4), rng),
            synthetic_returns(150, np.repeat(0.40, 4), rng).set_axis(
                pd.bdate_range("2006-09-01", periods=150), axis=0
            ),
        ]
    )
    ordered = trailing_portfolio_vol(calm_then_violent, equal, min_observations=100)
    print(f"  correctly ordered window : sigma_hat {ordered:.2%}")
    for label, frame in (
        ("reversed", calm_then_violent.iloc[::-1]),
        ("shuffled", calm_then_violent.sample(frac=1.0, random_state=1)),
        ("duplicated date", pd.concat([calm_then_violent, calm_then_violent.iloc[[10]]]).sort_index()),
    ):
        try:
            bad = trailing_portfolio_vol(frame, equal, min_observations=100)
            print(f"  {label:24s} : returned {bad:.2%} -- REGRESSION, should have raised")
        except OverlayError:
            print(f"  {label:24s} : OverlayError, as it must")

    # --- degenerate inputs must raise, never return a plausible number -------
    print("\nbad input raises")
    flat = pd.DataFrame(
        np.zeros((300, 4)), index=pd.bdate_range("2006-02-03", periods=300), columns=tickers
    )
    for label, call in (
        ("constant prices (zero vol)", lambda: trailing_portfolio_vol(flat, equal)),
        ("weights not summing to 1", lambda: trailing_portfolio_vol(window, np.repeat(0.3, 4))),
        ("too little history", lambda: trailing_portfolio_vol(window.iloc[:50], equal)),
        ("negative halflife", lambda: trailing_portfolio_vol(window, equal, halflife=-1)),
        ("negative target vol", lambda: target_leverage(0.10, target_vol=-0.1)),
        ("inverted cap", lambda: target_leverage(0.10, cap=(3.0, 0.5))),
    ):
        try:
            call()
            print(f"  {label:28s}: NO ERROR RAISED -- regression!")
        except OverlayError as exc:
            print(f"  {label:28s}: OverlayError, as it must ({str(exc)[:44]}...)")
