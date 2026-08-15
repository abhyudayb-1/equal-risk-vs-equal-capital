"""Volatility targeting and financing (TASKS.md steps 21-22).

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

**Financing (step 22).** ``r_p = L_t * (r_a - r_f) + r_f``. Borrowing ``L - 1`` of
capital costs ``(L-1) * r_f``, and leaving that leg out is the single most common
way a risk-parity backtest is oversold -- at ``L = 2`` against a 5% cash rate it
is worth ~5%/yr, more than the strategy's whole edge. It cuts both ways: below
``L = 1`` the un-deployed capital earns the rate instead, so a de-levered book is
not merely a smaller one. :func:`financed_returns` computes the algebraically
identical ``L*r_a - (L-1)*r_f`` for reasons its docstring measures.

**Which decision applies to which day.** Two rules live here and they differ by
one day, on purpose:

* :func:`leverage_path` sizes the book you are *about to hold*, so it uses the
  weights in force **at or before** the decision date.
* :func:`unlevered_returns` and :func:`financed_returns` attribute a return
  *earned on* a date, so they use the decision taken **strictly before** it --
  a choice made at the close of day ``d`` cannot have earned day ``d``'s move,
  and crediting it would be look-ahead of the flattering kind.

``_in_force_before`` is the second rule in one place. A plain
``reindex(...).ffill()`` implements the first, and using it for the second is the
bug this module is shaped to prevent.

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

# Ceiling on a plausible PER-PERIOD financing rate, used only to catch an
# annualised series passed where a de-compounded one belongs. src.data caps the
# annual rate at 25%, whose per-trading-day equivalent is ~8.9e-4, so 5e-3 (~250%
# annualised) is far above anything real and far below the annualised figure it
# exists to catch. Not a policy limit -- a units check.
_MAX_DAILY_RATE: float = 5e-3


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
# Step 22 -- financing. Leverage is borrowed, not free.
# ---------------------------------------------------------------------------
def unlevered_returns(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.DataFrame,
) -> pd.Series:
    """The portfolio return series ``w @ r``, before any leverage.

    ``weights`` is either a fixed vector or a frame of rebalance-date weights. In
    the frame case the allocation applied to date ``d`` is the one set at the last
    rebalance **strictly before** ``d`` -- see :func:`_in_force_before`, and note
    this is a *different* rule from the one :func:`leverage_path` uses. Both are
    correct for what they do:

    * estimating the vol of the book you are *about to hold* uses the weights at
      or before ``d``, because the allocation just chosen is the one being sized;
    * attributing the return *earned on* ``d`` uses the weights chosen strictly
      before ``d``, because a decision taken at the close of ``d`` cannot have
      earned that day's return.

    Collapsing the two into one ffill is look-ahead, and it is the flattering
    kind: the weights chosen at the close of a rebalance date would be credited
    with that date's move, which they were partly chosen *because of*.
    """
    if not isinstance(returns, pd.DataFrame):
        raise OverlayError("unlevered_returns needs a DataFrame of asset returns")
    _require_chronological(returns.index)

    if isinstance(weights, pd.DataFrame):
        aligned = _in_force_before(weights, returns.index, "weights")
        if list(aligned.columns) != list(returns.columns):
            raise OverlayError(
                f"weight columns {list(aligned.columns)} do not match return columns "
                f"{list(returns.columns)}"
            )
        row_sums = aligned.to_numpy(dtype="float64").sum(axis=1)
        worst = float(np.max(np.abs(row_sums - 1.0)))
        if worst > _WEIGHT_SUM_TOLERANCE:
            raise OverlayError(
                f"weights must sum to 1.0 on every date; worst deviation {worst:.3e}"
            )
        portfolio = (returns.to_numpy(dtype="float64") * aligned.to_numpy(dtype="float64")).sum(
            axis=1
        )
    else:
        w = _validate_weights(weights, expected=returns.shape[1])
        portfolio = returns.to_numpy(dtype="float64") @ w

    if not np.all(np.isfinite(portfolio)):
        raise OverlayError("unlevered portfolio returns are not all finite")
    return pd.Series(portfolio, index=returns.index, name="unlevered")


def financed_returns(
    unlevered: pd.Series,
    leverage: pd.Series | float,
    risk_free: pd.Series | float,
) -> pd.Series:
    """``r_p = L*(r_a - r_f) + r_f`` -- the levered return net of financing.

    Parameters
    ----------
    unlevered:
        The portfolio's own return series, from :func:`unlevered_returns`.
    leverage:
        A constant, or a series of rebalance-date decisions. As a series, the
        leverage applied to date ``d`` is the one set at the last rebalance
        **strictly before** ``d``, for the same reason as in
        :func:`unlevered_returns`.
    risk_free:
        The financing rate **per period, as a decimal** -- what
        ``src.data.fetch_risk_free`` returns, already de-compounded from FRED's
        annualised percentage. Passing an annualised figure here would overstate
        the financing cost by roughly 252x; the guard in :func:`_validate_rate`
        exists because that error produces a smoothly wrong equity curve rather
        than a crash.

    Notes
    -----
    **Why this leg exists.** Borrowing ``L - 1`` of capital costs ``(L-1)*r_f``.
    Omitting it hands the strategy free money in exactly the regimes it levers
    hardest -- calm ones, which historically are also high-rate ones -- and it is
    the single most common way a risk-parity backtest is oversold. At ``L = 2``
    and a 5% cash rate the omission is worth ~5%/yr, which is more than the
    strategy's entire edge.

    **It cuts both ways.** With ``L < 1`` the term ``-(L-1)*r_f`` is *positive*:
    the un-deployed capital sits in cash and earns the rate. So a de-levered book
    in a crisis is not simply a smaller book, and the floor of ``LEVERAGE_CAP``
    is not a pure drag.

    **The form actually computed** is the algebraically identical
    ``L*r_a - (L-1)*r_f``, and the rearrangement is not cosmetic. Step 22's
    "done when" is that ``L = 1`` reproduces the unlevered return *exactly*, and
    the two forms differ on precisely that point: measured over 2,000,000 draws
    at realistic magnitudes, ``L*(r_a - r_f) + r_f`` is bit-identical to ``r_a``
    on only 92.7% of them -- ``(a - b) + b != a`` in floating point -- while
    ``L*r_a - (L-1)*r_f`` collapses to ``r_a - 0.0`` and is exact on all of them.
    The second form also reads as what it is: the return on the levered book,
    minus the interest on the borrowed part.
    """
    if not isinstance(unlevered, pd.Series):
        raise OverlayError("unlevered must be a Series indexed by date")
    _require_chronological(unlevered.index)
    r_a = unlevered.to_numpy(dtype="float64")
    if not np.all(np.isfinite(r_a)):
        raise OverlayError("unlevered returns contain NaN or inf")

    lev = _decision_values(leverage, unlevered.index, "leverage")
    if np.any(lev <= 0.0):
        raise OverlayError(
            "leverage must be strictly positive; a zero or negative multiplier is "
            "not a de-levered book, it is a different strategy"
        )

    rate = _published_values(risk_free, unlevered.index, "risk_free")
    _validate_rate(rate)

    # L*r_a - (L-1)*r_f, not L*(r_a - r_f) + r_f. See the docstring: only this
    # form is bit-exact at L == 1, which is what the step's "done when" asks for.
    financed = lev * r_a - (lev - 1.0) * rate
    return pd.Series(financed, index=unlevered.index, name="strategy")


def apply_overlay(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.DataFrame,
    leverage: pd.Series | float,
    risk_free: pd.Series | float,
) -> pd.DataFrame:
    """Compose steps 21 and 22 into the frame step 23 accrues from.

    Columns: ``unlevered`` (``w @ r``), ``leverage`` (the multiplier in force),
    ``risk_free`` (the financing rate charged) and ``strategy`` (the financed
    return). Keeping all four side by side is deliberate -- the difference between
    ``unlevered`` and ``strategy`` is the financing leg, and a run that cannot
    show it is a run that cannot prove it charged for leverage at all.
    """
    unlevered = unlevered_returns(returns, weights)
    strategy = financed_returns(unlevered, leverage, risk_free)
    return pd.DataFrame(
        {
            "unlevered": unlevered,
            "leverage": _decision_values(leverage, unlevered.index, "leverage"),
            "risk_free": _published_values(risk_free, unlevered.index, "risk_free"),
            "strategy": strategy,
        },
        index=unlevered.index,
    )


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


def _in_force_before(source, target_index: pd.DatetimeIndex, label: str):
    """The ``source`` row in force for each target date: the last one *strictly* before it.

    This is the look-ahead guard for realised returns, and it is one character
    different from the wrong answer. ``reindex(target).ffill()`` takes the last
    entry at or *before* each date; on a date that is itself a rebalance, that is
    the decision taken at that day's close -- credited with a return it could not
    have earned, and taken partly *because of* that return. Over a monthly
    calendar that is roughly one day in twenty-one given the strategy's own
    hindsight, which is not a rounding error.

    ``searchsorted(side="left") - 1`` is the strict version: the last source
    index entry strictly less than the target date.
    """
    if not isinstance(source.index, pd.DatetimeIndex):
        raise OverlayError(f"{label} must be indexed by date")
    if not source.index.is_monotonic_increasing:
        raise OverlayError(f"{label} index must be sorted ascending")
    if source.index.has_duplicates:
        raise OverlayError(f"{label} index repeats a date")

    positions = source.index.searchsorted(target_index, side="left") - 1
    if np.any(positions < 0):
        first_usable = target_index[positions >= 0]
        raise OverlayError(
            f"{len(target_index) - len(first_usable)} return date(s) precede the first "
            f"{label} decision ({source.index[0].date()}); a return earned before any "
            f"decision cannot be attributed to one. Trim the returns to start after "
            f"{source.index[0].date()}"
        )
    return source.iloc[positions].set_axis(target_index)


def _scalar_or(value, target_index: pd.DatetimeIndex, label: str) -> np.ndarray | None:
    """Broadcast a finite scalar across ``target_index``; ``None`` if not scalar."""
    if not np.isscalar(value):
        if not isinstance(value, pd.Series):
            raise OverlayError(
                f"{label} must be a Series or a scalar, got {type(value).__name__}"
            )
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        raise OverlayError(f"{label} must be finite, got {scalar}")
    return np.full(len(target_index), scalar, dtype="float64")


def _decision_values(
    value: pd.Series | float, target_index: pd.DatetimeIndex, label: str
) -> np.ndarray:
    """Align a *decision* series: always the last entry STRICTLY before each date.

    Leverage is a decision, so this rule applies to it unconditionally -- and that
    "unconditionally" is the whole point. An earlier draft picked the rule by
    asking whether the series already covered the target index, on the theory
    that a series on the return calendar must be a published quantity rather than
    a decision. That is false for exactly the input step 23 is most likely to
    hand it: :func:`leverage_path` with its default daily dates produces a
    decision for *every* return date, so the shape test classified it as
    published and applied ``L_d`` to ``r_d`` -- reintroducing the one-day
    look-ahead this module is shaped to prevent, in the one code path most likely
    to be used.

    Kind, not shape. A decision is a decision whatever calendar it lands on.
    """
    broadcast = _scalar_or(value, target_index, label)
    if broadcast is not None:
        return broadcast

    aligned = _in_force_before(value, target_index, label)
    out = aligned.to_numpy(dtype="float64")
    if not np.all(np.isfinite(out)):
        raise OverlayError(f"{label} contains NaN or inf after alignment")
    return out


def _published_values(
    value: pd.Series | float, target_index: pd.DatetimeIndex, label: str
) -> np.ndarray:
    """Align a *published* series date for date.

    The risk-free rate is not a decision anyone in this backtest makes: the rate
    that applies to date ``d`` is the one published for ``d``, known at that day's
    close, so there is nothing to defer. Use ``src.data.align_risk_free`` to put
    it on the price calendar first -- it forward-fills across holidays only, which
    is the direction that cannot leak a future rate into a past day.
    """
    broadcast = _scalar_or(value, target_index, label)
    if broadcast is not None:
        return broadcast

    missing = target_index.difference(value.index)
    if len(missing) > 0:
        raise OverlayError(
            f"{label} does not cover {len(missing)} of the return dates, first "
            f"{missing[0].date()}; run it through src.data.align_risk_free before "
            "passing it here rather than letting the gap become a silent zero"
        )
    out = value.reindex(target_index).to_numpy(dtype="float64")
    if not np.all(np.isfinite(out)):
        raise OverlayError(f"{label} contains NaN or inf after alignment")
    return out


def _validate_rate(rate: np.ndarray) -> None:
    """Catch an annualised rate passed where a per-period one belongs.

    ``src.data.fetch_risk_free`` already de-compounds FRED's annualised
    percentage to a per-trading-day decimal, and ``data`` caps the annual figure
    at 25%, whose daily equivalent is ~8.9e-4. Anything above _MAX_DAILY_RATE is
    therefore not a daily rate at all -- almost always the annualised series
    passed straight through, which overstates financing by ~252x.

    Worth a guard rather than a comment because of how it fails: the equity curve
    stays smooth and merely underperforms, so it reads as the strategy being bad
    rather than as the units being wrong.
    """
    if np.any(rate < 0.0):
        # Negative policy rates are real, but they arrive as a small negative
        # annual figure, which de-compounds to a small negative daily one. data
        # already floors the annual series at -1%.
        worst = float(rate.min())
        if worst < -_MAX_DAILY_RATE:
            raise OverlayError(
                f"risk-free rate {worst:.3e}/period is more negative than any plausible "
                f"policy rate; check the units"
            )
    if np.any(rate > _MAX_DAILY_RATE):
        worst = float(rate.max())
        raise OverlayError(
            f"risk-free rate {worst:.3e} per period is implausible (bar {_MAX_DAILY_RATE:.1e}, "
            f"~{100 * ((1 + _MAX_DAILY_RATE) ** TRADING_DAYS_PER_YEAR - 1):.0f}%/yr). "
            "src.data.fetch_risk_free returns a PER-PERIOD decimal; this looks like the "
            "annualised series, which would overstate financing by about 252x"
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

    # =====================================================================
    # Step 22 -- financing
    # =====================================================================
    financing_window = synthetic_returns(300, np.array([0.16, 0.11, 0.19, 0.06]), rng)
    unlev = unlevered_returns(financing_window, equal)
    # 4e-5/day de-compounds to about 1.01%/yr, a plausible cash rate.
    cash = pd.Series(np.full(len(financing_window), 4e-5), index=financing_window.index)

    # --- step 22's "done when": L = 1 is the unlevered return, EXACTLY -------
    # Bit-identical, not merely close. This is why financed_returns computes
    # L*r_a - (L-1)*r_f rather than the spec's L*(r_a - r_f) + r_f: the latter is
    # (a - b) + b, which is not exactly a in floating point. Measured over
    # 2,000,000 draws at these magnitudes it misses on 7.3% of them.
    print("\nfinancing: L = 1 reproduces the unlevered return")
    # The series case drops the first day on purpose: a decision series covering
    # every return date has no decision strictly before its own first date, and
    # _in_force_before refuses to invent one rather than quietly dropping the day.
    after_first = unlev.iloc[1:]
    for label, target, lever, rate in (
        ("scalar L = 1.0", unlev, 1.0, cash),
        (
            "series of 1.0",
            after_first,
            pd.Series(np.ones(len(unlev)), index=unlev.index),
            cash.loc[after_first.index],
        ),
    ):
        same = np.array_equal(
            financed_returns(target, lever, rate).to_numpy(), target.to_numpy()
        )
        print(f"  {label:22s}: bit-identical to unlevered = {same}")

    # --- the financing leg costs what it should ------------------------------
    # (L-1) * r_f per period, annualised. Positive L below 1: the un-deployed
    # capital sits in cash and earns the rate, so de-levering is not a pure drag.
    print("\nfinancing costs (L-1)*r_f, and pays when L < 1")
    for lever in (0.5, 1.0, 1.5, 2.0, 3.0):
        financed = financed_returns(unlev, lever, cash)
        free = lever * unlev
        drag = TRADING_DAYS_PER_YEAR * float((financed - free).mean())
        expected = -(lever - 1.0) * 4e-5 * TRADING_DAYS_PER_YEAR + 0.0  # + 0.0 kills -0.0
        print(f"  L = {lever:<4}: {drag:+.4%}/yr against an expected {expected:+.4%}/yr "
              f"({'ok' if abs(drag - expected) < 1e-12 else 'MISMATCH'})")

    # --- the decision that applies to a day is the one made BEFORE it --------
    # One character from the wrong answer: reindex().ffill() takes the decision
    # at or before each date, which on a rebalance date is the one taken at that
    # day's close -- credited with a move it could not have earned and was partly
    # chosen because of. On the real four-sleeve run that hindsight is worth
    # +0.053%/yr of CAGR and +0.006 of Sharpe across 233 of 4896 days: small,
    # free, and exactly the kind of thing that never announces itself.
    print("\nleverage in force is the last decision STRICTLY before the day")
    days = financing_window.index
    schedule = pd.Series([1.0, 3.0], index=[days[0], days[5]])
    strict = _in_force_before(schedule, days[1:8], "leverage")
    naive = schedule.reindex(days[1:8].union(schedule.index)).ffill().reindex(days[1:8])
    for day in days[3:8]:
        marker = "  <-- rebalance date" if day in schedule.index else ""
        print(f"  {day.date()}  strictly-before {strict.loc[day]:.1f}   "
              f"naive ffill {naive.loc[day]:.1f}{marker}")

    # A DAILY leverage series is still a decision series, and must still be
    # deferred by a day. Caught while building step 22: an earlier draft picked
    # the alignment rule by asking whether the series already covered the return
    # index, on the theory that anything on the return calendar must be a
    # published quantity. leverage_path's default output is a decision for every
    # return date, so that test classified it as published and applied L_d to
    # r_d -- the one-day look-ahead this module exists to prevent, in the code
    # path step 23 is most likely to use. Kind, not shape.
    daily_decisions = pd.Series(
        np.where(np.arange(len(days)) < 5, 1.0, 3.0), index=days
    )
    live = financing_window.loc[financing_window.index > days[0]]
    effective = financed_returns(
        unlevered_returns(live, equal),
        daily_decisions,
        pd.Series(0.0, index=live.index),
    ) / unlevered_returns(live, equal)
    switch = days[5]
    print(f"  daily decision series: L switches at {switch.date()}, applied from "
          f"{effective[effective > 2.0].index[0].date()} "
          f"(deferred = {bool(np.isclose(effective.loc[switch], 1.0))})")

    # --- composed, and the financing leg is visible in the output ------------
    panel = apply_overlay(financing_window, equal, 2.0, cash)
    print(f"\napply_overlay columns: {list(panel.columns)}")
    print(f"  unlevered mean {TRADING_DAYS_PER_YEAR * panel['unlevered'].mean():+.3%}/yr, "
          f"strategy mean {TRADING_DAYS_PER_YEAR * panel['strategy'].mean():+.3%}/yr at L = 2")

    # --- degenerate inputs must raise, never return a plausible number -------
    print("\nbad input raises")
    flat = pd.DataFrame(
        np.zeros((300, 4)), index=pd.bdate_range("2006-02-03", periods=300), columns=tickers
    )
    # An ANNUALISED rate where a per-period one belongs: ~252x the real financing
    # cost, and it fails by making the equity curve smoothly worse rather than by
    # crashing, so it reads as a bad strategy instead of a units error.
    annualised = pd.Series(np.full(len(unlev), 0.05), index=unlev.index)
    for label, call in (
        ("constant prices (zero vol)", lambda: trailing_portfolio_vol(flat, equal)),
        ("weights not summing to 1", lambda: trailing_portfolio_vol(window, np.repeat(0.3, 4))),
        ("too little history", lambda: trailing_portfolio_vol(window.iloc[:50], equal)),
        ("negative halflife", lambda: trailing_portfolio_vol(window, equal, halflife=-1)),
        ("negative target vol", lambda: target_leverage(0.10, target_vol=-0.1)),
        ("inverted cap", lambda: target_leverage(0.10, cap=(3.0, 0.5))),
        ("annualised risk-free rate", lambda: financed_returns(unlev, 2.0, annualised)),
        ("zero leverage", lambda: financed_returns(unlev, 0.0, cash)),
        ("returns before any decision",
         lambda: financed_returns(unlev, pd.Series([2.0], index=[days[10]]), cash)),
    ):
        try:
            call()
            print(f"  {label:28s}: NO ERROR RAISED -- regression!")
        except OverlayError as exc:
            print(f"  {label:28s}: OverlayError, as it must ({str(exc)[:44]}...)")
