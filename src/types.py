"""THE CONTRACT between the engine and everything that measures it.

Every strategy in this project -- the ERC engine, all four benchmarks, the
comparison's matched column -- returns a ``BacktestResult``, so every metric,
chart, stress test and comparison treats them identically. Nothing downstream
has to know which one it is holding.

Because the contract is fixed, charts, metrics and stress tests can be built
and tested against ``tests.fixtures.make_fake_result()`` without a live engine.
That is what makes every test suite and module self-check run offline, and what
lets ``src/comparison.py`` consume four benchmarks and two engine runs without
special-casing any of them.

When this file needs to change, change the contract and every call site in the
same commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Weights and risk-contribution fractions are produced by a numerical solver, so
# they sum to 1 only to solver precision.
SUM_TO_ONE_TOLERANCE: float = 1e-6


class ContractViolation(AssertionError):
    """A BacktestResult does not satisfy the invariants documented here.

    Raised by :meth:`BacktestResult.validate`. This is deliberately loud: a
    silently malformed result is the one failure mode that would let the engine and
    the analysis layer diverge without anyone noticing.
    """


@dataclass(frozen=True)
class BacktestResult:
    """The output of any strategy in this project, engine or benchmark.

    Attributes
    ----------
    weights:
        Capital weights. Index is the rebalance date (monthly, a real trading
        day); columns are tickers. **Each row sums to 1.0** -- this is the
        *unlevered* allocation, before the overlay is applied. Long-only, so all
        entries are >= 0.
    risk_contributions:
        Same index and columns as ``weights``. Entry (t, i) is asset i's share
        of total portfolio volatility at t, i.e. ``w_i * (cov @ w)_i / sigma_p``
        normalised by ``sigma_p``. **Each row sums to 1.0.** For a perfect ERC
        portfolio every entry is 1/N -- that divergence from ``weights`` is the
        entire point of the project, so the two frames are deliberately kept
        side by side rather than derived on demand.
    leverage:
        The overlay multiplier L_t, indexed by rebalance date (same index as
        ``weights``). 1.0 means no overlay -- benchmarks set this to 1.0
        throughout. Bounded by ``config.LEVERAGE_CAP``.
    returns:
        **Daily** simple returns of the strategy, net of financing cost on the
        levered portion. This is the series every metric is computed from, so it
        is daily rather than monthly: annualise with
        ``config.TRADING_DAYS_PER_YEAR``.
    meta:
        Free-form diagnostics. Conventional keys: ``strategy`` (str, required),
        ``target_vol``, ``solver_failures`` (int), ``max_condition_number``.
        Anything a run wants to surface but no chart depends on goes here.
    """

    weights: pd.DataFrame
    risk_contributions: pd.DataFrame
    leverage: pd.Series
    returns: pd.Series
    meta: dict = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def strategy(self) -> str:
        return str(self.meta.get("strategy", "unnamed"))

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.columns)

    # -- invariants -------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`ContractViolation` if any documented invariant fails.

        Call this at the end of every function that builds a ``BacktestResult``.
        It is cheap, and it turns an integration bug into an immediate, located
        failure instead of a wrong chart three steps later.
        """
        w, rc = self.weights, self.risk_contributions

        if not isinstance(w, pd.DataFrame) or not isinstance(rc, pd.DataFrame):
            raise ContractViolation("weights and risk_contributions must be DataFrames")
        if not isinstance(self.leverage, pd.Series) or not isinstance(self.returns, pd.Series):
            raise ContractViolation("leverage and returns must be Series")

        if list(w.columns) != list(rc.columns):
            raise ContractViolation(
                f"column mismatch: weights {list(w.columns)} vs "
                f"risk_contributions {list(rc.columns)}"
            )
        if not w.index.equals(rc.index):
            raise ContractViolation("weights and risk_contributions must share an index")
        if not w.index.equals(self.leverage.index):
            raise ContractViolation("leverage must share the rebalance index of weights")

        for name, frame in (("weights", w), ("risk_contributions", rc)):
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ContractViolation(f"{name} must have a DatetimeIndex")
            if not frame.index.is_monotonic_increasing:
                raise ContractViolation(f"{name} index must be sorted ascending")
            if frame.isna().to_numpy().any():
                raise ContractViolation(f"{name} contains NaN")
            row_sums = frame.to_numpy(dtype="float64").sum(axis=1)
            worst = float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 0.0
            if worst > SUM_TO_ONE_TOLERANCE:
                raise ContractViolation(
                    f"{name} rows must sum to 1.0; worst deviation {worst:.3e}"
                )

        if (w.to_numpy(dtype="float64") < -SUM_TO_ONE_TOLERANCE).any():
            raise ContractViolation("weights must be long-only (non-negative)")

        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ContractViolation("returns must have a DatetimeIndex")
        if not self.returns.index.is_monotonic_increasing:
            raise ContractViolation("returns index must be sorted ascending")
        if self.returns.empty:
            raise ContractViolation("returns is empty")
        # Checked WITHOUT dropna: a NaN return is not a missing observation, it
        # is a day the strategy could not be priced. Metrics drop them silently,
        # so an unnoticed NaN run shortens the series and flatters every
        # annualized number computed from it.
        if not np.isfinite(self.returns.to_numpy(dtype="float64")).all():
            raise ContractViolation("returns contains NaN or non-finite values")

        # Anchor the return series to the allocation that produced it. Without
        # this, a returns Series covering a completely different date range
        # validates cleanly and every chart silently plots misaligned data.
        #
        # <= , not < : the project's one-day rule (overlay.py, backtest.py) is
        # that a decision taken at the close of day t earns the return from
        # t+1 onward, never t itself -- so the FIRST return must be strictly
        # after the first rebalance date, not merely not-before it. A `<`
        # check here let returns.index[0] == w.index[0] through, which is
        # exactly the look-ahead this project is built to prevent, silently
        # validated by the one function every BacktestResult is supposed to
        # pass through. tests/test_contract.py already asserts the strict
        # inequality on a fixture; this makes the contract itself enforce it.
        if self.returns.index[0] <= w.index[0]:
            raise ContractViolation(
                f"returns start {self.returns.index[0].date()} is not strictly "
                f"after the first rebalance {w.index[0].date()} -- a decision "
                f"taken at the close of a rebalance date cannot earn that same "
                f"day's return (the one-day rule), and a return before any "
                f"allocation decision cannot be explained by these weights"
            )

        if not np.isfinite(self.leverage.to_numpy(dtype="float64")).all():
            raise ContractViolation("leverage contains non-finite values")
        if (self.leverage.to_numpy(dtype="float64") <= 0.0).any():
            raise ContractViolation("leverage must be strictly positive")

        if "strategy" not in self.meta:
            raise ContractViolation("meta must carry a 'strategy' name")
