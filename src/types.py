"""THE CONTRACT between the engine (Person A) and the analysis layer (Person B).

Everything Person A produces and everything Person B consumes passes through
``BacktestResult``. Freezing it in Phase 0 is what lets both halves of the
project be written at the same time: Person B builds charts, metrics and stress
tests against ``tests.fixtures.make_fake_result()`` long before a real engine
exists, and in Phase 5 the fixture is swapped for the real thing.

**Do not change this file on a feature branch.** When it needs to change -- and
it will, once or twice -- say so first, then open ONE pull request that changes
the contract and both call sites together. If it changes more than about twice,
it was cut wrong; re-cut it together rather than patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

# Weights and risk-contribution fractions are produced by a numerical solver, so
# they sum to 1 only to solver precision.
SUM_TO_ONE_TOLERANCE: float = 1e-6


class ContractViolation(AssertionError):
    """A BacktestResult does not satisfy the invariants documented here.

    Raised by :meth:`BacktestResult.validate`. This is deliberately loud: a
    silently malformed result is the one failure mode that would let A's and B's
    halves diverge without anyone noticing until integration.
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
        if not np.isfinite(self.returns.dropna().to_numpy(dtype="float64")).all():
            raise ContractViolation("returns contains non-finite values")

        if not np.isfinite(self.leverage.to_numpy(dtype="float64")).all():
            raise ContractViolation("leverage contains non-finite values")

        if "strategy" not in self.meta:
            raise ContractViolation("meta must carry a 'strategy' name")


class ERCSolver(Protocol):
    """Signature every equal-risk-contribution solver in this project satisfies.

    Person A implements this as ``src.riskparity.solve_erc`` (SLSQP, step 17).
    Person B implements it independently as the convex log-barrier oracle inside
    ``tests/test_riskparity.py`` (step 19), and the two are asserted to agree to
    ``config.ERC_AGREEMENT_TOLERANCE``.

    Parameters
    ----------
    cov:
        (N, N) positive semi-definite covariance matrix, annualised.

    Returns
    -------
    np.ndarray
        (N,) long-only weights summing to 1.0, whose risk contributions are all
        equal to within the solver's tolerance.
    """

    def __call__(self, cov: np.ndarray, **kwargs) -> np.ndarray: ...


def solve_erc(cov: np.ndarray, **kwargs) -> np.ndarray:
    """Contract stub -- the real implementation lands in ``src/riskparity.py``.

    Present so that Person B can import and type against the signature during
    Phase 1-2 while Person A is still building it (step 17).
    """
    raise NotImplementedError(
        "solve_erc is implemented in src/riskparity.py (TASKS.md step 17)"
    )
