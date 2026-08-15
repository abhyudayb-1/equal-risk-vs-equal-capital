"""Self-checks for the fixed-weight benchmarks (TASKS.md step 11).

Runs standalone with no test framework, matching the house style::

    .venv/bin/python tests/test_benchmarks.py

The bar this file enforces is step 11's "done when": both benchmarks produce a
contract-valid ``BacktestResult`` with ``leverage == 1.0`` everywhere. The
leverage check is not a formality -- a benchmark that quietly borrowed would
flatter risk parity by comparison, and the whole 2022 argument rests on the
comparison being fair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.benchmarks import (  # noqa: E402
    _tangency_weights,
    equal_weight,
    max_sharpe,
    naive_risk_parity,
    sixty_forty,
)
from src.types import BacktestResult  # noqa: E402
from tests.fixtures import make_fake_returns  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


def check_is_valid_result(result: BacktestResult, returns: pd.DataFrame) -> None:
    """Every invariant step 11 promises, for one benchmark."""
    # validate() raises ContractViolation rather than returning a bool, so
    # reaching the next line at all is the assertion.
    result.validate()
    check(f"{result.strategy}: satisfies the BacktestResult contract", True)

    check(
        f"{result.strategy}: spans the full universe {list(returns.columns)}",
        result.tickers == list(returns.columns),
    )

    leverage = result.leverage
    check(
        f"{result.strategy}: leverage is exactly 1.0 at all {len(leverage)} rebalances",
        bool((leverage.to_numpy() == 1.0).all()),
    )
    check(
        f"{result.strategy}: leverage is indexed by rebalance date, not by day",
        leverage.index.equals(result.weights.index),
    )
    check(
        f"{result.strategy}: meta reports no target vol (unlevered)",
        result.meta.get("target_vol") is None,
    )
    check(
        f"{result.strategy}: returns start after the first allocation",
        result.returns.index[0] > result.weights.index[0],
    )


def test_equal_weight() -> None:
    print("equal weight:")
    returns = make_fake_returns()
    result = equal_weight(returns)
    check_is_valid_result(result, returns)

    expected = 1.0 / len(returns.columns)
    check(
        f"every sleeve holds {expected:.0%} of capital at every rebalance",
        bool(np.allclose(result.weights.to_numpy(), expected)),
    )


def test_sixty_forty() -> None:
    print("60/40:")
    returns = make_fake_returns()
    result = sixty_forty(returns)
    check_is_valid_result(result, returns)

    weights = result.weights
    check(
        "equities are 60% and nominal bonds 40% at every rebalance",
        bool(
            np.allclose(weights[config.EQUITY_TICKER], 0.60)
            and np.allclose(weights[config.NOMINAL_BOND_TICKER], 0.40)
        ),
    )

    unallocated = [
        t for t in weights.columns
        if t not in (config.EQUITY_TICKER, config.NOMINAL_BOND_TICKER)
    ]
    check(
        f"the unallocated sleeves {unallocated} are carried at exactly 0%",
        bool((weights[unallocated].to_numpy() == 0.0).all()),
    )
    check(
        "a sleeve held at 0% contributes exactly 0 risk",
        bool((result.risk_contributions[unallocated].to_numpy() == 0.0).all()),
    )


def test_naive_risk_parity() -> None:
    """The fifth comparator: inverse volatility, and where it is *not* equal risk.

    Added with the crisis table (TASKS.md step 24), which needs it as a row. The
    check that carries weight is the last one: ``1/sigma`` equals a true ERC
    portfolio only when every pairwise correlation is the same, and the fixture's
    correlations are emphatically not -- so its risk contributions must visibly
    miss 1/N. If they ever stop missing it, either the fixture has lost its
    correlation structure or this has quietly become the engine.
    """
    print("naive 1/vol:")
    returns = make_fake_returns()
    result = naive_risk_parity(returns)
    check_is_valid_result(result, returns)

    weights = result.weights
    check(
        "weights are strictly positive at every rebalance -- 1/sigma cannot be zero",
        bool((weights.to_numpy() > 0.0).all()),
    )
    check(
        "the lowest-volatility sleeve takes the largest mean weight",
        weights.mean().idxmax() == config.INFLATION_TICKER,
    )

    n = len(returns.columns)
    worst = float(np.abs(result.risk_contributions.to_numpy() - 1.0 / n).max())
    check(
        f"risk contributions miss 1/N by {worst:.3f} -- 1/sigma is not ERC when "
        f"correlations differ",
        worst > 0.02,
    )


def test_max_sharpe() -> None:
    print("max Sharpe:")
    returns = make_fake_returns()
    result = max_sharpe(returns)
    check_is_valid_result(result, returns)

    weights = result.weights.to_numpy()
    check(
        "weights are non-negative at every rebalance",
        bool((weights >= 0.0).all()),
    )
    check(
        "weights sum to 1.0 at every rebalance",
        bool(np.abs(weights.sum(axis=1) - 1.0).max() < 1e-9),
    )
    check(
        "the allocation actually moves, unlike the fixed-weight benchmarks",
        float(result.weights.std().max()) > 0.01,
    )
    check(
        "cvxpy converged at every rebalance",
        result.meta["solver_failures"] == 0,
    )
    # Not asserted to be zero: with every sleeve losing money over a lookback
    # there is genuinely no long-only tangency portfolio, and 2022 produces
    # exactly that. The requirement is that it is counted, not that it never
    # happens -- a silent substitution is the failure mode here.
    check(
        "minimum-variance substitutions are counted in meta",
        isinstance(result.meta.get("tangency_fallbacks"), int),
    )


def test_max_sharpe_on_random_covariances() -> None:
    """Step 12's stated bar, on covariances the fixtures never produce."""
    print("max Sharpe solver, 50 random PSD covariances:")
    rng = np.random.default_rng(0)
    solved = 0

    for _ in range(50):
        n = int(rng.integers(2, 7))
        factor = rng.standard_normal((n, n + 3))
        cov = factor @ factor.T / 10.0  # PSD by construction
        mean_excess = rng.standard_normal(n) * 0.1

        w = _tangency_weights(mean_excess, cov)
        if w is None:
            # Legitimate only when no asset has a positive expected excess
            # return, in which case no long-only tangency portfolio exists.
            if (mean_excess > 0).any():
                raise AssertionError(
                    f"FAILED: solver returned None despite {int((mean_excess > 0).sum())} "
                    "positive expected returns"
                )
            continue
        if w.min() < -1e-9 or abs(w.sum() - 1.0) > 1e-8:
            raise AssertionError(f"FAILED: weights off the simplex: sum={w.sum()}, min={w.min()}")
        solved += 1

    check(f"{solved} tangency portfolios are non-negative and sum to 1.0", solved > 0)
    check("no solver errors across 50 covariances of sizes 2-6", True)


def test_benchmarks_are_comparable() -> None:
    """All benchmarks must line up with each other, or no chart can hold them."""
    print("cross-benchmark:")
    returns = make_fake_returns()
    results = [equal_weight(returns), sixty_forty(returns), max_sharpe(returns)]
    first = results[0]

    check("same columns", all(r.tickers == first.tickers for r in results))
    check(
        "same rebalance calendar",
        all(r.weights.index.equals(first.weights.index) for r in results),
    )
    check(
        "same daily return index",
        all(r.returns.index.equals(first.returns.index) for r in results),
    )
    check(
        "distinct strategy names, so meta['strategy'] can key a results table",
        len({r.strategy for r in results}) == len(results),
    )


if __name__ == "__main__":
    test_equal_weight()
    test_sixty_forty()
    test_naive_risk_parity()
    test_max_sharpe()
    test_max_sharpe_on_random_covariances()
    test_benchmarks_are_comparable()
    print(f"\n{_checks} checks passed")
