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
from src.benchmarks import equal_weight, sixty_forty  # noqa: E402
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


def test_benchmarks_are_comparable() -> None:
    """Both benchmarks must line up with each other, or no chart can hold both."""
    print("cross-benchmark:")
    returns = make_fake_returns()
    ew, sf = equal_weight(returns), sixty_forty(returns)

    check("same columns", ew.tickers == sf.tickers)
    check("same rebalance calendar", ew.weights.index.equals(sf.weights.index))
    check("same daily return index", ew.returns.index.equals(sf.returns.index))
    check(
        "distinct strategy names, so meta['strategy'] can key a results table",
        ew.strategy != sf.strategy,
    )


if __name__ == "__main__":
    test_equal_weight()
    test_sixty_forty()
    test_benchmarks_are_comparable()
    print(f"\n{_checks} checks passed")
