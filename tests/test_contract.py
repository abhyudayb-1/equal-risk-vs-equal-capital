"""Self-checks for the Phase 0 contract, the metrics port, and the fixtures.

Runs standalone with no test framework, matching the house style:

    .\\.venv\\Scripts\\python.exe tests\\test_contract.py

This is the Phase 0 gate (TASKS.md step 7). It proves three things before
anyone builds on top of them:

1. ``BacktestResult.validate()`` accepts a well-formed result and *rejects*
   malformed ones -- a contract that never rejects anything is not a contract.
2. The metrics ported from ``sp500-factor-backtester`` kept the two subtleties
   that make them correct (the zero-variance guard and the drawdown peak floor).
3. The fixtures produce data with real regime structure, including the 2022
   equity/bond correlation flip that step 25's diagnostic has to detect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache, config, metrics  # noqa: E402
from src.types import BacktestResult, ContractViolation  # noqa: E402
from tests.fixtures import make_fake_result, make_fake_returns  # noqa: E402

_checks = 0


def check(label: str, condition: bool) -> None:
    global _checks
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    _checks += 1
    print(f"  ok   {label}")


def expect_violation(label: str, build) -> None:
    """Assert that building/validating a bad result raises ContractViolation."""
    try:
        build().validate()
    except ContractViolation:
        check(label, True)
        return
    raise AssertionError(f"FAILED: {label} -- no ContractViolation raised")


def test_contract() -> None:
    print("contract:")
    result = make_fake_result()
    result.validate()
    check("validate() accepts a well-formed result", True)
    check("strategy name is exposed", result.strategy == "fixture-risk-parity")
    check("tickers match the configured universe", tuple(result.tickers) == config.UNIVERSE)

    expect_violation(
        "rejects weights that do not sum to 1",
        lambda: BacktestResult(
            result.weights * 2, result.risk_contributions, result.leverage,
            result.returns, result.meta,
        ),
    )
    expect_violation(
        "rejects risk contributions that do not sum to 1",
        lambda: BacktestResult(
            result.weights, result.risk_contributions * 0.5, result.leverage,
            result.returns, result.meta,
        ),
    )
    expect_violation(
        "rejects negative (short) weights",
        lambda: BacktestResult(
            result.weights - 0.5, result.risk_contributions, result.leverage,
            result.returns, result.meta,
        ),
    )
    expect_violation(
        "rejects a leverage index that does not match weights",
        lambda: BacktestResult(
            result.weights, result.risk_contributions, result.leverage.iloc[:-3],
            result.returns, result.meta,
        ),
    )
    expect_violation(
        "rejects meta without a strategy name",
        lambda: BacktestResult(
            result.weights, result.risk_contributions, result.leverage,
            result.returns, {},
        ),
    )
    expect_violation(
        "rejects non-finite returns",
        lambda: BacktestResult(
            result.weights, result.risk_contributions, result.leverage,
            result.returns.copy().astype("float64").mask(
                result.returns.index == result.returns.index[5], np.inf
            ),
            result.meta,
        ),
    )


def test_metrics_port() -> None:
    print("metrics port:")
    result = make_fake_result()
    ppy = config.TRADING_DAYS_PER_YEAR
    summary = metrics.summarize(result.returns, periods_per_year=ppy, name=result.strategy)

    check(
        f"annualized vol {summary['annual_volatility']:.1%} is plausible",
        0.02 < summary["annual_volatility"] < 0.60,
    )
    check(f"max drawdown {summary['max_drawdown']:.1%} is negative", summary["max_drawdown"] < 0)
    check("sharpe is finite", bool(np.isfinite(summary["sharpe_ratio"])))
    check("drawdown peak precedes trough",
          summary["max_drawdown_peak"] is None
          or summary["max_drawdown_peak"] <= summary["max_drawdown_trough"])

    # The two subtleties carried over from sp500-factor-backtester.
    flat = pd.Series(0.01, index=result.returns.index)
    check(
        "zero-variance guard returns nan, not a ~1e16 Sharpe",
        bool(np.isnan(metrics.sharpe_ratio(flat, ppy))),
    )
    first_negative = pd.Series([-0.10, 0.02, 0.02], index=pd.bdate_range("2020-01-01", periods=3))
    check(
        "drawdown peak is floored at initial capital",
        metrics.max_drawdown(first_negative) <= -0.099,
    )

    # periods_per_year is required precisely so this cannot be got wrong silently.
    daily_vol = metrics.annualized_volatility(result.returns, config.TRADING_DAYS_PER_YEAR)
    monthly_vol = metrics.annualized_volatility(result.returns, config.MONTHS_PER_YEAR)
    check("annualization actually depends on periods_per_year", daily_vol > monthly_vol)


def test_cache_port() -> None:
    print("cache port:")
    probe = "_phase0_probe.csv"
    frame = make_fake_result().weights.head()
    cache.write_frame(probe, frame)
    restored = cache.read_frame(probe)
    check("frame round-trips through the cache", restored is not None
          and restored.shape == frame.shape)
    check("is_fresh() is true for a just-written file", cache.is_fresh(probe, 1.0))
    check("is_fresh() is false for a zero-age ceiling", not cache.is_fresh(probe, 0.0))
    check("missing entries read as None", cache.read_frame("_does_not_exist.csv") is None)
    (config.CACHE_DIR / probe).unlink()


def test_fixture_regimes() -> None:
    print("fixture regimes:")
    returns = make_fake_returns()
    equity, bond = config.CORRELATION_PAIR

    calm = returns.loc["2013":"2019"]
    calm_corr = calm[equity].corr(calm[bond])
    check(f"equity/bond correlation is negative in calm years ({calm_corr:+.2f})", calm_corr < 0)

    start, end = config.CRISIS_WINDOWS["Rate shock (2022)"]
    stressed = returns.loc[start:end]
    flip_corr = stressed[equity].corr(stressed[bond])
    check(
        f"equity/bond correlation FLIPS POSITIVE in 2022 ({flip_corr:+.2f})",
        flip_corr > 0,
    )

    for label, (a, b) in config.CRISIS_WINDOWS.items():
        window = returns.loc[a:b]
        window_vol = window[equity].std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        check(
            f"{label} is genuinely stressed (equity vol {window_vol:.0%})",
            window_vol > calm[equity].std() * np.sqrt(config.TRADING_DAYS_PER_YEAR),
        )

    check("fixtures are deterministic under a fixed seed",
          make_fake_returns(seed=7).equals(make_fake_returns(seed=7)))
    check("different seeds give different data",
          not make_fake_returns(seed=7).equals(make_fake_returns(seed=8)))


def test_capital_vs_risk() -> None:
    """The claim the whole project rests on, asserted on fixture data."""
    print("capital weight vs risk contribution:")
    result = make_fake_result()
    capital = result.weights.mean()
    risk = result.risk_contributions.mean()
    spread = float(capital.max() - capital.min())
    check(
        f"capital weights are far from equal (spread {spread:.0%})",
        spread > 0.20,
    )
    check(
        "risk contributions are near 1/N for every sleeve",
        bool((risk - 1.0 / len(result.tickers)).abs().max() < 0.01),
    )
    print(pd.DataFrame({"capital": capital, "risk": risk}).round(4).to_string(
        header=True, justify="right"))


if __name__ == "__main__":
    test_contract()
    test_metrics_port()
    test_cache_port()
    test_fixture_regimes()
    test_capital_vs_risk()
    print(f"\n{_checks} checks passed")
