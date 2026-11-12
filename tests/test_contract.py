"""Self-checks for the result contract, the metrics port, and the fixtures.

Runs standalone with no test framework:

    .\\.venv\\Scripts\\python.exe tests\\test_contract.py

It proves three things everything else builds on:

1. ``BacktestResult.validate()`` accepts a well-formed result and *rejects*
   malformed ones -- a contract that never rejects anything is not a contract.
2. The metrics ported from ``sp500-factor-backtester`` kept the two subtleties
   that make them correct (the zero-variance guard and the drawdown peak floor).
3. The fixtures produce data with real regime structure, including the 2022
   equity/bond correlation flip that the regime diagnostic has to detect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache, config, metrics  # noqa: E402
from src.types import BacktestResult, ContractViolation  # noqa: E402
from tests.fixtures import (  # noqa: E402
    _ANNUAL_VOL,
    _CALM_CORR,
    _CRISIS_PROFILES,
    _FLIPPED_CORR,
    _corr_matrix,
    make_fake_result,
    make_fake_returns,
)

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
    # Shorting one sleeve into another keeps every row summing to 1.0, so this
    # actually exercises the long-only rule. The obvious `weights - 0.5` does
    # NOT: it breaks the row sum and is caught by the earlier check, leaving the
    # long-only branch silently untested.
    shorted = result.weights.copy()
    shorted.iloc[:, 0] -= 0.5
    shorted.iloc[:, 1] += 0.5
    expect_violation(
        "rejects short weights that still sum to 1",
        lambda: BacktestResult(
            shorted, result.risk_contributions, result.leverage,
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

    nan_returns = result.returns.copy().astype("float64")
    nan_returns.iloc[5] = np.nan
    expect_violation(
        "rejects NaN returns -- metrics drop them silently, flattering the result",
        lambda: BacktestResult(
            result.weights, result.risk_contributions, result.leverage,
            nan_returns, result.meta,
        ),
    )

    detached = result.returns.copy()
    detached.index = detached.index - pd.Timedelta(days=400)
    expect_violation(
        "rejects returns that start before the first allocation decision",
        lambda: BacktestResult(
            result.weights, result.risk_contributions, result.leverage,
            detached, result.meta,
        ),
    )

    # Regression: with no rebalance at all, validate() reached w.index[0] and
    # escaped as a bare IndexError instead of a ContractViolation.
    expect_violation(
        "rejects a result with no rebalances, as a violation rather than an IndexError",
        lambda: BacktestResult(
            result.weights.iloc[:0], result.risk_contributions.iloc[:0],
            result.leverage.iloc[:0], result.returns, result.meta,
        ),
    )

    dead_leverage = result.leverage.copy()
    dead_leverage.iloc[3] = 0.0
    expect_violation(
        "rejects non-positive leverage",
        lambda: BacktestResult(
            result.weights, result.risk_contributions, dead_leverage,
            result.returns, result.meta,
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
    probe = "_contract_probe.csv"
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

    # The 2022 assertions deliberately do NOT test for a positive correlation.
    # Live data refuted that framing: the real window averaged -0.09 and was
    # positive on only 31% of days. What actually happened is below.
    start, end = config.CRISIS_WINDOWS["Rate shock (2022)"]
    stressed = returns.loc[start:end]
    window_corr = stressed[equity].corr(stressed[bond])
    check(
        f"2022 equity/bond correlation rises toward zero without flipping hard "
        f"({window_corr:+.2f} vs {calm_corr:+.2f} calm)",
        window_corr > calm_corr,
    )

    # Asserted on the PROFILE, not on a realized path. Over a 0.78-year window
    # at ~24% annualized volatility, one path's total return carries a ~21%
    # standard deviation, so the realized ordering of two sleeves whose expected
    # returns differ by 7% flips from seed to seed. A test on the draw would be
    # flaky; a test on the parameters is exactly as informative and never is.
    profile = _CRISIS_PROFILES["Rate shock (2022)"]["annual_drift"]
    check(
        f"2022 profile makes BONDS fall harder than equities "
        f"({profile[bond]:+.1%}/yr vs {profile[equity]:+.1%}/yr)",
        profile[bond] < profile[equity] < 0,
    )
    check(
        f"2022 profile makes commodities the only sleeve that works "
        f"({profile[config.COMMODITY_TICKER]:+.1%}/yr)",
        profile[config.COMMODITY_TICKER] > 0,
    )
    check(
        "every crisis window has a calibrated profile",
        set(config.CRISIS_WINDOWS) == set(_CRISIS_PROFILES),
    )

    # The regime diagnostic: diversification degrades through 2022
    # even though the correlation sign never sustains a flip.
    #
    # Computed from the correlation TABLES, not from a sampled window. The
    # sampled version failed on 3 of 15 seeds -- the fixture's DR gap is only
    # ~0.04, because commodities decoupling (SPY/DBC +0.35 -> -0.15) partly
    # offsets the equity/bond move, which is realistic but leaves the realized
    # margin inside sampling noise. The ratio is scale-invariant, so the crisis
    # volatility multiplier drops out and the table comparison is exact.
    def theoretical_dr(table: dict) -> float:
        sd = np.array([_ANNUAL_VOL[t] for t in returns.columns])
        weights = (1.0 / sd) / (1.0 / sd).sum()
        cov = np.diag(sd) @ _corr_matrix(list(returns.columns), table) @ np.diag(sd)
        return float((weights @ sd) / np.sqrt(weights @ cov @ weights))

    calm_dr, stressed_dr = theoretical_dr(_CALM_CORR), theoretical_dr(_FLIPPED_CORR)
    check(
        f"the 2022 correlation regime degrades diversification "
        f"({stressed_dr:.3f} vs {calm_dr:.3f} calm)",
        stressed_dr < calm_dr,
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


def test_fixture_alignment() -> None:
    """No look-ahead in the fixture, and the covariance warmup is respected.

    Both of these were once wrong in the fixture. They matter because every
    benchmark copies the fixture's alignment convention, and the look-ahead
    test would then be asserting the opposite of what the fixture does.
    """
    print("fixture alignment:")
    result = make_fake_result()
    daily = make_fake_returns()
    first = result.weights.index[0]

    history = int((daily.index <= first).sum())
    check(
        f"first rebalance clears the {config.COV_WARMUP_TRADING_DAYS}-day warmup "
        f"({history} days of history)",
        history >= config.COV_WARMUP_TRADING_DAYS,
    )
    check(
        "returns start strictly after the first allocation decision",
        result.returns.index[0] > first,
    )

    applied = result.weights.reindex(daily.index).ffill().shift(1)
    check(
        "no weight is applied on the rebalance date itself (no look-ahead)",
        bool(applied.loc[first].isna().all()),
    )
    next_day = daily.index[daily.index.get_loc(first) + 1]
    check(
        "the new weights take effect the day AFTER the rebalance",
        bool(np.allclose(applied.loc[next_day].to_numpy(),
                         result.weights.iloc[0].to_numpy())),
    )

    try:
        _corr_matrix([config.EQUITY_TICKER, "NOT_A_TICKER"], _CALM_CORR)
        raise AssertionError("FAILED: unlisted correlation pair did not raise")
    except KeyError:
        check("an unlisted correlation pair raises instead of defaulting to 0.0", True)


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
    test_fixture_alignment()
    test_capital_vs_risk()
    print(f"\n{_checks} checks passed")
