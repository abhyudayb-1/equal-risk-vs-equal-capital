"""Systematic risk parity and volatility-targeting portfolio engine.

Pipeline, in order, one module per stage:

    config.py       constants, universe, crisis windows, policy
    cache.py        on-disk cache for network pulls
    types.py        the A/B contract: BacktestResult
    data.py         yfinance ETF closes + keyless FRED risk-free series
    covariance.py   EWMA covariance, shrinkage, PSD repair
    riskparity.py   risk contributions and the ERC solver
    overlay.py      volatility targeting, leverage caps, financing cost
    backtest.py     monthly rebalance loop -> BacktestResult
    metrics.py      Sharpe, CAGR, drawdown, crisis-window slices
    riskmetrics.py  VaR/CVaR, Sortino, Calmar, drawdown duration, the shared
                    metric table -- new code beside the metrics.py port
    headtohead.py   one return series against another: correlation, beta,
                    capture, the tail block, the crisis table
    benchmarks.py   equal-weight, 60/40, max-Sharpe, naive 1/vol
    stress.py       crisis decomposition, correlation-regime diagnostics
    viz.py          the report charts

``run_backtest.py`` wires it end to end. See TASKS.md for build order and
ownership, PLAN.md for the design rationale.

The invariant that must never be violated: at rebalance date ``t``, only data
timestamped ``<= t`` may be used -- covariance, volatility estimate and
risk-free rate alike. ``tests/test_lookahead.py`` exists to prove it.
"""
