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
    benchmarks.py   equal-weight, 60/40, max-Sharpe, naive 1/vol
    stress.py       crisis decomposition, correlation-regime diagnostics
    viz.py          the report charts

``run_backtest.py`` wires it end to end. See TASKS.md for build order and
ownership, PLAN.md for the design rationale.

Three further modules sit beside that pipeline rather than in it:

    riskmetrics.py  VaR/CVaR, Sortino, Calmar, drawdown duration, and the
                    shared metric table -- new code beside the metrics.py port
    headtohead.py   one return series against another: correlation, beta,
                    capture, the tail block, the crisis table
    comparison.py   the study itself: equal risk against equal capital, with
                    ERC also levered to 1/N's own volatility so the returns are
                    comparable and the financing is counted

``comparison.py`` is what the other two were salvaged for; it consumes both and
runs the engine a second time at 1/N's volatility to build its matched column.
``compare_rules.py`` (step 56) will be its CLI.

The invariant that must never be violated: at rebalance date ``t``, only data
timestamped ``<= t`` may be used -- covariance, volatility estimate and
risk-free rate alike. ``tests/test_lookahead.py`` exists to prove it.
"""
