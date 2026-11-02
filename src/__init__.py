"""Equal risk against equal capital: a multi-asset portfolio-construction study.

The engine below builds an equal-risk-contribution book and can put it at any
volatility you ask for. The study is what that buys over equal capital and what
it costs -- see comparison.py.

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

Four further modules sit beside that pipeline rather than in it:

    riskmetrics.py  VaR/CVaR, Sortino, Calmar, drawdown duration, and the
                    shared metric table -- new code beside the metrics.py port
    headtohead.py   one return series against another: correlation, beta,
                    capture, the tail block, the crisis table
    comparison.py   the study itself: equal risk against equal capital, with
                    ERC also levered to 1/N's own volatility so the returns are
                    comparable and the financing is counted
    sensitivity.py  whether that result survives the covariance being
                    *estimated*: a one-knob-at-a-time sweep of the estimator
                    with both rules re-run together, plus reconstructed
                    turnover and the cost at which ERC stops beating 1/N

``comparison.py`` is what riskmetrics and headtohead were salvaged for; it
consumes both and runs the engine a second time at 1/N's volatility to build its
matched column. ``compare_rules.py`` is the CLI for both it and sensitivity.py.

The invariant that must never be violated: at rebalance date ``t``, only data
timestamped ``<= t`` may be used -- covariance, volatility estimate and
risk-free rate alike. ``tests/test_lookahead.py`` exists to prove it.
"""
