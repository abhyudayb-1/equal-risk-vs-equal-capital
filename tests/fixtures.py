"""Synthetic data with the right *shape* -- Person B's unblocking mechanism.

Person B builds benchmarks, metrics, stress tests and charts against these
fixtures during Phases 1-4, while Person A is still building the engine that
will eventually produce the real thing. In Phase 5 the fixture call is swapped
for a real ``BacktestResult`` and, if the contract in ``src/types.py`` held,
nothing else changes.

The data is deliberately more than noise:

* per-asset volatilities and correlations are roughly realistic for the four
  sleeves, so charts look like charts rather than static;
* the three crisis windows from ``config.CRISIS_WINDOWS`` are *actually*
  stressed -- elevated volatility and negative drift -- so drawdown tables and
  stress code have something to find;
* the 2022 window flips the equity/bond correlation **positive**, so the
  correlation diagnostic (TASKS.md step 25) can be tested before any real data
  exists.

Everything is seeded. Same seed in, same numbers out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.types import BacktestResult  # noqa: E402

DEFAULT_END_DATE = "2024-12-31"

# Annualized volatility per sleeve, roughly what these ETFs actually realise.
_ANNUAL_VOL: dict[str, float] = {
    config.EQUITY_TICKER: 0.16,
    config.NOMINAL_BOND_TICKER: 0.13,
    config.COMMODITY_TICKER: 0.18,
    config.INFLATION_TICKER: 0.05,
    config.GOLD_TICKER: 0.15,
}

# Annualized drift in the calm regime.
_ANNUAL_DRIFT: dict[str, float] = {
    config.EQUITY_TICKER: 0.08,
    config.NOMINAL_BOND_TICKER: 0.03,
    config.COMMODITY_TICKER: 0.04,
    config.INFLATION_TICKER: 0.02,
    config.GOLD_TICKER: 0.05,
}

_ORDER = (
    config.EQUITY_TICKER,
    config.NOMINAL_BOND_TICKER,
    config.COMMODITY_TICKER,
    config.INFLATION_TICKER,
    config.GOLD_TICKER,
)

# Calm-regime correlations. Equities and nominal bonds are NEGATIVELY
# correlated -- the assumption the whole strategy rests on.
_CALM_CORR = {
    ("SPY", "TLT"): -0.30, ("SPY", "DBC"): 0.35, ("SPY", "TIP"): -0.05,
    ("SPY", "GLD"): 0.05, ("TLT", "DBC"): -0.10, ("TLT", "TIP"): 0.75,
    ("TLT", "GLD"): 0.20, ("DBC", "TIP"): 0.20, ("DBC", "GLD"): 0.30,
    ("TIP", "GLD"): 0.30,
}

# 2022: the correlation flips positive and the diversification benefit vanishes.
_FLIPPED_CORR = dict(_CALM_CORR) | {
    ("SPY", "TLT"): 0.50, ("SPY", "TIP"): 0.45, ("TLT", "DBC"): 0.10,
    ("TLT", "TIP"): 0.85, ("SPY", "DBC"): 0.20,
}


def _corr_matrix(tickers: list[str], table: dict) -> np.ndarray:
    n = len(tickers)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = (tickers[i], tickers[j])
            rho = table.get(key, table.get((tickers[j], tickers[i]), 0.0))
            corr[i, j] = corr[j, i] = rho
    # Hand-written correlation tables are not guaranteed PSD; repair rather than
    # let a Cholesky failure surface as a confusing fixture error.
    eigvals, eigvecs = np.linalg.eigh(corr)
    if eigvals.min() < 1e-8:
        corr = eigvecs @ np.diag(np.clip(eigvals, 1e-8, None)) @ eigvecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def _regime_flags(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(in_any_crisis, in_2022_window) boolean masks over ``index``."""
    crisis = np.zeros(len(index), dtype=bool)
    flipped = np.zeros(len(index), dtype=bool)
    for label, (start, end) in config.CRISIS_WINDOWS.items():
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        crisis |= mask
        if "2022" in label:
            flipped |= mask
    return crisis, flipped


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available date of each month in ``index``.

    Mirrors what ``src/data.py`` will do with the real trading calendar
    (TASKS.md step 10), so code written against the fixture keeps working.
    """
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).last().to_numpy())


def make_fake_returns(
    tickers: tuple[str, ...] = config.UNIVERSE,
    start: str = config.DATA_START_DATE,
    end: str = DEFAULT_END_DATE,
    seed: int = 0,
) -> pd.DataFrame:
    """Daily simple returns per asset, with crisis regimes baked in."""
    tickers = list(tickers)
    unknown = set(tickers) - set(_ORDER)
    if unknown:
        raise ValueError(f"no fixture parameters for {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, end=end, name="date")
    n_days, n_assets = len(index), len(tickers)

    vols = np.array([_ANNUAL_VOL[t] for t in tickers]) / np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )
    drift = np.array([_ANNUAL_DRIFT[t] for t in tickers]) / config.TRADING_DAYS_PER_YEAR

    calm = np.linalg.cholesky(_corr_matrix(tickers, _CALM_CORR))
    flipped = np.linalg.cholesky(_corr_matrix(tickers, _FLIPPED_CORR))

    in_crisis, in_flip = _regime_flags(index)
    shocks = rng.standard_normal((n_days, n_assets))

    out = np.empty((n_days, n_assets))
    for i in range(n_days):
        chol = flipped if in_flip[i] else calm
        correlated = chol @ shocks[i]
        if in_crisis[i]:
            # Crises are higher-vol and negative-drift; the 2022 rate shock is a
            # grind rather than a crash, so it is scaled less than 2008/2020.
            scale = 1.6 if in_flip[i] else 2.5
            mu = -3.0 * drift if in_flip[i] else -6.0 * drift
        else:
            scale, mu = 1.0, drift
        out[i] = mu + vols * scale * correlated

    return pd.DataFrame(out, index=index, columns=tickers)


def make_fake_prices(**kwargs) -> pd.DataFrame:
    """Price levels implied by :func:`make_fake_returns`, all starting at 100."""
    returns = make_fake_returns(**kwargs)
    return 100.0 * (1.0 + returns).cumprod()


def make_fake_risk_free(
    start: str = config.DATA_START_DATE,
    end: str = DEFAULT_END_DATE,
    seed: int = 0,
) -> pd.Series:
    """Daily risk-free rate, shaped like DTB3: ~5%, ~0% post-GFC, rising in 2022."""
    index = pd.bdate_range(start=start, end=end, name="date")
    years = (index - index[0]).days / 365.25
    annual = np.where(
        years < 2.5, 0.045, np.where(years < 15.5, 0.005, np.clip((years - 15.5) * 0.03, 0, 0.05))
    )
    rng = np.random.default_rng(seed)
    annual = np.clip(annual + rng.normal(0, 0.0003, len(index)), 0.0, None)
    return pd.Series(annual / config.TRADING_DAYS_PER_YEAR, index=index, name="risk_free")


def make_fake_result(
    tickers: tuple[str, ...] = config.UNIVERSE,
    start: str = config.DATA_START_DATE,
    end: str = DEFAULT_END_DATE,
    strategy: str = "fixture-risk-parity",
    levered: bool = True,
    seed: int = 0,
) -> BacktestResult:
    """A contract-valid ``BacktestResult`` with no engine behind it.

    Weights lean toward the low-volatility sleeves the way a real risk-parity
    solution does, risk contributions sit near 1/N with solver-scale noise, and
    leverage mean-reverts inside ``config.LEVERAGE_CAP``. Pass ``levered=False``
    to imitate a benchmark, which pins leverage at 1.0.
    """
    tickers = list(tickers)
    rng = np.random.default_rng(seed + 1)

    daily = make_fake_returns(tuple(tickers), start=start, end=end, seed=seed)
    rebalance = month_end_dates(daily.index)
    n_reb, n_assets = len(rebalance), len(tickers)

    # Weights: inverse-vol baseline plus a smooth random walk, so the series
    # drifts the way a real solution does instead of jumping every month.
    inv_vol = np.array([1.0 / _ANNUAL_VOL[t] for t in tickers])
    base = np.log(inv_vol / inv_vol.sum())
    walk = np.cumsum(rng.normal(0.0, 0.05, (n_reb, n_assets)), axis=0)
    raw = np.exp(base + walk)
    weights = raw / raw.sum(axis=1, keepdims=True)

    # Risk contributions: an ERC solution sits at 1/N, off by solver tolerance.
    rc_raw = np.full((n_reb, n_assets), 1.0 / n_assets) + rng.normal(
        0.0, 0.004, (n_reb, n_assets)
    )
    rc_raw = np.clip(rc_raw, 1e-6, None)
    risk_contributions = rc_raw / rc_raw.sum(axis=1, keepdims=True)

    if levered:
        # Mean-reverting around 1.5x, and deliberately slow to fall -- trailing
        # volatility is a lagging estimator, which is the point of the 2022 story.
        lev = np.empty(n_reb)
        level = 1.5
        for i in range(n_reb):
            level = 0.85 * level + 0.15 * 1.5 + rng.normal(0.0, 0.12)
            lev[i] = level
        lev = np.clip(lev, *config.LEVERAGE_CAP)
    else:
        lev = np.ones(n_reb)

    weights_df = pd.DataFrame(weights, index=rebalance, columns=tickers)
    rc_df = pd.DataFrame(risk_contributions, index=rebalance, columns=tickers)
    leverage = pd.Series(lev, index=rebalance, name="leverage")

    # Daily portfolio return: last-known weights and leverage applied to daily
    # asset returns. Uses ffill so a day before the first rebalance is dropped
    # rather than silently treated as equal-weight.
    aligned_w = weights_df.reindex(daily.index).ffill()
    aligned_l = leverage.reindex(daily.index).ffill()
    gross = (aligned_w * daily).sum(axis=1)
    rf = make_fake_risk_free(start=start, end=end, seed=seed).reindex(daily.index).ffill()
    net = aligned_l * (gross - rf) + rf
    net = net.loc[aligned_w.dropna().index].rename("returns")

    result = BacktestResult(
        weights=weights_df,
        risk_contributions=rc_df,
        leverage=leverage,
        returns=net,
        meta={
            "strategy": strategy,
            "target_vol": config.TARGET_VOL if levered else None,
            "solver_failures": 0,
            "source": "tests.fixtures.make_fake_result",
            "seed": seed,
        },
    )
    result.validate()
    return result


if __name__ == "__main__":
    res = make_fake_result()
    print(f"strategy   : {res.strategy}")
    print(f"rebalances : {len(res.weights)}  ({res.weights.index[0].date()} -> "
          f"{res.weights.index[-1].date()})")
    print(f"daily obs  : {len(res.returns)}")
    print(f"leverage   : {res.leverage.min():.2f} - {res.leverage.max():.2f}")
    print("\nmean capital weight vs mean risk contribution:")
    comparison = pd.DataFrame(
        {"capital": res.weights.mean(), "risk": res.risk_contributions.mean()}
    )
    print(comparison.round(4).to_string())
