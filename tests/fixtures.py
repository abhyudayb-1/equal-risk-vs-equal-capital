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

# Keyed off the config constants rather than string literals on purpose: keying
# these by "SPY"/"TLT" meant renaming a ticker in config.py silently produced an
# identity correlation matrix -- zero correlation everywhere, no error, and a
# fixture that quietly stopped exercising the diversification story.
_EQ = config.EQUITY_TICKER
_BOND = config.NOMINAL_BOND_TICKER
_CMD = config.COMMODITY_TICKER
_INFL = config.INFLATION_TICKER
_GOLD = config.GOLD_TICKER

# Calm-regime correlations. Equities and nominal bonds are NEGATIVELY
# correlated -- the assumption the whole strategy rests on.
_CALM_CORR = {
    (_EQ, _BOND): -0.30, (_EQ, _CMD): 0.35, (_EQ, _INFL): -0.05,
    (_EQ, _GOLD): 0.05, (_BOND, _CMD): -0.10, (_BOND, _INFL): 0.75,
    (_BOND, _GOLD): 0.20, (_CMD, _INFL): 0.20, (_CMD, _GOLD): 0.30,
    (_INFL, _GOLD): 0.30,
}

# 2022. NOT a dramatic positive flip -- that was this project's original
# assumption and live data refuted it. Across the 2022 drawdown the real
# SPY/TLT correlation averaged -0.09 and was positive on only 31% of days.
#
# The mechanism was subtler and more interesting: correlation merely rose toward
# zero, which removed the OFFSET. For two equal-vol legs at 50/50, going from
# -0.34 to ~0 lifts portfolio vol from 0.57 to 0.71 sigma -- a third of the
# diversification benefit gone with the sign still negative. That was enough,
# because the sleeve risk parity holds most of by capital is the one that fell
# hardest, with leverage on top.
#
# The fixture reproduces that subtle version on purpose. A caricature would let
# step 25's diagnostic pass while being unable to detect the real thing.
_FLIPPED_CORR = dict(_CALM_CORR) | {
    (_EQ, _BOND): 0.05, (_EQ, _INFL): 0.30, (_BOND, _INFL): 0.85,
    # Commodities were the one sleeve that worked in 2022, so they decouple.
    (_EQ, _CMD): -0.15, (_BOND, _CMD): -0.25,
}

# Annualized drift and volatility scaling per crisis, calibrated so each window
# reproduces the SHAPE of the real one. Measured totals inside each window:
#
#   GFC   2007-10-09 -> 2009-03-09 : SPY -54.8%  TLT +24.9%  DBC -26.9%  TIP  +3.3%
#   COVID 2020-02-19 -> 2020-03-23 : SPY -33.4%  TLT +14.2%  DBC -25.0%  TIP  -0.6%
#   2022  2022-01-03 -> 2022-10-12 : SPY -24.1%  TLT -31.2%  DBC +21.9%  TIP -13.2%
#
# Read the 2022 row twice. The bond sleeve fell HARDER than equities, and
# commodities -- the sleeve risk parity under-weights most by capital, because
# they are the highest-volatility one -- were the only thing that made money.
_CRISIS_PROFILES: dict[str, dict] = {
    "GFC (2008)": {
        "vol_scale": 2.5,
        "flipped": False,
        "annual_drift": {_EQ: -0.386, _BOND: 0.175, _CMD: -0.189, _INFL: 0.023, _GOLD: 0.10},
    },
    "COVID (2020)": {
        "vol_scale": 3.5,
        "flipped": False,
        "annual_drift": {_EQ: -3.66, _BOND: 1.56, _CMD: -2.74, _INFL: -0.07, _GOLD: -0.50},
    },
    "Rate shock (2022)": {
        "vol_scale": 1.5,
        "flipped": True,
        "annual_drift": {_EQ: -0.310, _BOND: -0.401, _CMD: 0.281, _INFL: -0.170, _GOLD: -0.03},
    },
}


def _corr_matrix(tickers: list[str], table: dict) -> np.ndarray:
    n = len(tickers)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            pair, reverse = (tickers[i], tickers[j]), (tickers[j], tickers[i])
            if pair in table:
                rho = table[pair]
            elif reverse in table:
                rho = table[reverse]
            else:
                # Never default to 0.0 -- an unlisted pair is a mistake, and a
                # silent zero would look like a working fixture.
                raise KeyError(
                    f"no fixture correlation for {pair}; add it to the table in "
                    f"tests/fixtures.py rather than relying on a default"
                )
            corr[i, j] = corr[j, i] = rho
    # Hand-written correlation tables are not guaranteed PSD; repair rather than
    # let a Cholesky failure surface as a confusing fixture error.
    eigvals, eigvecs = np.linalg.eigh(corr)
    if eigvals.min() < 1e-8:
        corr = eigvecs @ np.diag(np.clip(eigvals, 1e-8, None)) @ eigvecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def _regime_arrays(
    index: pd.DatetimeIndex, tickers: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-day (vol_scale, daily_drift, use_flipped_correlations).

    Every window in ``config.CRISIS_WINDOWS`` must have a matching entry in
    ``_CRISIS_PROFILES``; a missing one raises rather than silently falling back
    to calm-regime parameters, which would look like a working fixture whose
    crises had quietly stopped being crises.
    """
    n_days = len(index)
    ppy = config.TRADING_DAYS_PER_YEAR

    vol_scale = np.ones(n_days)
    drift = np.tile(
        np.array([_ANNUAL_DRIFT[t] for t in tickers]) / ppy, (n_days, 1)
    )
    flipped = np.zeros(n_days, dtype=bool)

    for label, (start, end) in config.CRISIS_WINDOWS.items():
        if label not in _CRISIS_PROFILES:
            raise KeyError(
                f"crisis window {label!r} has no entry in _CRISIS_PROFILES; add one "
                f"in tests/fixtures.py rather than letting it run as a calm regime"
            )
        profile = _CRISIS_PROFILES[label]
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        vol_scale[mask] = profile["vol_scale"]
        drift[mask] = np.array([profile["annual_drift"][t] for t in tickers]) / ppy
        if profile["flipped"]:
            flipped |= mask

    return vol_scale, drift, flipped


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

    calm_chol = np.linalg.cholesky(_corr_matrix(tickers, _CALM_CORR))
    flipped_chol = np.linalg.cholesky(_corr_matrix(tickers, _FLIPPED_CORR))

    vol_scale, drift, use_flipped = _regime_arrays(index, tickers)
    shocks = rng.standard_normal((n_days, n_assets))

    out = np.empty((n_days, n_assets))
    for i in range(n_days):
        chol = flipped_chol if use_flipped[i] else calm_chol
        out[i] = drift[i] + vols * vol_scale[i] * (chol @ shocks[i])

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

    # Respect the covariance warmup. Without this the fixture's first rebalance
    # sits ~18 trading days after DATA_START_DATE while the real engine's sits
    # ~252 days after it, so every chart Person B builds would have an x-axis a
    # year longer than the one the engine eventually produces.
    if len(daily) <= config.COV_WARMUP_TRADING_DAYS:
        raise ValueError(
            f"need more than {config.COV_WARMUP_TRADING_DAYS} trading days of "
            f"warmup; got {len(daily)}"
        )
    warmup_end = daily.index[config.COV_WARMUP_TRADING_DAYS - 1]
    rebalance = month_end_dates(daily.index)
    rebalance = rebalance[rebalance >= warmup_end]

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
    # asset returns.
    #
    # The .shift(1) is the whole ballgame. Without it the weights chosen at
    # rebalance date t earn the return OF day t -- a decision made using day t's
    # close, collecting day t's move. That is precisely the look-ahead bias this
    # project exists to avoid, and step 28's test is written to catch it. A
    # fixture that models it wrong would teach the wrong alignment convention to
    # every benchmark built against it (step 11 onward).
    #
    # Decision at t -> realized return over t to t+1.
    aligned_w = weights_df.reindex(daily.index).ffill().shift(1)
    aligned_l = leverage.reindex(daily.index).ffill().shift(1)
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
