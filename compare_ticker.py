#!/usr/bin/env python
"""CLI for the head-to-head comparison (step 44).

How did the risk-parity strategy do against simply buying and holding a ticker
you choose, and what was the difference in risk? This runs the project's real
engine over its full history exactly as ``run_backtest.py`` does, buys and holds
the ticker (total return), cuts both to the dates the ticker exists, and prints
``src.compare.format_comparison_report``'s account of it -- risk metrics for the
levered strategy, the unlevered book and the ticker, then correlation, beta,
tracking error, up/down capture, what the strategy did on the ticker's worst
days, and the crisis windows. See ``src/compare.py`` for what each number means
and PLAN.md's "Phase 7" section for why it is built this way. This file owns none
of that: it parses arguments, fetches, runs, and prints, the same "wiring only"
role ``run_backtest.py`` and ``analyze_position.py`` play.

Run it::

    .venv/bin/python compare_ticker.py --ticker AAPL
    .venv/bin/python compare_ticker.py --help              # every flag, documented

Any ticker yfinance covers works, including the four sleeves themselves
(``--ticker SPY`` is the classic "risk parity versus the S&P 500" question).

**Live data, and a full engine run.** Like ``run_backtest.py`` this reaches the
network (yfinance and FRED, cached under ``data/``) and re-runs the whole
backtest each time -- nothing caches the strategy between invocations. The
ticker is cached in its own file, so this never rewrites the shared price cache
the engine reads. ``--refresh-prices`` forces a re-download. No API keys.

**This is an overlap-window comparison.** The printed report says so, and lists
the other things it is not, in its own "What this is, and is not" section every
time it runs.
"""

from __future__ import annotations

import argparse
import logging
import sys

from run_backtest import load_data
from src.backtest import run_backtest
from src.compare import compare_to_ticker, format_comparison_report
from src.config import DATA_START_DATE, LEVERAGE_CAP, TARGET_VOL, UNIVERSE
from src.data import DataUnavailableError, daily_returns, fetch_ticker_prices
from src.overlay import unlevered_returns
from src.riskparity import ERCSolverError
from src.types import ContractViolation

log = logging.getLogger("compare_ticker")


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface."""
    parser = argparse.ArgumentParser(
        prog="compare_ticker.py",
        description=(
            "Head-to-head: the risk-parity strategy against buy-and-hold of one "
            f"ticker, on the dates the ticker exists. The strategy is the "
            f"{'/'.join(UNIVERSE)} equal-risk book levered to {TARGET_VOL:.0%} vol "
            f"(cap {LEVERAGE_CAP[0]}-{LEVERAGE_CAP[1]}x) and financed at DTB3, run "
            "over its full history and then cut to the overlap window."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Data comes from yfinance and the FRED CSV endpoint -- no API keys. "
            "Prices are cached under data/; pass --refresh-prices to re-download."
        ),
    )
    parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="The ticker to compare against, e.g. AAPL or SPY. Any ticker yfinance covers",
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Bypass the on-disk price and risk-free caches and re-download",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO logging from src/ -- downloads, PSD repairs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # As in run_backtest.py: covariance.py logs one line per PSD repair, which at
    # --verbose would bury everything else. The repairs still surface in the
    # engine's run_health, which is printed below.
    logging.getLogger("src.covariance").setLevel(logging.WARNING)

    ticker = args.ticker.strip().upper()
    if not ticker:
        # argparse's required=True only guarantees --ticker was PASSED, not that
        # it was non-blank.
        print("error: --ticker cannot be empty", file=sys.stderr)
        return 2

    print(f"Comparing the risk-parity strategy against {ticker} buy-and-hold"
          f"{' (refreshing cache)' if args.refresh_prices else ''}...")
    try:
        print(f"\nLoading {'/'.join(UNIVERSE)} from {DATA_START_DATE}...")
        returns, risk_free = load_data(DATA_START_DATE, args.refresh_prices)

        print(f"Loading {ticker}...")
        prices = fetch_ticker_prices(ticker, refresh=args.refresh_prices)
        ticker_returns = daily_returns(prices.to_frame())[ticker]

        print("Running the engine over the full history (this is the slow step)...")
        engine = run_backtest(returns, risk_free=risk_free)

        # The same book with no leverage: the ticker is unlevered, so this is the
        # like-for-like risk comparison. Sliced to the engine's own dates first --
        # the engine's return series starts a warm-up after the data does.
        unlevered = unlevered_returns(returns.loc[engine.returns.index], engine.weights)

        comparison = compare_to_ticker(
            engine.returns,
            ticker_returns,
            ticker,
            unlevered=unlevered,
            strategy_name=engine.strategy,
        )
    except ERCSolverError as exc:
        # riskparity.ERCSolverError subclasses ValueError, so it has to be caught
        # BEFORE the ValueError clause below: it means no equal-risk portfolio
        # exists (a broken price series), an economic/data problem, not a
        # misconfigured command line.
        print(f"\nerror: the engine could not solve for equal risk contribution: {exc}",
              file=sys.stderr)
        return 1
    except ContractViolation as exc:
        print(f"\nerror: the engine produced a result that violates the contract: {exc}",
              file=sys.stderr)
        return 1
    except DataUnavailableError as exc:
        # data.py's policy, inherited: fail loudly rather than compare against a
        # ticker with no data, an unusable series, or too little overlap.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # A blank or otherwise malformed ticker, or a label collision.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2

    health = engine.meta.get("run_health", "unknown")
    marker = " " if health == "clean" else "!"
    print(f"\n{marker} engine run health: {health}")
    print(f"\n{format_comparison_report(comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
