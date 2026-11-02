#!/usr/bin/env python
"""The end-to-end run: data -> engine -> benchmarks -> stress -> charts (step 30).

Every module in ``src/`` has been buildable and testable on its own; this is the
one place they are wired together, and it is deliberately the *only* place. It
owns no maths. If a number here is wrong, the bug is upstream in the module that
produced it, not in this file -- which is why nothing below computes anything it
could have asked a module for.

Run it::

    .venv/bin/python run_backtest.py              # macOS/Linux (Person B)
    .\\.venv\\Scripts\\python.exe run_backtest.py   # Windows (Person A)
    .venv/bin/python run_backtest.py --help       # every flag, documented

**Live data.** Unlike the test suites, this reaches the network: yfinance for
prices and the FRED CSV endpoint for ``DTB3``. Both are cached under ``data/``,
so the second run of a session is offline and fast; ``--refresh-prices`` forces
a re-download. There are no API keys anywhere in this project and there must not
be -- anyone who clones the repo can run this command.

**Three decisions this file makes, because a call site is where they belong:**

1. ``cov_estimator=estimate_covariance`` is passed to all four benchmarks
   explicitly, even though ``src/benchmarks.py`` now defaults to the same
   estimator itself (the Phase 1 stand-ins it used before ``src/covariance.py``
   existed are retired -- see CLAUDE.md, "Two stand-ins"). Passing it here too is
   not redundant: it is what makes it unmistakable, at the one place a reader
   would check, that the benchmarks are judged on **exactly** the covariance the
   engine uses. A strategy comparison where the comparators saw a different
   covariance than the engine is not a comparison of strategies.
2. The engine is levered and every benchmark is not. That asymmetry is real --
   it is the strategy being tested -- and it is why ``mean_leverage`` is a
   column of the crisis table rather than a footnote. Nothing here hides it.
3. ``--max-leverage`` replaces the *upper* half of ``config.LEVERAGE_CAP`` and
   leaves the floor alone. The floor is not a risk control (it stops the overlay
   deleveraging to nothing in a calm market); the ceiling is, and it is the one
   worth sweeping.

**What this prints is a run summary, not a finding.** Step 29 adds the solver and
conditioning diagnostics to ``meta`` and ``src.backtest.format_run_summary`` renders
them into the curated block printed right after the engine runs -- a bad rebalance
is marked with a leading ``!`` there rather than being one number among twenty.
``_print_meta`` below it is the generic, complete dump of the same ``meta`` dict,
read key-by-key so it picks up new fields without an edit here; the two blocks
answer different questions ("can this run be trusted" vs. "what exactly is in
here") and both are printed rather than one replacing the other. Nothing that
lands in the README comes from a single invocation of this script without being
checked against the charts and the crisis table it writes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.backtest import format_run_summary, run_backtest
from src.benchmarks import equal_weight, max_sharpe, naive_risk_parity, sixty_forty
from src.config import (
    DATA_START_DATE,
    LEVERAGE_CAP,
    REPORTS_DIR,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from src.covariance import estimate_covariance
from src.data import (
    DataUnavailableError,
    align_risk_free,
    daily_returns,
    fetch_prices,
    fetch_risk_free,
)
from src.metrics import summarize
from src.stress import (
    correlation_regime_table,
    crisis_decomposition,
    drawdown_table,
    regime_summary,
    write_crisis_table,
)
from src.riskparity import ERCSolverError
from src.types import ContractViolation
from src.viz import render_all

log = logging.getLogger("run_backtest")

ENGINE_LABEL = "Risk parity + vol target"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _iso_date(text: str) -> str:
    """Validate a date at parse time rather than 40 seconds into a download."""
    try:
        stamp = pd.Timestamp(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date: {exc}") from exc
    if pd.isna(stamp):
        raise argparse.ArgumentTypeError(f"{text!r} is not a date")
    return stamp.strftime("%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface. Step 30's "done when" is that ``--help`` documents it all."""
    parser = argparse.ArgumentParser(
        prog="run_backtest.py",
        description=(
            "Risk-parity + volatility-targeting backtest over "
            f"{'/'.join(UNIVERSE)}, against four unlevered benchmarks. "
            "Writes the crisis table and four charts to reports/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Data comes from yfinance and the FRED CSV endpoint -- no API keys. "
            "Prices are cached under data/; pass --refresh-prices to re-download."
        ),
    )

    parser.add_argument(
        "--start",
        type=_iso_date,
        default=DATA_START_DATE,
        metavar="YYYY-MM-DD",
        help=(
            "First date of price history. The default is DBC's inception -- the "
            "brief asked for 2005 and the data cannot deliver it. Note the first "
            "rebalance is a further ~252 trading days in, since the covariance "
            "needs a year of warm-up"
        ),
    )
    parser.add_argument(
        "--target-vol",
        type=_positive_float,
        default=TARGET_VOL,
        metavar="SIGMA",
        help="Annualized volatility the overlay targets, as a decimal (0.10 = 10%%)",
    )
    parser.add_argument(
        "--max-leverage",
        type=_positive_float,
        default=LEVERAGE_CAP[1],
        metavar="L",
        help=(
            "Upper leverage cap. Replaces the ceiling of config.LEVERAGE_CAP; the "
            f"floor stays at {LEVERAGE_CAP[0]}. Leverage is financed at DTB3, so "
            "raising this raises the borrowing cost too, not just the exposure"
        ),
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help=(
            "Bypass the on-disk price and risk-free caches and re-download. Use "
            "after a trading day closes, or if a cached pull looks wrong"
        ),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        metavar="DIR",
        help="Where the crisis table and the four charts are written",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help=(
            "Skip chart rendering. The tables still get written -- matplotlib is "
            "the slow part of the run and a parameter sweep does not need it"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO logging from src/ -- downloads, PSD repairs, skipped windows",
    )
    return parser


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def load_data(start: str, refresh: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Daily sleeve returns and the per-day financing rate, on one calendar.

    The risk-free series is fetched over the *price* window and then aligned to
    the return index, so financing is charged on exactly the days the portfolio
    earns returns. Fetching it independently and letting pandas align later is
    how a backtest ends up charging leverage on days it did not hold anything.
    """
    prices = fetch_prices(start=start, refresh=refresh)
    returns = daily_returns(prices)

    risk_free = fetch_risk_free(start=start, refresh=refresh)
    aligned = align_risk_free(risk_free, returns.index)
    return returns, aligned


def build_benchmarks(returns: pd.DataFrame) -> dict[str, object]:
    """The four comparators, all unlevered, all on the engine's covariance.

    ``cov_estimator=estimate_covariance`` is passed explicitly here (see this
    module's docstring) even though it is also ``src.benchmarks``' own default
    now. It matters most for ``max_sharpe``: the real estimator repairs a
    non-PSD matrix in *correlation* space and leaves the asset volatilities
    intact, where the retired stand-in clipped the covariance's eigenvalues
    directly and quietly shifted them. A tangency portfolio is about as
    sensitive to those volatilities as a portfolio gets.
    """
    return {
        "Equal weight": equal_weight(returns, cov_estimator=estimate_covariance),
        "60/40": sixty_forty(returns, cov_estimator=estimate_covariance),
        "Naive 1/vol": naive_risk_parity(returns, cov_estimator=estimate_covariance),
        "Max Sharpe": max_sharpe(returns, cov_estimator=estimate_covariance),
    }


def _summary_frame(results: dict, periods_per_year: int = TRADING_DAYS_PER_YEAR):
    rows = []
    for name, result in results.items():
        row = summarize(result.returns, periods_per_year=periods_per_year, name=name)
        row["mean_leverage"] = float(result.leverage.mean())
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("name")
    return frame[
        [
            "total_return",
            "cagr",
            "annual_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "mean_leverage",
            "n_periods",
        ]
    ]


def _print_meta(result) -> None:
    """Whatever the engine chose to surface, printed rather than left in the object.

    Read generically on purpose: step 29 adds fields to ``meta`` and this block
    should pick them up without an edit here. A failed rebalance that only exists
    as a key nobody prints is a silent failure with extra steps.
    """
    skip = {"strategy"}
    print("\nEngine diagnostics (meta)")
    for key, value in result.meta.items():
        if key in skip:
            continue
        if isinstance(value, pd.DataFrame):
            print(f"  {key:28s} {len(value)} rows x {len(value.columns)} cols")
        elif isinstance(value, (pd.Series, list, tuple, dict)):
            print(f"  {key:28s} {len(value)} entries")
        else:
            print(f"  {key:28s} {value}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # The diversification-ratio panel estimates ~1000 covariances to draw itself,
    # and covariance.py logs one line per PSD repair. At --verbose that buries
    # everything else; the repairs still surface in the engine's meta, which is
    # where the run summary reads them from anyway.
    logging.getLogger("src.covariance").setLevel(logging.WARNING)

    leverage_cap = (LEVERAGE_CAP[0], args.max_leverage)
    if args.max_leverage <= LEVERAGE_CAP[0]:
        print(
            f"error: --max-leverage {args.max_leverage} is at or below the "
            f"leverage floor {LEVERAGE_CAP[0]}, which leaves the overlay nothing "
            "to choose between.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {'/'.join(UNIVERSE)} from {args.start}"
          f"{' (refreshing cache)' if args.refresh_prices else ''}...")
    try:
        returns, risk_free = load_data(args.start, args.refresh_prices)
    except DataUnavailableError as exc:
        # data.py raises rather than degrading, and this is where that pays off:
        # a silently dropped sleeve produces a plausible equity curve and an
        # entirely wrong conclusion. Nothing downstream runs.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print(f"  {len(returns)} trading days, {returns.index[0].date()} to "
          f"{returns.index[-1].date()}")
    print(f"  financing rate: DTB3, mean {risk_free.mean() * TRADING_DAYS_PER_YEAR:.2%}/yr")

    print(f"\nRunning the engine (target vol {args.target_vol:.0%}, "
          f"leverage capped at {leverage_cap[0]}-{leverage_cap[1]}x)...")
    try:
        engine = run_backtest(
            returns,
            risk_free=risk_free,
            target_vol=args.target_vol,
            leverage_cap=leverage_cap,
        )
    except ERCSolverError as exc:
        # Economic, not numerical: no equal-risk portfolio exists above the
        # weight floor (a sleeve's volatility collapsed, or the covariance
        # isn't PSD). backtest.py deliberately lets this propagate rather
        # than catching it -- see CLAUDE.md's "two solver failures are
        # different events" -- so this is where it has to stop being a raw
        # traceback and become a reported error.
        print(f"\nerror: the engine could not solve for equal risk contribution: {exc}",
              file=sys.stderr)
        return 1
    except ContractViolation as exc:
        # validate() is deliberately loud: a malformed result is the one failure
        # that lets the engine and the analysis layer diverge unnoticed.
        print(f"\nerror: the engine produced a result that violates the contract: {exc}",
              file=sys.stderr)
        return 1

    print(f"\n{format_run_summary(engine)}")

    print("\nRunning the benchmarks (unlevered)...")
    results = {ENGINE_LABEL: engine, **build_benchmarks(returns)}

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nFull sample")
        print(_summary_frame(results).to_string(float_format=lambda v: f"{v:>10.4f}"))

        crisis = crisis_decomposition(results)
        print("\nCrisis windows -- max drawdown")
        print(drawdown_table(results).to_string(float_format=lambda v: f"{v:>10.2%}"))
        print("\nCrisis windows -- total return")
        print(drawdown_table(results, value="total_return").to_string(
            float_format=lambda v: f"{v:>10.2%}"))

        print("\nRegime diagnostics -- correlation says one thing, "
              "the diversification ratio another")
        print(regime_summary(returns, weights=engine.weights).to_string(
            float_format=lambda v: f"{v:>8.3f}"))
        print("\nSPY/TLT rolling correlation by year")
        print(correlation_regime_table(returns).to_string(
            float_format=lambda v: f"{v:>8.3f}"))

    _print_meta(engine)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    written = [write_crisis_table(crisis, args.reports_dir / "crisis_drawdowns.csv")]
    if args.no_charts:
        print("\nCharts skipped (--no-charts).")
    else:
        print("\nRendering charts...")
        written.extend(render_all(engine, returns, reports_dir=args.reports_dir).values())

    print(f"\nWrote {len(written)} files to {args.reports_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
