#!/usr/bin/env python
"""Equal risk against equal capital, on live data (step 56).

``run_backtest.py`` runs *the strategy* and scores it against four unlevered
benchmarks. This runs *the study*: two construction rules over one multi-asset
book, with the volatility-matched column that makes their returns comparable at
all, and every caveat that column carries printed underneath it.

Run it::

    .venv/bin/python compare_rules.py                 # macOS/Linux (Person B)
    .\\.venv\\Scripts\\python.exe compare_rules.py      # Windows (Person A)
    .venv/bin/python compare_rules.py --help          # every flag, documented

**It owns no maths and no data path.** ``load_data`` is imported from
``run_backtest.py`` rather than reimplemented, so the prices, the financing rate
and the calendar alignment are bit-identical to the headline run -- two CLIs
that fetched their own data would eventually disagree about what "the sample"
is, and the first symptom would be two tables in the README that cannot both be
right. Everything printed comes out of ``src.comparison``; this file parses
arguments, hands them over, and writes what comes back.

**Three things it will not let you do quietly:**

1. **Mix data dates.** The window and the date the data ends are printed at the
   top of every run, because the one way to produce a wrong table here is to
   quote two rows measured on two different days.
2. **Read the matched column as exactly matched.** The target the overlay was
   given and the volatility it actually realized are printed side by side, with
   the overshoot between them. A trailing estimator cannot hit a target; saying
   so is cheaper than being asked.
3. **Take the financing-omitted rows for results.** They are marked
   counterfactual in the table and again in the caveats. The gap between each
   and its financed twin is the cost of leverage, which is one of the three
   findings -- but it is not a strategy anyone can run.

``--target-vol`` here is **not** ``run_backtest.py``'s ``--target-vol``. There,
it sets the volatility the published strategy targets. Here, it replaces 1/N's
own realized volatility as the point the matched column is levered to, which is
the knob this study has a reason to turn -- "what if the match point were 60/40's
volatility instead" is a question about the comparison. The published-target
rows stay at ``config.TARGET_VOL`` so the table keeps a fixed reference row, and
an overridden match point is announced in the report rather than left to be
inferred from a number that moved.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import pandas as pd

from run_backtest import load_data
from src.comparison import (
    EQUAL_CAPITAL,
    ERC_MATCHED_FINANCED,
    ERC_UNLEVERED,
    ComparisonError,
    compare_rules,
    format_comparison_report,
)
from src.config import (
    DATA_START_DATE,
    REPORTS_DIR,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
)
from src.data import DataUnavailableError
from src.riskparity import ERCSolverError
from src.sensitivity import (
    SensitivityError,
    cost_comparison,
    format_cost_table,
    format_sensitivity_report,
    format_turnover_table,
    sweep,
    turnover,
    turnover_table,
)
from src.types import ContractViolation
from src.viz import render_comparison

log = logging.getLogger("compare_rules")

SUMMARY_CSV = "rule_comparison.csv"
ALLOCATION_CSV = "risk_allocation.csv"
SWEEP_CSV = "sensitivity_sweep.csv"


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
    """The CLI surface. Step 56's "done when" is that ``--help`` documents it all."""
    parser = argparse.ArgumentParser(
        prog="compare_rules.py",
        description=(
            "Equal risk contribution against equal capital over "
            f"{'/'.join(UNIVERSE)}: the two rules unlevered, ERC levered to 1/N's "
            "own realized volatility with financing charged, and naive 1/vol, "
            "60/40 and max Sharpe as reference rows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Data comes from yfinance and the FRED CSV endpoint -- no API keys -- "
            "through the same load_data() run_backtest.py uses, so both CLIs see "
            "exactly one sample. Prices are cached under data/; pass "
            "--refresh-prices to re-download."
        ),
    )

    parser.add_argument(
        "--start",
        type=_iso_date,
        default=DATA_START_DATE,
        metavar="YYYY-MM-DD",
        help=(
            "First date of price history, for every rule alike. The default is "
            "DBC's inception. Note that this moves the covariance warm-up and "
            "therefore the ERC weights: a later start is a different study, not "
            "a zoomed-in view of this one"
        ),
    )
    parser.add_argument(
        "--target-vol",
        type=_positive_float,
        default=None,
        metavar="SIGMA",
        help=(
            "Override the volatility the MATCHED column is levered to, as a "
            "decimal (0.10 = 10%%). The default is 1/N's own realized volatility, "
            "which is what makes the two books comparable on return; override it "
            "to ask what the comparison looks like against some other match "
            f"point. The published-target rows stay at {TARGET_VOL * 100:.0f}%% "
            "either way. This is NOT run_backtest.py's --target-vol"
        ),
    )
    parser.add_argument(
        "--sensitivities",
        action="store_true",
        help=(
            "Also sweep the estimator: covariance halflife, shrinkage, warm-up "
            "and rebalance frequency, each across a grid bracketing its default, "
            "with ERC and 1/N re-run together at every setting. This is the "
            "answer to 'ERC needs an estimated covariance and 1/N needs nothing'. "
            "It is roughly fifty engine runs, so it is off by default; the "
            "turnover and trading-cost block is printed either way"
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
        help="Where the comparison tables and the two comparison charts are written",
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
        help=(
            "Show INFO logging from src/ -- downloads, the two engine runs, PSD "
            "repairs, the volatility the matched column was given"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def cost_block(comparison, returns: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Turnover and the cost breakeven, from the results the comparison already built.

    No engine runs: the turnover is reconstructed from the weights frames and
    the sleeve returns, and the cost sweep is arithmetic on return series. That
    is why this prints on every run and the estimator sweep does not.

    Both rules are measured on their *allocation*. The matched column's own
    leverage path is passed to its cost comparison separately, because a book
    held at 1.5x trades 1.5x the notional to move the same distance.
    """
    erc = comparison.rule_results[ERC_UNLEVERED]
    equal = comparison.rule_results[EQUAL_CAPITAL]
    traded = turnover_table({ERC_UNLEVERED: erc, EQUAL_CAPITAL: equal}, returns)
    erc_trades = turnover(erc.weights, returns)["total"]
    equal_trades = turnover(equal.weights, returns)["total"]
    costs = [
        cost_comparison(
            comparison.series[ERC_UNLEVERED], erc_trades,
            comparison.series[EQUAL_CAPITAL], equal_trades,
        ),
        # The levered row as well. The project's headline is about the matched
        # column, and charging a cost only to the unlevered one would answer
        # the easier half of the question.
        cost_comparison(
            comparison.series[ERC_MATCHED_FINANCED], erc_trades,
            comparison.series[EQUAL_CAPITAL], equal_trades,
            erc_leverage=comparison.matched_leverage,
            erc_label=ERC_MATCHED_FINANCED,
        ),
    ]
    return traded, costs


def write_tables(
    comparison, reports_dir: Path, swept: pd.DataFrame | None = None
) -> list[Path]:
    """The frames a reader would want in a spreadsheet, as CSV.

    The printed report is the deliverable; these exist so a number in the README
    can be checked against a file rather than against a screenshot. Written even
    under ``--no-charts``, which is the same split ``run_backtest.py`` makes.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = reports_dir / SUMMARY_CSV
    allocation = reports_dir / ALLOCATION_CSV
    comparison.summary.to_csv(summary)
    comparison.allocation.to_csv(allocation)
    written = [summary, allocation]
    if swept is not None:
        sweep_path = reports_dir / SWEEP_CSV
        swept.to_csv(sweep_path, index=False)
        written.append(sweep_path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # covariance.py logs one line per PSD repair and the comparison estimates a
    # covariance at every rebalance for six rules. At --verbose that buries the
    # two lines worth reading; the repairs still surface in each result's meta,
    # which is where the run summary reads them from anyway.
    logging.getLogger("src.covariance").setLevel(logging.WARNING)

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
    if args.target_vol is not None:
        print(f"  match point overridden to {args.target_vol:.2%} "
              "(default: 1/N's own realized volatility)")

    print("\nRunning both rules over the full history "
          "(two engine runs and four benchmarks)...")
    try:
        comparison = compare_rules(returns, risk_free, match_vol=args.target_vol)
    except ERCSolverError as exc:
        # Caught BEFORE ComparisonError, and both before any bare ValueError:
        # ERCSolverError subclasses ValueError, so the broader handler placed
        # first would report a genuine economic failure -- a sleeve whose
        # volatility collapsed, i.e. a broken price series -- as if the caller
        # had mis-configured something. See CLAUDE.md, "two failure modes".
        print(f"\nerror: the engine could not solve for equal risk contribution: {exc}",
              file=sys.stderr)
        return 1
    except ComparisonError as exc:
        # The inputs cannot produce a FAIR comparison: mismatched calendars, a
        # financing series with a gap, a benchmark with no volatility to match,
        # or the one that invalidates the whole construction -- the second
        # engine run coming back with different ERC weights.
        print(f"\nerror: these inputs cannot produce a fair comparison: {exc}",
              file=sys.stderr)
        return 1
    except ContractViolation as exc:
        # validate() is deliberately loud: a malformed result is the one failure
        # that lets the engine and the analysis layer diverge unnoticed.
        print(f"\nerror: a rule produced a result that violates the contract: {exc}",
              file=sys.stderr)
        return 1

    print()
    print(format_comparison_report(comparison))

    traded, costs = cost_block(comparison, returns)
    if args.sensitivities:
        print("\nSweeping the estimator (about fifty engine runs; this is the slow part)...")
        try:
            swept = sweep(returns, risk_free)
        except (SensitivityError, ComparisonError) as exc:
            print(f"\nerror: the estimator sweep could not be completed: {exc}",
                  file=sys.stderr)
            return 1
    else:
        swept = None

    print()
    if swept is None:
        # The turnover and cost half of step 58, which needs no extra engine
        # runs. The knob sweep's own header is inside format_sensitivity_report,
        # so this branch supplies its own.
        print("What each rule trades, and what that would cost")
        print(textwrap.indent(format_turnover_table(traded), "  "))
        for comparison_at_cost in costs:
            print(f"\n  {comparison_at_cost.erc_label} against "
                  f"{comparison_at_cost.benchmark_label}, both charged the same rate")
            print(textwrap.indent(format_cost_table(comparison_at_cost), "  "))
        print("\n  (--sensitivities also sweeps the estimator itself.)")
    else:
        print(format_sensitivity_report(swept, traded, costs))

    written = write_tables(comparison, args.reports_dir, swept)
    if args.no_charts:
        print("\nCharts skipped (--no-charts).")
    else:
        print("\nRendering the comparison charts...")
        written.extend(render_comparison(comparison, reports_dir=args.reports_dir).values())

    print(f"\nWrote {len(written)} files to {args.reports_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
