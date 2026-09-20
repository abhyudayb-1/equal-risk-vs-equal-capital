#!/usr/bin/env python
"""CLI for the incremental position analysis (Phase 6, step 40).

Given the existing four-sleeve risk-parity book, what happens if one more
stock is added to it -- as of right now, not as a second 2006-to-present
backtest. See ``src/incremental.py``'s module docstring for the three pieces
of math and PLAN.md's "Phase 6" section for the full derivation; this file
owns none of that -- it parses arguments, calls ``src.incremental.analyze_position``,
and prints ``src.incremental.format_position_report``'s account of the result,
the same "wiring only" role ``run_backtest.py`` plays for the core engine.

Run it::

    .venv/bin/python analyze_position.py --ticker AAPL
    .venv/bin/python analyze_position.py --help              # every flag, documented

**Live data.** Like ``run_backtest.py``, this reaches the network -- yfinance
for the candidate and the existing universe, both cached under ``data/``.
``--refresh-prices`` forces a re-download. No API keys.

**This is a snapshot, not a backtest.** The printed report says so in its own
"What this is, and is not" section every time it runs -- see PLAN.md's
"Explicit caveats the tool must state every time it runs" for why that lives
in the report itself rather than only in a doc.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import INCREMENTAL_CANDIDATE_WEIGHTS, INCREMENTAL_LOOKBACK_DAYS, UNIVERSE
from src.data import DataUnavailableError
from src.incremental import analyze_position, format_position_report

log = logging.getLogger("analyze_position")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _weight_list(text: str) -> tuple[float, ...]:
    """Comma-separated candidate weights, e.g. ``0.01,0.05,0.10``.

    Validated at parse time rather than after a live fetch, matching
    ``run_backtest.py``'s ``_iso_date``/``_positive_float`` -- a bad flag
    should fail before the network is touched, not after.
    """
    weights = []
    for piece in text.split(","):
        piece = piece.strip()
        try:
            value = float(piece)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{piece!r} is not a number") from exc
        if not 0.0 < value < 1.0:
            raise argparse.ArgumentTypeError(
                f"each weight must be strictly between 0 and 1, got {value}"
            )
        weights.append(value)
    if not weights:
        raise argparse.ArgumentTypeError("--weights needs at least one value")
    return tuple(weights)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface. Step 40's "done when" is that ``--help`` documents it all."""
    parser = argparse.ArgumentParser(
        prog="analyze_position.py",
        description=(
            "Incremental position analysis: what happens if one more stock is "
            f"added to the existing {'/'.join(UNIVERSE)} risk-parity book, as a "
            "point-in-time snapshot rather than a historical backtest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Data comes from yfinance -- no API keys. Prices are cached under "
            "data/; pass --refresh-prices to re-download."
        ),
    )

    parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="The candidate stock to analyze, e.g. AAPL. Any ticker yfinance covers",
    )
    parser.add_argument(
        "--lookback-days",
        type=_positive_int,
        default=INCREMENTAL_LOOKBACK_DAYS,
        metavar="N",
        help=(
            "Target trailing window, in trading days, ending today. If the "
            "candidate has less history than this (a recent IPO), the window "
            "shrinks to what it has, down to INCREMENTAL_MIN_HISTORY_DAYS"
        ),
    )
    parser.add_argument(
        "--weights",
        type=_weight_list,
        default=INCREMENTAL_CANDIDATE_WEIGHTS,
        metavar="W1,W2,...",
        help=(
            "Comma-separated candidate weights for the perturbation scan, each "
            "strictly between 0 and 1"
        ),
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Bypass the on-disk price cache and re-download",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO logging from src/ -- downloads, the min-history retry, PSD repairs",
    )
    return parser


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ticker = args.ticker.strip().upper()
    print(
        f"Analyzing {ticker} against the {'/'.join(UNIVERSE)} book "
        f"(target lookback {args.lookback_days} trading days)"
        f"{' (refreshing cache)' if args.refresh_prices else ''}..."
    )
    try:
        report = analyze_position(
            ticker,
            lookback_days=args.lookback_days,
            candidate_weights=args.weights,
            refresh=args.refresh_prices,
        )
    except ValueError as exc:
        # candidate already in the universe, or min_history_days > lookback_days
        # -- both are misconfigurations, not data problems.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except DataUnavailableError as exc:
        # data.py's policy, inherited: fail loudly rather than silently
        # analyzing a candidate with too little history to trust.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print(f"\n{format_position_report(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
