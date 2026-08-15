"""The report charts (steps 26-27).

Four figures, and each one exists to make a specific claim visible rather than to
decorate the README:

1. :func:`plot_risk_contributions` -- the risk-contribution stacked area. A flat
   set of 1/N bands is what a *working* ERC solve looks like, so this chart is
   read for where it stops being flat.
2. :func:`plot_capital_vs_risk_weights` -- capital weight beside risk weight, the
   project's thesis in one panel. Equal capital is not equal risk, and the gap
   between the two panels is the entire reason the solver exists.
3. :func:`plot_leverage` -- the overlay multiplier through time, with the crisis
   windows shaded. Trailing volatility is a lagging estimator, so the chart to
   look for is leverage still elevated *into* a drawdown and cut afterwards.
4. :func:`plot_correlation_and_diversification` -- the rolling SPY/TLT
   correlation and the diversification ratio on one time axis. This is the pair
   that carries the 2022 finding: the correlation is still negative through the
   drawdown and only flips in 2023, while the ratio is already falling.

Style helpers
-------------
``_new_figure`` / ``_style_axes`` / ``_title`` / ``_save`` and ``PALETTE`` follow
the house style from ``abhyudayb-1/sp500-factor-backtester``. That repo is not
checked out on this machine, so these are written to the documented shape rather
than copied line for line -- the four-helper split, the muted palette, the
de-chromed axes, the title-plus-subtitle convention. If the original is available
when this is reviewed, reconcile against it rather than keeping two dialects.

Every plotting function takes an explicit ``path`` and returns the ``Path`` it
wrote, and none of them calls ``plt.show()``: this runs headless from
``run_backtest.py`` (step 30). Colours are assigned **per ticker**, not per axis
position, so a sleeve is the same colour in every figure -- a chart set where SPY
changes colour between panels is actively misleading.

Run a self-check from the repo root::

    .venv/bin/python -m src.viz
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Headless by default: this is imported by a CLI that runs on two machines and in
# CI, and the interactive backends fail differently on each. Set before pyplot is
# imported, which is the only point at which it takes effect.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from .config import (  # noqa: E402
    CORRELATION_PAIR,
    CORRELATION_WINDOW_DAYS,
    CRISIS_WINDOWS,
    LEVERAGE_CAP,
    REPORTS_DIR,
    SLEEVE_LABELS,
    TARGET_VOL,
)
from .stress import rolling_correlation, rolling_diversification_ratio  # noqa: E402
from .types import BacktestResult  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
# Muted, and distinguishable in greyscale as well as in colour -- these end up
# printed in a submitted report at least once. Ordered so that adjacent series in
# a stacked area never share a hue.
PALETTE: tuple[str, ...] = (
    "#3B6EA8",  # blue
    "#C06C3E",  # burnt orange
    "#5B8C5A",  # green
    "#8B6BA8",  # violet
    "#B8A045",  # ochre
    "#9E5C63",  # muted red
)

# Everything that is not data: axis spines, gridlines, annotation text.
INK = "#2B2B2B"
MUTED = "#6E6E6E"
GRID = "#D8D8D8"

# Crisis shading. Light enough to sit under a line without competing with it.
CRISIS_SHADE = "#B0B0B0"
CRISIS_ALPHA = 0.18

FIGURE_DPI = 150
DEFAULT_FIGSIZE = (11.0, 5.5)


def _colour_map(tickers: Sequence[str]) -> dict[str, str]:
    """Assign a stable colour per ticker.

    Keyed by ticker rather than by position so a sleeve keeps its colour across
    every figure even when a strategy holds a different subset -- 60/40 has no
    DBC column, and without this its TIP band would inherit DBC's colour from the
    engine's chart.
    """
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(tickers)}


def _label(ticker: str) -> str:
    """``SPY (Equities)`` -- the ticker plus what it stands for.

    Both, because the audience for this report is split: the ticker is what the
    code and the data vendor use, the sleeve name is what the strategy is
    actually about, and a legend with only one of them makes half the readers
    translate in their heads.
    """
    sleeve = SLEEVE_LABELS.get(ticker)
    return f"{ticker} ({sleeve})" if sleeve else ticker


def _new_figure(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
):
    """A figure and axes with the house defaults already applied."""
    figure, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, dpi=FIGURE_DPI, **kwargs
    )
    figure.patch.set_facecolor("white")
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        _style_axes(ax)
    return figure, axes


def _style_axes(ax, ylabel: str = "", xlabel: str = "") -> None:
    """De-chrome an axis: no top/right spines, a soft horizontal grid, muted ink.

    The horizontal-only grid is deliberate. Every chart in this report is a time
    series read for its *level* -- a leverage of 3.0, a risk share of 1/N, a
    correlation of zero -- and vertical gridlines add ink without helping anyone
    read a level off the y-axis.
    """
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)


def _title(ax, title: str, subtitle: str = "") -> None:
    """Left-aligned title with an optional explanatory subtitle beneath it.

    The subtitle is where the chart says what it is *for*. A reader who takes one
    figure out of the report and puts it in a slide should still be told what
    they are looking at, and a caption that lives in the README does not travel
    with the PNG.
    """
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=18 if subtitle else 10)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.02),
            xycoords="axes fraction",
            color=MUTED,
            fontsize=9.5,
            va="bottom",
        )


def _save(figure, path: str | Path, tight: bool = True) -> Path:
    """Write a figure to ``path`` and close it. Returns the path.

    Closing matters: ``run_backtest.py`` renders every chart in one process, and
    matplotlib warns and then leaks once twenty figures are open. Parent
    directories are created rather than assumed, so a caller can write to a
    subdirectory of ``reports/`` without arranging it first.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(destination, dpi=FIGURE_DPI, facecolor=figure.get_facecolor())
    plt.close(figure)
    log.info("Wrote %s", destination)
    return destination


def _shade_crises(ax, windows: Mapping[str, tuple[str, str]] | None = None,
                  label_them: bool = True) -> None:
    """Shade the crisis windows behind whatever is already plotted.

    Drawn from ``config.CRISIS_WINDOWS`` rather than from hard-coded dates, so a
    window edited in config moves on every chart at once. Clipped to the axis's
    current x-limits, and skipped entirely where a window falls outside the data
    -- an unclipped ``axvspan`` silently stretches the x-axis to cover a crisis
    the series does not reach, which reads as missing data.
    """
    left, right = ax.get_xlim()
    for label, (start, end) in (windows or CRISIS_WINDOWS).items():
        start_num = matplotlib.dates.date2num(pd.Timestamp(start))
        end_num = matplotlib.dates.date2num(pd.Timestamp(end))
        if end_num < left or start_num > right:
            continue
        ax.axvspan(
            max(start_num, left), min(end_num, right),
            color=CRISIS_SHADE, alpha=CRISIS_ALPHA, linewidth=0, zorder=0,
        )
        if label_them:
            ax.annotate(
                label,
                xy=(max(start_num, left), 0.985),
                xycoords=("data", "axes fraction"),
                color=MUTED, fontsize=8, va="top", ha="left", rotation=90,
            )
    ax.set_xlim(left, right)


# ---------------------------------------------------------------------------
# Step 26 -- the risk contribution stacked area
# ---------------------------------------------------------------------------
def plot_risk_contributions(
    result: BacktestResult,
    path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Stacked area of each sleeve's share of portfolio risk through time.

    The chart to read for *flatness*. Rows of ``risk_contributions`` sum to 1.0
    by contract, so the bands always fill the panel; what carries information is
    whether each band holds its 1/N width. A converged ERC solve draws four
    horizontal ribbons, and the 1/N reference lines are drawn on top so the eye
    has something to measure against rather than having to judge band widths by
    area.

    Where a band wanders off 1/N, one of two things happened: the solver did not
    converge at that rebalance (check ``meta['solver_failures']``), or the
    strategy is not an ERC strategy at all -- run this on the equal-weight
    benchmark and the equity band swells to roughly half the risk on a quarter of
    the capital, which is the comparison the whole project is built on.
    """
    rc = result.risk_contributions
    if rc.empty:
        raise ValueError("cannot plot risk contributions from an empty frame")

    tickers = list(rc.columns)
    colours = _colour_map(tickers)
    n = len(tickers)

    figure, ax = _new_figure()
    ax.stackplot(
        rc.index,
        [rc[t].to_numpy() for t in tickers],
        labels=[_label(t) for t in tickers],
        colors=[colours[t] for t in tickers],
        edgecolor="white",
        linewidth=0.4,
    )

    # The 1/N ladder: the cumulative levels an equal-risk portfolio's bands land
    # on. Dashed white over the fill so they read as reference rather than data.
    for k in range(1, n):
        ax.axhline(k / n, color="white", linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.85)

    _style_axes(ax, ylabel="Share of portfolio volatility")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(rc.index[0], rc.index[-1])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(False)  # the fill is the chart; gridlines would sit under solid colour
    _shade_crises(ax, label_them=False)
    _title(
        ax,
        title or f"Risk contributions - {result.strategy}",
        f"Each sleeve's share of portfolio volatility. Dashed lines mark the "
        f"{1 / n:.0%} an equal-risk portfolio holds; flat bands mean the solve worked.",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(n, 4),
        frameon=False, fontsize=9, labelcolor=INK,
    )

    return _save(figure, path or REPORTS_DIR / "risk_contributions.png")


# ---------------------------------------------------------------------------
# Step 27 -- the remaining three
# ---------------------------------------------------------------------------
def plot_capital_vs_risk_weights(
    result: BacktestResult,
    path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Capital weight and risk weight, side by side. The thesis in one figure.

    Two stacked areas sharing a y-axis: what fraction of the *money* is in each
    sleeve, and what fraction of the *volatility*. For an ERC portfolio the right
    panel is flat by construction and the left one is emphatically not -- the
    low-volatility sleeves take a large majority of the capital to contribute
    their 1/N of the risk. That asymmetry is what the 2022 section is about: the
    capital weight is what determines the loss when a sleeve simply falls, and
    risk parity's is concentrated in bonds.

    The mean weights are printed in the legend, because the eye is poor at
    integrating a band's width over twenty years and the single number is what
    ends up quoted.
    """
    weights, rc = result.weights, result.risk_contributions
    if weights.empty:
        raise ValueError("cannot plot weights from an empty frame")

    tickers = list(weights.columns)
    colours = _colour_map(tickers)

    figure, axes = _new_figure(figsize=(12.0, 5.5), nrows=1, ncols=2, sharey=True)
    panels = (
        (axes[0], weights, "Capital weight", "share of money"),
        (axes[1], rc, "Risk weight", "share of volatility"),
    )

    for ax, frame, panel_title, panel_note in panels:
        ax.stackplot(
            frame.index,
            [frame[t].to_numpy() for t in tickers],
            colors=[colours[t] for t in tickers],
            labels=[
                f"{_label(t)} - {frame[t].mean():.0%} mean" for t in tickers
            ],
            edgecolor="white",
            linewidth=0.4,
        )
        _style_axes(ax)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(frame.index[0], frame.index[-1])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(False)
        _shade_crises(ax, label_them=False)
        _title(ax, panel_title, panel_note)
        # Per-panel legends: the mean weight differs between them, and that
        # difference is the point of the figure.
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
            frameon=False, fontsize=8.5, labelcolor=INK,
        )

    figure.suptitle(
        title or f"Capital weight vs risk weight - {result.strategy}",
        x=0.01, ha="left", color=INK, fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(figure, path or REPORTS_DIR / "capital_vs_risk_weights.png", tight=False)


def plot_leverage(
    result: BacktestResult,
    path: str | Path | None = None,
    title: str | None = None,
    target_vol: float | None = None,
) -> Path:
    """The overlay multiplier through time, against its cap and the crisis windows.

    Read this one for *lag*. The overlay sets ``L_t = target_vol / sigma_hat_t``
    from a trailing volatility estimate, and a trailing estimate is by
    construction late: volatility is still low when a drawdown begins, so
    leverage is still high, and the de-levering lands after the loss rather than
    before it. Both cap lines are drawn because time spent pinned to a bound is
    time the overlay was not actually targeting anything -- at the ceiling it is
    a constant-leverage strategy wearing a vol-target label.

    A benchmark's leverage is 1.0 throughout by construction; plotting one is a
    legitimate sanity check rather than a mistake, so this does not refuse.
    """
    leverage = result.leverage
    if leverage.empty:
        raise ValueError("cannot plot an empty leverage series")

    figure, ax = _new_figure(figsize=(11.0, 4.6))
    ax.plot(leverage.index, leverage.to_numpy(), color=PALETTE[0], linewidth=1.4,
            label="Leverage $L_t$")

    floor, ceiling = LEVERAGE_CAP
    for bound, name in ((floor, "cap floor"), (ceiling, "cap ceiling")):
        ax.axhline(bound, color=MUTED, linewidth=0.9, linestyle=(0, (5, 4)))
        ax.annotate(f"{name} {bound:g}x", xy=(0.995, bound), xycoords=("axes fraction", "data"),
                    color=MUTED, fontsize=8.5, ha="right", va="bottom")
    ax.axhline(1.0, color=INK, linewidth=0.8, alpha=0.45)

    _style_axes(ax, ylabel="Leverage multiplier")
    ax.set_xlim(leverage.index[0], leverage.index[-1])
    ax.set_ylim(0.0, max(ceiling, float(leverage.max())) * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}x"))
    _shade_crises(ax)

    pinned = float((leverage >= ceiling - 1e-9).mean())
    _title(
        ax,
        title or f"Leverage - {result.strategy}",
        f"Target volatility {(target_vol if target_vol is not None else TARGET_VOL):.0%}. "
        f"At the ceiling on {pinned:.0%} of rebalances. Trailing volatility lags, so the "
        f"overlay de-levers after a drawdown rather than before it.",
    )
    return _save(figure, path or REPORTS_DIR / "leverage.png")


def plot_correlation_and_diversification(
    returns: pd.DataFrame,
    path: str | Path | None = None,
    pair: tuple[str, str] = CORRELATION_PAIR,
    window_days: int = CORRELATION_WINDOW_DAYS,
    weights: pd.DataFrame | None = None,
    title: str | None = None,
    step_days: int = 5,
) -> Path:
    """Rolling correlation above, diversification ratio below. The 2022 exhibit.

    Two panels on one time axis, and the figure's argument is the disagreement
    between them. The correlation panel is the intuitive story and it is late:
    through the 2022 drawdown the SPY/TLT correlation averaged -0.09 and was
    positive on only 31% of days, so a sign alarm does not fire until 2023, a
    year after the damage. The ratio panel is the diagnostic: it falls right
    through 2021-22 and bottoms in 2023.

    Stacked rather than twinned on one y-axis on purpose. A correlation lives in
    [-1, 1] and a diversification ratio lives in [1, inf); overlaying them on
    twin axes would put two unrelated scales on one frame and invite reading a
    crossing point that means nothing.

    ``step_days`` thins the diversification ratio, which re-estimates a
    covariance per point -- 5 is invisible at chart resolution over twenty years
    and about five times faster.
    """
    correlation = rolling_correlation(returns, pair=pair, window_days=window_days)
    ratio = rolling_diversification_ratio(returns, weights=weights, step_days=step_days)

    figure, axes = _new_figure(figsize=(11.0, 7.0), nrows=2, ncols=1, sharex=True)
    top, bottom = axes

    top.plot(correlation.index, correlation.to_numpy(), color=PALETTE[0], linewidth=1.1)
    top.axhline(0.0, color=PALETTE[5], linewidth=1.0)
    # Shade only the positive side: "the hedge has stopped being a hedge" is a
    # one-sided claim, and filling both sides would make it look like a band.
    top.fill_between(
        correlation.index, 0.0, correlation.to_numpy(),
        where=(correlation.to_numpy() > 0), color=PALETTE[5], alpha=0.25, linewidth=0,
    )
    _style_axes(top, ylabel=f"{pair[0]}/{pair[1]} correlation")
    top.set_ylim(-1.0, 1.0)
    _shade_crises(top)
    _title(
        top,
        title or f"Diversification, two ways ({window_days}-day rolling)",
        "Correlation is the intuitive picture and it is late: still negative through the "
        "2022 drawdown, positive only from 2023.",
    )

    bottom.plot(ratio.index, ratio.to_numpy(), color=PALETTE[2], linewidth=1.3)
    bottom.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (5, 4)))
    bottom.annotate(
        "1.0 = no diversification at all", xy=(0.995, 1.0),
        xycoords=("axes fraction", "data"), color=MUTED, fontsize=8.5,
        ha="right", va="bottom",
    )
    _style_axes(bottom, ylabel=r"Diversification ratio  $\sum w_i \sigma_i / \sigma_p$")
    bottom.set_xlim(ratio.index[0], ratio.index[-1])
    bottom.set_ylim(0.95, float(np.nanmax(ratio.to_numpy())) * 1.08)
    _shade_crises(bottom, label_them=False)
    _title(
        bottom,
        "",
        "The ratio is the diagnostic: it degrades while the damage is being done.",
    )

    return _save(figure, path or REPORTS_DIR / "correlation_diversification.png")


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------
def render_all(
    result: BacktestResult,
    returns: pd.DataFrame,
    reports_dir: str | Path | None = None,
    use_result_weights: bool = True,
) -> dict[str, Path]:
    """Render all four charts and return ``{name: path}``.

    What ``run_backtest.py`` (step 30) calls. ``returns`` is the daily asset
    return frame the diagnostics need -- it is not recoverable from a
    ``BacktestResult``, which carries portfolio returns rather than per-sleeve
    ones.

    ``use_result_weights`` measures the diversification ratio on the strategy's
    own allocation. Turning it off falls back to inverse-volatility weights,
    which makes the series a property of the market rather than of one strategy
    and is the form the published annual path in ``config.py`` was measured on.
    """
    directory = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    weights = result.weights if use_result_weights else None
    return {
        "risk_contributions": plot_risk_contributions(
            result, directory / "risk_contributions.png"
        ),
        "capital_vs_risk_weights": plot_capital_vs_risk_weights(
            result, directory / "capital_vs_risk_weights.png"
        ),
        "leverage": plot_leverage(result, directory / "leverage.png"),
        "correlation_diversification": plot_correlation_and_diversification(
            returns, directory / "correlation_diversification.png", weights=weights
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    # Imported here, not at module scope: src/ must never depend on tests/.
    from tests.fixtures import make_fake_result, make_fake_returns

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # src.covariance logs one INFO line per PSD repair, and the diversification
    # ratio estimates ~1000 covariances to draw one panel. Raised to WARNING for
    # the self-check only, so the four "Wrote ..." lines are visible; anything
    # that actually went wrong still prints.
    logging.getLogger("src.covariance").setLevel(logging.WARNING)

    fake_returns = make_fake_returns()
    fake_result = make_fake_result(strategy="Risk parity (fixture)")

    written = render_all(fake_result, fake_returns)
    for name, destination in written.items():
        size_kb = destination.stat().st_size / 1024
        print(f"  {name:28s} -> {destination}  ({size_kb:.0f} KB)")
    print(f"\n{len(written)} charts written from fixture data. Shape, not magnitude -- "
          f"the real ones come from the Phase 5 run (TASKS.md step 33).")
