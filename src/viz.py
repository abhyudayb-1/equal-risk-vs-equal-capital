"""The report charts.

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
``run_backtest.py``. Colours are assigned **per ticker**, not per axis
position, so a sleeve is the same colour in every figure -- a chart set where SPY
changes colour between panels is actively misleading.

Run a self-check from the repo root::

    .venv/bin/python -m src.viz
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Headless by default: this is imported by a CLI that runs on two machines and in
# CI, and the interactive backends fail differently on each. Set before pyplot is
# imported, which is the only point at which it takes effect.
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
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

# The return/volatility plane. Iso-ratio rays every 0.10 of
# return-to-volatility: 0.05 was tried first and drew eleven near-parallel grey
# lines across a chart with nine points on it, which is texture rather than a
# reference. Label geometry is in fractions of the axis so it does not have to
# know what the units are -- a label sits LABEL_OFFSET_FRACTION of the width
# from its marker, flips to the marker's left once the marker is past
# LABEL_FLIP_FRACTION of the width, and keeps at least LABEL_MIN_GAP of the
# height from its neighbour.
ISO_RATIO_STEP = 0.10
LABEL_OFFSET_FRACTION = 0.018
LABEL_FLIP_FRACTION = 0.58
LABEL_MIN_GAP = 0.058

# ``src.comparison``'s row labels, resolved on demand rather than copied here.
# Three reasons the import is inside the functions: ``src.comparison`` pulls in
# the whole engine and ``src.viz`` is imported by callers that only want a
# chart; ``src.comparison`` stays free to import ``src.viz`` later without a
# cycle; and a label copied as a literal here would silently stop matching the
# day the label is edited there. ``render_comparison`` raises rather than
# rendering an empty figure if a name stops resolving.


def comparison_panel_rules() -> tuple[str, ...]:
    """The rules that get a risk-share panel, in the order a reader meets them.

    The two protagonists, equal risk first: the flat panel, then the one that is
    supposed to look like it and does not.
    """
    from .comparison import EQUAL_CAPITAL, ERC_UNLEVERED

    return (ERC_UNLEVERED, EQUAL_CAPITAL)


def comparison_counterfactual() -> frozenset[str]:
    """The row names that are not strategies anyone can run."""
    from .comparison import COUNTERFACTUAL

    return COUNTERFACTUAL


def comparison_lever_path() -> tuple[str, str]:
    """The unlevered row and its matched twin, for the arrow on the plane chart."""
    from .comparison import ERC_MATCHED_FINANCED, ERC_UNLEVERED

    return (ERC_UNLEVERED, ERC_MATCHED_FINANCED)


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
        # mdates is imported explicitly rather than reached through the
        # `matplotlib` package object: `import matplotlib` does not bind the
        # `dates` submodule, and this worked only because pyplot's own import
        # chain happens to pull it in. That is a dependency on someone else's
        # internals, and it breaks silently the day it changes.
        start_num = mdates.date2num(pd.Timestamp(start))
        end_num = mdates.date2num(pd.Timestamp(end))
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


def _mark_crises_over_fill(ax, windows: Mapping[str, tuple[str, str]] | None = None) -> None:
    """Mark the crisis windows on top of a filled chart, as edges rather than a wash.

    ``_shade_crises`` draws an ``axvspan`` at ``zorder=0``, which an opaque
    stackplot covers completely -- on a stacked-area panel it is ink that never
    reaches the page. This draws the window *boundaries* over the fill instead:
    two hairlines and a label, in white, which reads against every colour in the
    palette without tinting the bands underneath. The windows still come from
    ``config.CRISIS_WINDOWS``, so one edit moves them everywhere.
    """
    left, right = ax.get_xlim()
    for label, (start, end) in (windows or CRISIS_WINDOWS).items():
        start_num = mdates.date2num(pd.Timestamp(start))
        end_num = mdates.date2num(pd.Timestamp(end))
        if end_num < left or start_num > right:
            continue
        for edge in (start_num, end_num):
            if left <= edge <= right:
                ax.axvline(edge, color="white", linewidth=0.9, alpha=0.8,
                           linestyle=(0, (3, 2)), zorder=2)
        ax.annotate(
            label, xy=(max(start_num, left), 0.985), xycoords=("data", "axes fraction"),
            color="white", fontsize=8, va="top", ha="left", rotation=90, zorder=3,
        )
    ax.set_xlim(left, right)


# ---------------------------------------------------------------------------
# The risk contribution stacked area
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
# The remaining three
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
    # Set explicitly, matching every other _shade_crises call site in this
    # file, and BEFORE _shade_crises reads it: _shade_crises clips against
    # ax.get_xlim() at call time, and the two panels share an x-axis
    # (sharex=True), so leaving this on matplotlib's transient autoscaled
    # range let bottom's LATER set_xlim(ratio...) retroactively change what
    # top's shading had already been clipped against. Invisible today only
    # because correlation's and ratio's date ranges happen to nearly match.
    top.set_xlim(correlation.index[0], correlation.index[-1])
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
# The two comparison charts
# ---------------------------------------------------------------------------
# The four charts above all describe ONE strategy. These two describe the
# comparison: where two rules put their risk, and where every rule lands on the
# return/volatility plane. They take plain frames rather than a RuleComparison
# so that neither is tied to the shape of one dataclass -- ``render_comparison``
# at the bottom is the adapter that pulls them out of one.


def plot_rule_risk_shares(
    risk_shares: Mapping[str, pd.DataFrame],
    path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Each rule's risk shares through time, one panel per rule, on one y-axis.

    :func:`plot_risk_contributions` is this chart with one panel. Putting the
    rules side by side is what makes the comparison visible rather than
    inferable: equal risk draws four flat ribbons at 1/N, and equal capital
    draws the same four sleeves wandering between 9% and 60% of the risk -- on
    the same scale, in the same colours, over the same crisis shading. Read it
    for *which panel is flat*.

    Each panel's subtitle carries that rule's mean effective bets, because the
    number is the one-line version of what the panel shows and the eye cannot
    integrate a band's width over twenty years.

    ``risk_shares`` maps a rule's name to its ``risk_contributions`` frame --
    rows summing to 1, one column per sleeve. Panels are drawn in the order the
    mapping gives them, so the caller controls which rule a reader meets first.
    """
    if not risk_shares:
        raise ValueError("cannot plot risk shares for no rules at all")
    for name, frame in risk_shares.items():
        if frame.empty:
            raise ValueError(f"{name}'s risk-share frame is empty")

    # Colours keyed on the union of the sleeves, in first-seen order, so a rule
    # holding a subset (60/40 has no DBC) cannot inherit another sleeve's colour
    # -- the same reason _colour_map is keyed by ticker everywhere else.
    sleeves: list[str] = []
    for frame in risk_shares.values():
        sleeves.extend(t for t in frame.columns if t not in sleeves)
    colours = _colour_map(sleeves)

    n_panels = len(risk_shares)
    figure, axes = _new_figure(
        figsize=(6.2 * n_panels, 5.4), nrows=1, ncols=n_panels, sharey=True
    )
    panels = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()

    for ax, (name, frame) in zip(panels, risk_shares.items()):
        tickers = list(frame.columns)
        n = len(tickers)
        ax.stackplot(
            frame.index,
            [frame[t].to_numpy() for t in tickers],
            colors=[colours[t] for t in tickers],
            labels=[f"{_label(t)} - {frame[t].mean():.0%} mean" for t in tickers],
            edgecolor="white",
            linewidth=0.4,
        )
        # The 1/N ladder, as in plot_risk_contributions: the levels an
        # equal-RISK book's bands land on. Drawn on both panels on purpose -- it
        # is the reference the equal-capital panel is being read against.
        for k in range(1, n):
            ax.axhline(k / n, color="white", linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.85)

        _style_axes(ax, ylabel="Share of portfolio volatility")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(frame.index[0], frame.index[-1])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(False)
        _mark_crises_over_fill(ax)

        # Computed here rather than imported from src.comparison: this chart
        # must not depend on the module that will import it back.
        bets = float((1.0 / (frame.astype("float64") ** 2).sum(axis=1)).mean())
        _title(ax, name, f"{bets:.2f} of {n:.2f} effective bets on average")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
            frameon=False, fontsize=8.5, labelcolor=INK,
        )

    figure.suptitle(
        title or "Where each rule puts its risk",
        x=0.01, ha="left", color=INK, fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, path or REPORTS_DIR / "rule_risk_shares.png", tight=False)


def _spread_labels(
    ax,
    points: Sequence[tuple[float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float, str]]:
    """Label anchors for a scatter whose points cluster, as ``(x, y, ha)``.

    Returns one anchor per point, in the order given. Points in the right-hand
    part of the axes get their label to the *left* so it cannot run off the
    edge; the rest get it to the right. Within each side, labels are sorted by
    height and pushed apart to at least ``LABEL_MIN_GAP`` of the axis height,
    from the bottom up and then from the top down, so a cluster fans out
    instead of overprinting.

    Deliberately simple rather than a full label-placement solver: the chart it
    serves has fewer than a dozen points, an anchor is always joined to its
    marker by a leader line, and a reader can follow a leader. What it must not
    do is move the *markers*, which is the one thing that would be dishonest.
    """
    x0, x1 = xlim
    y0, y1 = ylim
    width = max(x1 - x0, 1e-12)
    height = max(y1 - y0, 1e-12)
    offset = LABEL_OFFSET_FRACTION * width

    anchors: list[tuple[float, float, str]] = [(0.0, 0.0, "left")] * len(points)
    for on_right in (False, True):
        group = [
            index
            for index, (x, _) in enumerate(points)
            if ((x - x0) / width > LABEL_FLIP_FRACTION) == on_right
        ]
        group.sort(key=lambda index: points[index][1])

        # Up from the bottom, then back down from the top: the second pass is
        # what stops the first from pushing the whole stack off the top edge.
        placed: list[float] = []
        for index in group:
            fraction = (points[index][1] - y0) / height
            if placed:
                fraction = max(fraction, placed[-1] + LABEL_MIN_GAP)
            placed.append(fraction)
        overflow = placed[-1] - (1.0 - LABEL_MIN_GAP / 2) if placed else 0.0
        if overflow > 0:
            for position in range(len(placed) - 1, -1, -1):
                placed[position] = min(
                    placed[position],
                    (placed[position + 1] - LABEL_MIN_GAP)
                    if position + 1 < len(placed)
                    else (1.0 - LABEL_MIN_GAP / 2),
                )

        for index, fraction in zip(group, placed):
            x = points[index][0]
            anchors[index] = (
                x - offset if on_right else x + offset,
                y0 + fraction * height,
                "right" if on_right else "left",
            )
    return anchors


def plot_return_vol_plane(
    summary: pd.DataFrame,
    path: str | Path | None = None,
    title: str | None = None,
    counterfactual: Collection[str] = (),
    match_vol: float | None = None,
    lever_path: tuple[str, str] | None = None,
) -> Path:
    """Every rule as a point on the volatility/return plane, with iso-ratio rays.

    The chart that makes the comparison read as a *position* rather than a
    number. Two books at different volatilities are not comparable on return,
    which is the entire reason the matched column exists -- and on this plane
    that stops being an argument and becomes a picture. The rays are constant
    return-to-volatility, so a point's ratio is which ray it sits on, and
    levering a book would move it *along* its own ray if leverage were free.

    It is not, and the chart is drawn to show it. ``lever_path`` names the
    unlevered row and its matched twin, and the arrow between them is where the
    book actually went once financing was charged -- it lands below the ray it
    started on, by the cost of the leverage. ``match_vol`` draws the vertical
    the matched column was aiming at, so the distance from the vertical to where
    the point landed is the overlay's overshoot, also shown rather than stated.

    ``counterfactual`` rows are drawn hollow. They are not strategies anyone can
    run, and a filled marker among filled markers is exactly how a reader ends
    up quoting one.

    ``summary`` is a :func:`src.riskmetrics.realized_risk_table` frame, or
    anything with ``annual_volatility`` and ``sharpe_ratio`` columns indexed by
    rule.

    **The y-axis is the arithmetic annualized return, not CAGR, and that is
    load-bearing.** This project's ratio is ``mean(r) * periods / (sd *
    sqrt(periods))`` -- the return-to-volatility ratio every table here prints --
    and ``CAGR / volatility`` is a different number, smaller by the compounding
    drag and by a different amount for every row. Plotting CAGR against
    volatility would draw points whose position implied one ratio while their
    label quoted another, which is precisely the kind of two-numbers-for-one-
    thing this repo refuses elsewhere. With the arithmetic mean on the axis,
    a point's ratio IS the ray it sits on, exactly.
    """
    required = {"annual_volatility", "sharpe_ratio"}
    missing = required - set(summary.columns)
    if missing or summary.empty:
        raise ValueError(
            f"need a non-empty summary with {sorted(required)}; missing {sorted(missing)}"
        )

    counterfactual = set(counterfactual)
    vols = summary["annual_volatility"].astype("float64")
    # ratio * volatility == mean(excess) * periods_per_year, i.e. the annualized
    # arithmetic return, recovered rather than recomputed so the two cannot drift.
    ratios = summary["sharpe_ratio"].astype("float64")
    cagrs = ratios * vols

    figure, ax = _new_figure(figsize=(9.8, 6.6))

    x_max = float(max(vols.max(), match_vol or 0.0)) * 1.24
    y_min = min(0.0, float(cagrs.min()) * 1.25)
    y_max = float(cagrs.max()) * 1.25

    # Iso-ratio rays through the origin, at round ratios bracketing the data,
    # each labelled where it leaves the axes rather than in a legend -- a legend
    # of parallel grey lines tells a reader nothing about which is which. Every
    # ray gets a label: the shallow ones exit through the right-hand edge, the
    # steep ones through the top, and an unlabelled ray is just clutter.
    finite = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        step = ISO_RATIO_STEP
        low = np.floor(float(finite.min()) / step) * step
        high = np.ceil(float(finite.max()) / step) * step
        for level in np.arange(max(low, step), high + step / 2, step):
            ax.plot([0.0, x_max], [0.0, level * x_max], color=GRID, linewidth=0.8,
                    zorder=0, linestyle=(0, (6, 4)))
            if level * x_max <= y_max:
                anchor, align = (x_max, level * x_max), {"ha": "right", "va": "bottom"}
            else:
                anchor, align = (y_max / level, y_max), {"ha": "left", "va": "top"}
            ax.annotate(f"{level:.2f}", xy=anchor, xycoords="data",
                        color=MUTED, fontsize=8, zorder=0, **align)

    if match_vol is not None:
        ax.axvline(match_vol, color=INK, linewidth=0.9, alpha=0.4, linestyle=(0, (3, 3)))
        # Anchored in axes fraction rather than at y_max, so it clears the ray
        # label that also lives on the top edge.
        ax.annotate(
            f"match point {match_vol:.2%}", xy=(match_vol, 0.92),
            xycoords=("data", "axes fraction"), color=MUTED, fontsize=8.5,
            ha="right", va="top", rotation=90,
        )

    # The lever path, drawn before the markers so the arrow sits under them.
    if lever_path is not None:
        start, end = lever_path
        if start in summary.index and end in summary.index:
            ax.annotate(
                "",
                xy=(vols[end], cagrs[end]), xytext=(vols[start], cagrs[start]),
                arrowprops={"arrowstyle": "-|>", "color": PALETTE[1],
                            "linewidth": 1.3, "shrinkA": 7, "shrinkB": 7,
                            "linestyle": "--"},
            )

    for name in summary.index:
        hollow = name in counterfactual
        ax.scatter(
            vols[name], cagrs[name], s=74, zorder=3,
            facecolor="white" if hollow else PALETTE[0],
            edgecolor=PALETTE[1] if hollow else PALETTE[0],
            linewidth=1.5 if hollow else 0.8,
        )

    _style_axes(ax, ylabel="Annualized return", xlabel="Annualized volatility")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)

    # Labels last, and spread apart. Several rules land within a tenth of a
    # percent of each other on both axes -- that clustering is the finding, so
    # the labels cannot simply be nailed to their markers or the interesting
    # part of the chart is the part that is unreadable. Each label is pushed to
    # a free vertical slot and joined to its marker by a hairline, which keeps
    # the marker where the data puts it and moves only the text.
    placements = _spread_labels(
        ax, [(float(vols[n]), float(cagrs[n])) for n in summary.index],
        (0.0, x_max), (y_min, y_max),
    )
    for name, (label_x, label_y, align) in zip(summary.index, placements):
        ax.annotate(
            f"{name}  {ratios[name]:.3f}",
            xy=(vols[name], cagrs[name]), xytext=(label_x, label_y), xycoords="data",
            textcoords="data", color=INK, fontsize=8.5, va="center", ha=align,
            zorder=4,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6,
                        "shrinkA": 1, "shrinkB": 5},
        )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _title(
        ax,
        title or "Return against volatility - every construction rule",
        "Dashed rays are constant return-to-volatility; a point sits on its own. "
        "Hollow markers are not runnable.",
    )
    return _save(figure, path or REPORTS_DIR / "return_vol_plane.png")


def render_comparison(
    comparison,
    reports_dir: str | Path | None = None,
    panel_rules: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Render both comparison charts from one ``src.comparison.RuleComparison``.

    The adapter, and the only thing here that knows what a ``RuleComparison``
    looks like -- duck-typed rather than imported, so ``src.viz`` keeps sitting
    strictly downstream and imports nothing that imports it back. This is what
    ``compare_rules.py`` calls.

    The risk-share panels are limited to the two protagonists. The reference
    rules have blocks in the printed report, but a five-panel figure is a
    different chart; this one exists to put equal risk beside equal capital.
    """
    directory = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    panel_rules = comparison_panel_rules() if panel_rules is None else panel_rules
    results = comparison.rule_results
    panels = {
        name: results[name].risk_contributions for name in panel_rules if name in results
    }
    if not panels:
        raise ValueError(
            f"none of {list(panel_rules)} is in the comparison's rule_results "
            f"({sorted(results)}); there is nothing to put side by side"
        )
    return {
        "rule_risk_shares": plot_rule_risk_shares(
            panels, directory / "rule_risk_shares.png"
        ),
        "return_vol_plane": plot_return_vol_plane(
            comparison.summary,
            directory / "return_vol_plane.png",
            counterfactual=comparison_counterfactual(),
            match_vol=comparison.matched_target_vol,
            lever_path=comparison_lever_path(),
        ),
    }


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

    What ``run_backtest.py`` calls. ``returns`` is the daily asset
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
    from tests.fixtures import make_fake_result, make_fake_returns, make_fake_risk_free

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # src.covariance logs one INFO line per PSD repair, and the diversification
    # ratio estimates ~1000 covariances to draw one panel. Raised to WARNING for
    # the self-check only, so the four "Wrote ..." lines are visible; anything
    # that actually went wrong still prints.
    logging.getLogger("src.covariance").setLevel(logging.WARNING)

    fake_returns = make_fake_returns()
    fake_result = make_fake_result(strategy="Risk parity (fixture)")

    # The two comparison charts need a whole comparison behind them, not one
    # BacktestResult -- two engine runs and four benchmarks, which is a few
    # seconds. Built here rather than faked because the point of this check is
    # that render_comparison can read a real RuleComparison.
    from .comparison import compare_rules  # local: only this self-check needs it

    fake_rf = make_fake_risk_free().reindex(fake_returns.index).ffill().bfill()
    fake_comparison = compare_rules(fake_returns, fake_rf)

    # Rendered into a throwaway directory, NOT reports/. The charts and
    # crisis_drawdowns.csv tracked in reports/ are the real results the README
    # cites, and both render functions default to exactly those paths: a
    # self-check that wrote there overwrote them with fixture charts, which look
    # identical to real ones. tests/test_stress.py already does the same.
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        written = {
            **render_all(fake_result, fake_returns, reports_dir=scratch),
            **render_comparison(fake_comparison, reports_dir=scratch),
        }
        for name, destination in written.items():
            size_kb = destination.stat().st_size / 1024
            print(f"  {name:28s} -> {destination.name}  ({size_kb:.0f} KB)")
        missing = [name for name, d in written.items() if not d.exists()]
        assert not missing, f"render returned paths that do not exist: {missing}"

    print(f"\n{len(written)} charts rendered from fixture data into a temporary directory "
          f"and discarded. Shape, not magnitude -- the real ones come from "
          f"run_backtest.py and compare_rules.py and live in reports/.")
