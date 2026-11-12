"""Is a gap between two return-to-volatility ratios distinguishable from noise?

Every table in this project ranks rules by a ratio -- the annualised arithmetic
mean return over the annualised volatility -- and the headline is a *difference*
of two of them: 0.723 for ERC against 0.683 for 1/N. A ratio estimated from
twenty years of daily returns is a noisy number, and a difference of two noisy
numbers is the thing most easily mistaken for a finding. This module puts an
interval on that difference, so the point estimate is never printed alone.

Consumed by :mod:`src.comparison`, which measures every rule against 1/N with
:func:`ratio_difference_table` and prints :func:`format_ratio_difference_table`
under the performance table. Changing a signature here changes that report.

**Two derivations, on purpose.** The interval that is reported comes from a
bootstrap; a closed-form test is computed beside it as a cross-check, for the
same reason ``tests/test_riskparity.py`` keeps an independent oracle for the
solver. They rest on different assumptions and should be read together:

* :func:`ratio_difference_test` -- a **paired stationary block bootstrap**
  (Politis & Romano, 1994). *Paired*: each resample draws the same dates for
  both series, so their correlation is preserved. That is not a refinement.
  ERC and 1/N hold the same four sleeves and are about 0.9 correlated, so most
  of the noise in one ratio is also in the other and cancels in the difference;
  resampling them independently would throw that away and report an interval
  far too wide. *Block*: dates are drawn in runs of random (geometric) length
  rather than one at a time, so whatever serial dependence the returns carry --
  chiefly volatility clustering, which matters to a statistic with a standard
  deviation in its denominator -- survives into each resample. *Stationary*:
  random block lengths and wrap-around keep the resampled series stationary,
  which fixed-length blocks do not.
* :func:`memmel_test` -- the Jobson-Korkie (1981) z-test for the difference of
  two Sharpe ratios with Memmel's (2003) correction. Closed-form and free of
  simulation, but it assumes returns are **independent and normal**, which
  daily returns are not. It is the standard test; it is reported as the
  cross-check rather than the headline because its assumptions are the ones
  the block bootstrap exists to drop.

**What the interval is and is not.** It is a percentile interval: the central
``confidence`` share of the bootstrap distribution of the difference. The
p-value is that same distribution inverted -- twice the share of resamples on
the far side of zero -- so the two cannot disagree about whether zero is
inside. Both describe sampling noise *within one history*. Neither says
anything about whether the next twenty years will resemble these, and neither
corrects for the number of comparisons a reader makes across the table.

Every function is a pure function of return series and a seed, so this module
runs identically on fixtures and on live data and needs no network. Like
``riskmetrics.py`` and ``headtohead.py`` it carries no ``__main__`` self-check;
its coverage is ``tests/test_significance.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    ANNUAL_RISK_FREE_RATE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_MEAN_BLOCK_DAYS,
    BOOTSTRAP_SEED,
    RATIO_TEST_CONFIDENCE,
    TRADING_DAYS_PER_YEAR,
)
from .metrics import _ZERO_VARIANCE_EPS, sharpe_ratio

# Resamples are generated this many at a time. Purely a memory bound: each one
# is a row of n indices, and 10,000 x 5,000 of them at once is ~400MB. It does
# not affect the result -- the draws come from one generator in one order
# whatever the chunking -- which is why it is not a config.py constant.
_CHUNK_DRAWS: int = 250


class SignificanceError(ValueError):
    """The test cannot be run on the inputs given.

    Raised rather than returning NaN bounds: an interval of ``[nan, nan]`` in a
    table reads as "not computed", and a reader skims past it to the point
    estimate -- which is exactly the number this module exists to qualify.
    """


# ---------------------------------------------------------------------------
# The resampling scheme
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(
    n_observations: int,
    n_draws: int,
    mean_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row indices for ``n_draws`` stationary-bootstrap resamples, shape ``(n_draws, n)``.

    Each row is built from blocks: start at a uniformly random position, then on
    every step either continue to the next observation (probability
    ``1 - 1/mean_block``) or jump to a fresh random position (probability
    ``1/mean_block``). Block lengths are therefore geometric with mean
    ``mean_block``. Running off the end wraps to the start, so every observation
    is equally likely to appear at every position.

    ``mean_block = 1`` restarts on every step, which is the ordinary i.i.d.
    bootstrap -- the limit the tests pin the scheme against.
    """
    if n_observations < 1 or n_draws < 1:
        raise SignificanceError(
            f"need at least one observation and one draw, got {n_observations} and {n_draws}"
        )
    if not np.isfinite(mean_block) or mean_block < 1.0:
        raise SignificanceError(
            f"mean_block must be at least 1 observation, got {mean_block}"
        )

    restart = rng.random((n_draws, n_observations)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n_observations, size=(n_draws, n_observations))

    position = np.arange(n_observations)
    # For each step, the position of the most recent restart at or before it;
    # the index is that block's random start plus the steps taken since.
    last_restart = np.maximum.accumulate(np.where(restart, position, 0), axis=1)
    block_start = np.take_along_axis(starts, last_restart, axis=1)
    return (block_start + position - last_restart) % n_observations


def _bootstrap_ratios(
    excess: np.ndarray,
    periods_per_year: int,
    n_draws: int,
    mean_block: float,
    seed: int,
) -> np.ndarray:
    """The ratio of every column on every resample, shape ``(n_draws, n_columns)``.

    One set of resampled dates is applied to **all** columns, which is what
    makes the comparison paired. A resample is summarised by how many times it
    drew each date, so its mean and variance are two matrix products against
    the counts rather than a gather of the returns themselves.
    """
    n, _ = excess.shape
    squares = excess**2
    rng = np.random.default_rng(seed)

    blocks = []
    remaining = n_draws
    while remaining > 0:
        m = min(_CHUNK_DRAWS, remaining)
        remaining -= m
        indices = stationary_bootstrap_indices(n, m, mean_block, rng)
        offsets = np.arange(m)[:, None] * n
        counts = np.bincount((indices + offsets).ravel(), minlength=m * n)
        counts = counts.reshape(m, n).astype("float64")

        mean = counts @ excess / n
        variance = (counts @ squares - n * mean**2) / (n - 1)
        if not np.all(variance > 0.0):
            raise SignificanceError(
                "a resample has zero variance in some series; the series is too "
                "short or too nearly constant to bootstrap a ratio from"
            )
        blocks.append(mean / np.sqrt(variance) * np.sqrt(periods_per_year))
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# Input handling shared by both tests
# ---------------------------------------------------------------------------
def _period_risk_free(periods_per_year: int, annual_risk_free_rate: float) -> float:
    """De-compounded per-period rate -- the same conversion ``sharpe_ratio`` makes."""
    return (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0


def _excess_matrix(
    frame: pd.DataFrame, periods_per_year: int, annual_risk_free_rate: float
) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] < 2:
        raise SignificanceError("need a DataFrame with at least two return columns")
    values = frame.to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        raise SignificanceError(
            "returns contain NaN or inf; a paired test needs every series on every date"
        )
    if len(values) < 3:
        raise SignificanceError(
            f"need at least 3 observations to test a ratio difference, got {len(values)}"
        )
    spread = values.std(axis=0, ddof=1)
    if np.any(spread < _ZERO_VARIANCE_EPS):
        flat = [str(c) for c, s in zip(frame.columns, spread) if s < _ZERO_VARIANCE_EPS]
        raise SignificanceError(
            f"{flat} has no variance, so its ratio is undefined and cannot be compared"
        )
    return values - _period_risk_free(periods_per_year, annual_risk_free_rate)


def _check_settings(confidence: float, n_draws: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise SignificanceError(
            f"confidence must be strictly between 0 and 1, got {confidence}"
        )
    if int(n_draws) != n_draws or n_draws < 100:
        raise SignificanceError(
            f"n_draws must be a whole number of at least 100, got {n_draws!r}; fewer "
            "cannot resolve the tails of an interval"
        )


def _pair(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    if not isinstance(strategy, pd.Series) or not isinstance(benchmark, pd.Series):
        raise SignificanceError("strategy and benchmark must both be pandas Series")
    if not strategy.index.equals(benchmark.index):
        raise SignificanceError(
            "the two series must share one index; a paired test resamples the same "
            "dates for both"
        )
    return pd.DataFrame({"strategy": strategy, "benchmark": benchmark})


# ---------------------------------------------------------------------------
# The closed-form cross-check
# ---------------------------------------------------------------------------
def memmel_test(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    confidence: float = RATIO_TEST_CONFIDENCE,
) -> dict[str, float]:
    """Jobson-Korkie test for a difference of two Sharpe ratios, Memmel-corrected.

    With per-period ratios ``s1`` and ``s2``, correlation ``rho`` and ``T``
    observations, the asymptotic variance of ``s1 - s2`` under i.i.d. normal
    returns is::

        V = (1/T) * [ 2 - 2*rho + 0.5 * (s1^2 + s2^2 - 2*s1*s2*rho^2) ]

    The first two terms carry almost all of it at daily frequency, where the
    per-period ratios are ~0.05: the variance is essentially ``2(1 - rho)/T``,
    which is why the correlation between the two books decides how wide the
    interval is. Everything returned is annualised to match the ratios the
    project prints (``sqrt(periods_per_year)`` on the difference and its error).

    This assumes independence and normality. See the module docstring for why it
    is the cross-check and the bootstrap is the headline.
    """
    if not 0.0 < confidence < 1.0:
        raise SignificanceError(
            f"confidence must be strictly between 0 and 1, got {confidence}"
        )
    excess = _excess_matrix(_pair(strategy, benchmark), periods_per_year, annual_risk_free_rate)
    n = len(excess)
    s1, s2 = excess.mean(axis=0) / excess.std(axis=0, ddof=1)
    rho = float(np.corrcoef(excess[:, 0], excess[:, 1])[0, 1])

    variance = (2.0 - 2.0 * rho + 0.5 * (s1**2 + s2**2 - 2.0 * s1 * s2 * rho**2)) / n
    annualise = float(np.sqrt(periods_per_year))
    difference = float(s1 - s2) * annualise
    if variance <= 0.0:
        # Two perfectly correlated series with the same ratio: the difference is
        # exactly zero and has no sampling error to test it against.
        return {
            "difference": difference,
            "std_error": 0.0,
            "z": float("nan"),
            "p_value": float("nan"),
            "ci_low": difference,
            "ci_high": difference,
        }

    std_error = float(np.sqrt(variance)) * annualise
    z = difference / std_error
    half_width = float(norm.ppf(0.5 + confidence / 2.0)) * std_error
    return {
        "difference": difference,
        "std_error": std_error,
        "z": float(z),
        "p_value": float(2.0 * norm.sf(abs(z))),
        "ci_low": difference - half_width,
        "ci_high": difference + half_width,
    }


# ---------------------------------------------------------------------------
# The bootstrap, for one pair and for a whole table
# ---------------------------------------------------------------------------
def _summarise_draws(draws: np.ndarray, confidence: float) -> dict[str, float]:
    """Percentile interval, standard error and two-sided p-value of one difference."""
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    n = len(draws)
    # Twice the smaller tail beyond zero, with the +1 that keeps a p-value from
    # being reported as exactly 0 off a finite number of resamples.
    below = (int((draws <= 0.0).sum()) + 1) / (n + 1)
    above = (int((draws >= 0.0).sum()) + 1) / (n + 1)
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "std_error": float(draws.std(ddof=1)),
        "p_value": float(min(1.0, 2.0 * min(below, above))),
    }


def ratio_difference_table(
    returns: pd.DataFrame,
    benchmark: str,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    confidence: float = RATIO_TEST_CONFIDENCE,
    n_draws: int = BOOTSTRAP_DRAWS,
    mean_block_days: float = BOOTSTRAP_MEAN_BLOCK_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Every column's ratio against ``benchmark``'s, with an interval on the gap.

    One row per column other than ``benchmark``. All rows share one set of
    resampled dates, so the table is internally consistent -- two rows' intervals
    come from the same simulated histories -- and a row here is identical to
    :func:`ratio_difference_test` on that pair with the same seed.

    Columns: ``ratio`` and ``benchmark_ratio`` (the point estimates, exactly
    ``metrics.sharpe_ratio``'s), ``difference``, the bootstrap ``ci_low`` /
    ``ci_high`` / ``std_error`` / ``p_value``, the closed-form
    ``memmel_std_error`` / ``memmel_p_value`` beside them, and the settings the
    row was computed at, so a table read on its own says how it was made.
    """
    _check_settings(confidence, n_draws)
    if not isinstance(returns, pd.DataFrame) or benchmark not in returns.columns:
        raise SignificanceError(
            f"{benchmark!r} is not a column of the returns frame, so there is nothing "
            "to measure the others against"
        )
    excess = _excess_matrix(returns, periods_per_year, annual_risk_free_rate)
    ratios = _bootstrap_ratios(
        excess, periods_per_year, int(n_draws), float(mean_block_days), seed
    )

    position = list(returns.columns).index(benchmark)
    benchmark_ratio = sharpe_ratio(returns[benchmark], periods_per_year, annual_risk_free_rate)
    rows = []
    for k, name in enumerate(returns.columns):
        if k == position:
            continue
        ratio = sharpe_ratio(returns[name], periods_per_year, annual_risk_free_rate)
        closed_form = memmel_test(
            returns[name], returns[benchmark], periods_per_year, annual_risk_free_rate, confidence
        )
        rows.append(
            {
                "name": name,
                "ratio": ratio,
                "benchmark_ratio": benchmark_ratio,
                "difference": ratio - benchmark_ratio,
                **_summarise_draws(ratios[:, k] - ratios[:, position], confidence),
                "memmel_std_error": closed_form["std_error"],
                "memmel_p_value": closed_form["p_value"],
                "confidence": float(confidence),
                "n_observations": int(len(returns)),
                "n_draws": int(n_draws),
                "mean_block_days": float(mean_block_days),
            }
        )
    return pd.DataFrame(rows).set_index("name").rename_axis(None)


def ratio_difference_test(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    annual_risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
    confidence: float = RATIO_TEST_CONFIDENCE,
    n_draws: int = BOOTSTRAP_DRAWS,
    mean_block_days: float = BOOTSTRAP_MEAN_BLOCK_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """``strategy``'s ratio minus ``benchmark``'s, with a block-bootstrap interval.

    The single-pair form of :func:`ratio_difference_table`; see that function for
    the keys and the module docstring for the method.
    """
    table = ratio_difference_table(
        _pair(strategy, benchmark),
        "benchmark",
        periods_per_year=periods_per_year,
        annual_risk_free_rate=annual_risk_free_rate,
        confidence=confidence,
        n_draws=n_draws,
        mean_block_days=mean_block_days,
        seed=seed,
    )
    return table.loc["strategy"].to_dict()


# ---------------------------------------------------------------------------
# The printed form
# ---------------------------------------------------------------------------
def format_ratio_difference_table(table: pd.DataFrame, benchmark_label: str) -> str:
    """:func:`ratio_difference_table`'s frame, the interval beside each point estimate.

    The interval column is marked when it contains zero, because that is the
    reading a table of bare gaps hides: a gap whose interval spans zero is one
    this sample cannot tell from no gap at all.
    """
    cells: dict[str, dict[str, str]] = {}
    for name in table.index:
        row = table.loc[name]
        spans_zero = row["ci_low"] <= 0.0 <= row["ci_high"]
        level = f"{row['confidence']:.0%}"
        cells[name] = {
            "ratio": f"{row['ratio']:.3f}",
            f"gap to {benchmark_label}": f"{row['difference']:+.3f}",
            f"{level} interval": f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f}",
            "spans zero": "yes" if spans_zero else "NO",
            "p (bootstrap)": f"{row['p_value']:.3f}",
            "p (Memmel)": "--" if pd.isna(row["memmel_p_value"]) else f"{row['memmel_p_value']:.3f}",
        }
    return pd.DataFrame(cells).T.to_string()
