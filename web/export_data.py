"""Export real engine outputs atomically; run from the repository virtualenv."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from web.serialize import clean
from run_backtest import load_data, build_benchmarks, ENGINE_LABEL
from src import config
from src.backtest import run_backtest, format_run_summary
from src.metrics import summarize, _drawdown_series as drawdown_series
from src.overlay import unlevered_returns, financed_returns
from src.stress import crisis_decomposition, rolling_correlation, rolling_diversification_ratio

REPO = "abhyudayb-1/equal-risk-vs-equal-capital"


# encoding="utf-8" is not optional on Windows. Without it `text=True` decodes
# with the locale codec (cp1252 here), and both of these read text this project
# actually produces: commit subjects and, via gh, pull-request bodies full of
# em-dashes, arrows and Greek. Measured: the gh call died on byte 0x81 at offset
# 52863 of a real `gh api pulls` response and took the whole export down with it.
def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def github(endpoint):
    return json.loads(subprocess.check_output(
        ["gh", "api", f"repos/{REPO}/{endpoint}", "--paginate", "--slurp"],
        cwd=ROOT, text=True, encoding="utf-8"))


def history():
    commits = []
    for row in git("log", "--format=%H%x1f%aI%x1f%an%x1f%s").splitlines():
        sha, date, author, title = row.split("\x1f", 3)
        commits.append(dict(sha=sha, date=date, author=author, title=title,
                            url=f"https://github.com/{REPO}/commit/{sha}"))
    phases, current = [], None
    # encoding="utf-8" for the same reason git() needs it. Without it TASKS.md is
    # read as cp1252 on Windows and every em-dash becomes "a-tilde-euro-dash" in
    # the export. That also broke the phase titles: the title regex skips leading
    # non-word characters, and the first byte of the mojibake IS a word character,
    # so the garbage leaked into the title instead of being skipped like the dash.
    for line in (ROOT / "TASKS.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"## Phase (\d+) [^\w]*(.*)", line)
        if match:
            current = dict(number=int(match[1]), title=match[2].split("(")[0].strip(), steps=[])
            phases.append(current)
        cells = [c.strip() for c in line.strip("|").split("|")]
        if current and len(cells) == 5 and cells[0].isdigit():
            n = int(cells[0])
            matches = []
            for c in reversed(commits):
                m = re.search(r"[Ss]teps? (\d+)(?:[-\u2013](\d+))?", c["title"])
                if m and int(m[1]) <= n <= int(m[2] or m[1]):
                    matches.append(c)
            # Early analysis steps used descriptive rather than numbered subjects.
            evidence = {11:"43cef53", 12:"1fea96f", 13:"4f6b332",
                        18:"6c26f98", 19:"6c26f98", 20:"6c26f98", 32:"61d7c1f"}
            commit = matches[0] if matches else next(
                (c for c in commits if c["sha"].startswith(evidence.get(n, "NO MATCH"))), None)
            current["steps"].append(dict(number=n, owner=cells[1], branch=cells[2].strip("`"),
                description=cells[3], done=cells[4], commit=commit,
                evidence="Live-run evidence (shared commit)" if n == 32 else "Implementation commit"))
    # Steps 43 and 44 used to be appended here by hand, because neither was ever
    # written into TASKS.md. Both products were removed when the project was
    # narrowed to equal risk vs equal capital, so the injections are gone too --
    # the phases below are exactly what TASKS.md carries, with no hand-patching
    # and therefore no step numbering that cannot be traced back to that file.
    tags = [dict(name=t, sha=git("rev-list", "-n", "1", t),
                 url=f"https://github.com/{REPO}/tree/{t}") for t in git("tag", "--sort=version:refname").splitlines()]
    activity = {"commits": commits[:20], "pulls": [], "issues": [], "contributors": [], "tags": tags}
    try:
        fields = ("id", "number", "title", "state", "html_url", "merged_at", "updated_at")
        activity["pulls"] = [{k:p.get(k) for k in fields} for p in sum(github("pulls?state=all&per_page=100"), [])]
        activity["issues"] = [{k:i.get(k) for k in fields} for i in sum(github("issues?state=all&per_page=100"), []) if "pull_request" not in i]
        activity["contributors"] = [{k:c[k] for k in ("login", "html_url", "avatar_url", "contributions")} for c in sum(github("contributors?per_page=100"), [])]
        activity["status"] = "snapshot"
    # Widened from (CalledProcessError, OSError): the block already intends a
    # missing GitHub connection to degrade to "unavailable" rather than fail the
    # export, and a decode or JSON error is the same kind of event. Before this,
    # either one escaped and killed a run whose engine numbers were already good.
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError,
            json.JSONDecodeError) as exc:
        activity["status"] = "unavailable"
        activity["error"] = f"{type(exc).__name__}: {exc}"
    return dict(phases=phases, tags=tags, commits=commits, activity=activity)


def export(refresh=False, output=None):
    out = Path(output) if output else ROOT / "web/data"
    out.mkdir(parents=True, exist_ok=True)
    print("Loading engine data...", flush=True)
    returns, rate = load_data(config.DATA_START_DATE, refresh)
    engine = run_backtest(returns, risk_free=rate)
    results = {ENGINE_LABEL: engine, **build_benchmarks(returns)}
    common = engine.returns.index
    for result in results.values():
        result.validate()
        if not result.returns.index.equals(common):
            raise ValueError("Benchmark calendars do not match; cannot export a fair comparison")
    print(format_run_summary(engine), flush=True)
    metrics = []
    for name, result in results.items():
        metrics.append({**summarize(result.returns, periods_per_year=config.TRADING_DAYS_PER_YEAR, name=name),
                        "mean_leverage": float(result.leverage.mean())})
    unlevered = unlevered_returns(returns.loc[common], engine.weights)
    free = financed_returns(unlevered, engine.leverage, pd.Series(0., index=common))
    financing = {name: summarize(r, periods_per_year=config.TRADING_DAYS_PER_YEAR, name=name)
                 for name, r in [("Unlevered ERC", unlevered), ("Financing charged", engine.returns), ("Financing omitted", free)]}
    equity = pd.DataFrame({name: (1 + r.returns).cumprod() for name, r in results.items()})
    correlation = rolling_correlation(returns)
    ratio = rolling_diversification_ratio(returns, weights=engine.weights, step_days=5)
    crisis = crisis_decomposition(results)
    mechanisms = []
    for name, (start, end) in config.CRISIS_WINDOWS.items():
        sliced = returns.loc[(returns.index > start) & (returns.index <= end)]
        corr = correlation.loc[start:end].dropna()
        before = engine.leverage.loc[engine.leverage.index < pd.Timestamp(start)]
        mechanisms.append(dict(name=name, start=start, end=end,
            sleeve_returns=((1+sliced).prod()-1).to_dict(), correlation_mean=corr.mean(),
            correlation_positive_fraction=(corr > 0).mean(), entry_leverage=float(before.iloc[-1])))
    payload = dict(schema_version=1, repository=REPO, source_commit=git("rev-parse", "HEAD"),
        generated_at=datetime.now(timezone.utc).isoformat(), market_data_through=str(common[-1].date()),
        refresh_requested=refresh, start=str(common[0].date()), metrics=metrics,
        config={k: getattr(config,k) for k in ["UNIVERSE","SLEEVE_LABELS","TARGET_VOL","LEVERAGE_CAP",
            "DATA_START_DATE","COV_EWMA_HALFLIFE_DAYS","VOL_EWMA_HALFLIFE_DAYS","COV_SHRINKAGE",
            "CORRELATION_WINDOW_DAYS","VAR_CONFIDENCE"]},
        weights=engine.weights, risk_contributions=engine.risk_contributions,
        leverage=engine.leverage.rename("leverage").to_frame(), equity=equity,
        drawdowns=pd.DataFrame({name:drawdown_series(r.returns) for name,r in results.items()}),
        correlation=correlation.rename("correlation").to_frame(),
        diversification=ratio.rename("ratio").to_frame(), crisis=crisis, mechanisms=mechanisms,
        diagnostics=engine.meta, benchmark_diagnostics={name:r.meta for name,r in results.items()},
        run_summary=format_run_summary(engine), financing=financing)
    files = {"results.json": payload, "history.json": history()}
    encoded = {name:json.dumps(clean(data), allow_nan=False, separators=(",", ":")) for name,data in files.items()}
    version = hashlib.sha256("".join(encoded.values()).encode()).hexdigest()[:16]
    for name, data in encoded.items():
        temp = out / (name + ".tmp")
        temp.write_text(data + "\n")
        temp.replace(out / name)
    manifest = dict(version=version, generated_at=payload["generated_at"], source_commit=payload["source_commit"],
                    market_data_through=payload["market_data_through"], files=list(files))
    temp = out / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2)+"\n")
    temp.replace(out / "manifest.json")
    print(f"Published {version} to {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    export(args.refresh, args.output)
