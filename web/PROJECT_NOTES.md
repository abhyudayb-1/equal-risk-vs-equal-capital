# Project research

> **Predates the cleanup.** These notes describe the project as it stood at `v1.4`,
> including Phase 6's candidate analysis and the on-demand `api/analyze.py`. Both are
> removed; see the repository's `AGENTS.md`/`CLAUDE.md` ledger for what went and which
> tags still hold it. Kept as a record rather than rewritten.

Research baseline: `b16a9c5` on `origin/main`, retrieved 2026-09-21. Repository:
https://github.com/shivakasula/portfolio-allocation-capstone-project (private).
GitHub confirmed the transfer from `abhyudayb-1` when the web branch was pushed.
The local main checkout was behind at `7e9f134`; `web-ui` starts from current origin/main.

## Sources and findings

Read README, CLAUDE, local AGENTS, config, BacktestResult, the CLI, visualization
functions, tracked reports, and test definitions; inspected TASKS, incremental and
comparison APIs, git log/stat, all tags, all 61 pull requests and issues (none at
research time). The current repository is beyond the prompt's steps 1-42.

The engine allocates equal volatility contributions across SPY, TLT, DBC and TIP,
using EWMA covariance, diagonal shrinkage, correlation-space PSD repair and a
restarting SLSQP solver. Monthly decisions earn returns only after their close.
The faster trailing-volatility overlay finances leverage at FRED DTB3.

- Phase 0, steps 1-7: scaffold, configuration, frozen contract, cache and metrics
  ports, synthetic fixtures, and contract gate let two owners work independently.
- Phase 1, steps 8-13: adjusted price data, financing data and calendar alongside
  unlevered benchmarks and crisis-window metrics.
- Phase 2, steps 14-20: covariance and ERC solver alongside an independent convex
  oracle and degenerate-input tests. Optimizer success is not ERC convergence.
- Phase 3, steps 21-27: overlay, financing and backtest alongside crisis diagnostics
  and four charts. The estimator de-levers after damage; that is a finding.
- Phase 4, steps 28-31: look-ahead mutation tests, run-health diagnostics, CLI and docs.
- Phase 5, steps 32-35: live integration and honest results, tagged v1.0.
  Some steps were committed together, and step 32 has no independently titled
  commit; its evidence is the live-results commit, explicitly labeled as such.
- Phase 6, steps 36-42: candidate configuration, proportional perturbations,
  five-asset ERC, marginal compensation test, tests, CLI, write-up, final audit.
  Tagged v1.1; follow-up audit fixes ship in v1.2.
- Step 43, v1.3: realized before/after risk on an in-sample replay of today's weights.
  v1.3.1 prevents self-checks overwriting tracked report charts.
- Step 44, v1.4: compare the historical strategy with ticker buy-and-hold over their
  overlapping dates, including relative and tail metrics and calendar warnings.

Published v1.0 figures (historical claims, not a fresh run): CAGR 7.12%, volatility
11.26%, drawdown -29.10%; 60/40 CAGR 8.72%. The README's 1.06% annual financing
drag is not a percentage-point reduction in Sharpe. Its CAGR rows differ by
about 1.14 percentage points. The UI computes and labels the CAGR difference
directly from the new export instead of repeating that ambiguity.

COVID is the deepest strategy drawdown in the published run. The 2022 explanation
is lost diversification and a greater TLT loss than SPY, not a persistent positive
stock/bond correlation. Crisis sleeve returns must exclude the opening peak's
return. Published README metrics use zero-risk-free return/vol; analyzer Sharpe
uses its own configured convention. Never silently equate them.

## Architecture and ownership

Vite + React, Tailwind, accessible Radix/shadcn-style controls, Recharts, Framer
Motion and GSAP ScrollTrigger. Recharts covers all plots without an additional
D3 or Plotly bundle. Inter and a monospace font are self-hosted font packages.
Canva is unavailable and unnecessary; real exported financial plots provide the
primary visual assets. Browser QA uses Playwright if agent-browser is unavailable.

Precompute AAPL, MSFT, NVDA and GLD snapshots for instant load; any other ticker is
computed on demand by `api/analyze.py`. A ticker that cannot be analysed gets an
explicit unavailable state, never a substituted result. Python export imports the existing
APIs, including incremental and comparison functions; no finance math is ported
to JavaScript. Full daily observations are exported, preserving crisis extrema.
Candidate failures are explicit, timestamped records. Core engine failure aborts
publication so a failed run cannot replace a successful export.

A minimal Vercel serverless API is justified by this private repository: the
browser cannot safely carry a GitHub token. The server polls allowlisted GitHub
endpoints with ETags, caches for 25 seconds, and reports rate limits/staleness.
Browsers poll every 25 seconds, pause while hidden, and retain the last good data
on failure. A manifest gives the immutable commit for a coherent data version.
The server-side fine-grained token needs Contents, Pull requests and Issues read.
There is no webhook and consequently no webhook secret to manage.

On pushes, GitHub Actions runs the exporter, commits web/data only with the
workflow token, and explicitly deploys via Vercel CLI (bot pushes do not trigger
another workflow). Branch concurrency and a source-HEAD check prevent an old run
publishing over a newer push. Deployment requires configured Vercel secrets.
Keep Vercel deployment protection enabled: source and exported activity are private.

Person A (abhyudayb-1) owns engine modules. Person B (shivakasula) owns analysis,
tests, CLI and README; Phase 5 onward is joint. This task only adds web/ and a
workflow. AGENTS.md and Untitled remain untracked and excluded from commits.
Neither src/ nor the tracked reports are modified by this work.

## Verification record

Fresh-export findings, browser results and deployment limitations are recorded
in web/README.md and web/QA.md as implementation and verification complete.
