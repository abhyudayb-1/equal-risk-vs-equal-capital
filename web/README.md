# Risk Parity Lab

The interactive research notebook for the Python study in this repository: what allocating
equal *risk* buys over allocating equal *capital*, and what it costs in volatility and return.

Every figure comes from `web/export_data.py`, which calls the same `src/` functions
`run_backtest.py` does, so nothing on the site is computed twice. No tracked report under
`reports/` is edited by anything here. (An earlier version of this line said no engine files are
touched either; that stopped being true when the tree moved onto `main` and `src/config.py`
gained the read-only-filesystem hook the deployment needs.) Historical results and repository
activity have separate dates and provenance.

## Local preview

From the repository root:

```bash
.venv/bin/python web/export_data.py
cd web
npm ci
npm run dev
```

The exporter uses the real cached/live price data and the existing Python APIs.
`--refresh` explicitly refreshes prices and financing data. It writes strict JSON
to `web/data/`: results, history and a version manifest. A failed core run exits
before publication rather than publishing a partial export. Do not use fixtures for
a published export. The initial checked-in run uses cached market
data through the date displayed on the site, not today's real-time quotes.

The Vite dev server serves the same Node handler as Vercel at `/api/live`.
`npm run build` embeds the checked-in JSON into `dist/data/`; it does not require
Python. `npm run preview` serves the static production bundle, without the API.

## Live repository updates

Set server-only environment variables in Vercel (Preview and Production), or in
ignored `web/.env.local` for local development:

| Variable | Value |
| --- | --- |
| `GITHUB_READ_TOKEN` | Fine-grained token scoped only to the capstone repository, Contents / Issues / Pull requests read-only |
| `GITHUB_REPOSITORY` | `shivakasula/portfolio-allocation-capstone-project` |
| `GITHUB_DATA_BRANCH` | `web-ui` for this preview, `main` after merge |

Never use a `VITE_` prefix for a secret. Never commit tokens. No webhook is used,
so there is no webhook secret or webhook-signature validation surface.

The browser polls every 25 seconds while visible. A server-side 25-second cache
and GitHub ETags avoid repeating unchanged upstream transfers. GitHub's
[conditional-request guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)
explains when authenticated 304 responses avoid rate-limit cost. The proxy accepts
only named data files and immutable commit hashes, never arbitrary paths or URLs.
The manifest and all files for an update come from the same Git commit. Rate-limit
reset times and Retry-After are respected. Offline or failed requests keep the last
successful state. A fresh page load always has the deployed snapshot available.

The indicator reads LIVE only after a successful API response. It otherwise says
EXPORTED SNAPSHOT or OFFLINE SNAPSHOT. A banner names the missing connection or old
market-data date. The last checked time is distinct from the market-data date.
This is a live repository, not a live intraday market-price service.

This repository is private. Keep
[Vercel deployment protection](https://vercel.com/docs/deployment-protection/methods-to-protect-deployments)
enabled for previews and any production domain that serves this data. The API
relies on that deployment-level access control, and the bundled JSON also contains
private repository activity. Do not publish an unprotected mirror accidentally.

## Export and deploy on push

The workflow runs on `main` and `web-ui` pushes (excluding generated-data and
screenshots-only changes) and manual dispatch. It installs the
Python dependencies, runs the three relevant consumer/guard suites, exports with
`--refresh`, validates the result, builds, commits only `web/data/*.json`, and
explicitly invokes the Vercel CLI. The explicit deployment is necessary because
pushes made with `GITHUB_TOKEN` do not start another workflow run.

Configure repository Actions secrets:

- `VERCEL_TOKEN`: a Vercel automation token for the selected account/team.
- `VERCEL_ORG_ID`: `team_TJIEoAtL2cA1itwERvNjBQ2B`.
- `VERCEL_PROJECT_ID`: `prj_RxtV0PF90MyGjZaWqZgPbd0yoywS`.

The team and project identifiers were configured during implementation. The
automation token still requires an account owner to provide it.

Enable workflow write access to Contents and permit the generated-data commit on
the tracked branch. Branch protection may require a separate data branch or an
approved GitHub App; do not bypass your team's branch protection rules. A newer
push cancels the older run, and a final remote-HEAD check prevents stale publishing.
Failed publication/deployment is visible in Actions; the prior export stays usable.
Missing deployment secrets produce an explicit warning, not a fake deployment URL.
After secrets are configured, run the workflow once to verify the whole push cycle.

## Deployment target

User-selected target: [sahas7 / portfolio-optimization-engine](https://vercel.com/sahas7/portfolio-optimization-engine),
username `skasulas-4436`. Ask which account to use for future deployment work.
The other Vercel account must remain untouched. This session authenticated through
an isolated CLI configuration at `/private/tmp/risk-parity-vercel-auth`; that
temporary directory is not a portable or permanent project credential.

The selected Vercel project's existing Git integration points to a different repo
(`shivakasula/portfolio-optimization-engine`). This task preserves that integration
and deploys the `web/` directory explicitly. Its Python project preset is overridden
by this directory's `vercel.json` for the preview. For automatic capstone deployments,
use the workflow secrets above; changing the project's Git integration is a separate
owner decision. Do not deploy this directory as an unreviewed production replacement.

```bash
cd web
vercel link --project portfolio-optimization-engine --scope sahas7
vercel deploy --scope sahas7 --env GITHUB_DATA_BRANCH=web-ui
```

The initial deployment is a preview only. Do not add `--prod` until the web UI is
reviewed and merged. Vercel's [Vite documentation](https://vercel.com/docs/frameworks/frontend/vite)
describes the `api/` serverless route used here.

## UI and analysis

Six tabs: Overview, Findings, The data, Benchmarks, Hardening and Method. The page
shows the headline comparison, the COVID and 2022 findings, interactive replacements
for the four report figures, the benchmark table, the guardrails and the methodology.

**Four features were removed when the project was narrowed to equal risk vs equal
capital** (see the repository's `AGENTS.md`/`CLAUDE.md`): the position-analysis tab and
its `api/analyze.py` / `api/search.py` functions, the research chatbot, the two-person
timeline workspace, and the Build log tab. The first three had no subject left once
`src/incremental.py` and `src/compare.py` were deleted from `main`; the Build log was
process rather than research. `web/data/history.json` is still exported, but now only
feeds the version badge in the header — trimming it is `TASKS.md` step 45's job.

There are **no Python serverless functions left**; `api/live.js` is the only handler, so
`bundle_engine.py` and `web/requirements.txt` are gone too.

Theme tokens live in `src/theme.css`. Inter and IBM Plex Mono are self-hosted.
Native selects retain built-in keyboard support; the tab bar is a plain
`role="tablist"` (the Radix dependency went with the removed sub-tabs).
Recharts covers the plots without extra D3/Plotly. Framer Motion
handles entry and count-up effects, GSAP handles scroll movement; reduced-motion
users see no animation. Chart summaries supplement tooltips and data downloads.
Canva and agent-browser were unavailable; real-data charts and Playwright/Chrome
were used instead. No generated financial observations or marketing stock assets.

## Verification

```bash
npm test
npm run build
npm run qa
```

`npm test` checks strict export contracts, step-to-commit links, financial
identities, API allowlists, conditional 304 reuse, and rate-limit backoff.
`npm run qa` expects the dev server on 5173 (override with `QA_URL`) and local Chrome.
It produces screenshots and an axe report, exercising themes, crisis selection, tabs
and offline retention. See `QA.md` for measured results and remaining deployment
configuration.

**Nothing in this directory was built or tested in the PR that removed those four
features** — node and npm were not installed on the machine that made the edits. The
imports were checked statically instead (every relative import resolves; no identifier
is left imported and unused), and `web/export_data.py` was run end to end against live
data. Run `npm ci && npm test && npm run build` before trusting the bundle, and note
that `package-lock.json` is stale until `npm install` regenerates it.

`npm run qa:live` uses a controlled, test-only GitHub transport to verify atomic
version refresh, retained ticker/crisis selections and stale-data retention.
It never changes the exported financial observations. The static build is
prerendered from the same JSON, then hydrated; charts load when they enter view.
