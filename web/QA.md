# Verification and handoff

> **Dated evidence, and it predates the cleanup.** Everything below was measured on
> the pre-cleanup site, which had a position-analysis tab, a research chatbot, a
> two-person timeline workspace and a Build log tab. All four were removed when the
> project was narrowed to equal risk vs equal capital, and the chatbot's own
> verification section was deleted from this file with it. The remaining
> measurements are **not** re-run — node and npm were not installed on the machine
> that made the removals, so nothing here was re-measured and nothing here was
> edited to pretend otherwise. Re-run `npm test`, `npm run build` and `npm run qa`
> and replace these figures before treating them as current.

## Verified locally

- Real engine export: 235 monthly decisions, 4,920 daily returns, market data
  through 2026-09-18. Risk parity CAGR 7.10%, 60/40 8.64%, strategy maximum
  drawdown -29.10%. Source commit and export time are in the manifest.
- AAPL, MSFT, NVDA and GLD snapshots completed through the existing incremental
  and comparison APIs. No synthetic observations are published.
- `npm test`: 7 passing tests, including all 44 step references, exported return
  identities, candidate book labels, API input restrictions, ETag reuse, rate-limit
  backoff and crisis-window decision carry-in.
- Existing Python suites: look-ahead 49 checks, incremental 76 checks, comparison
  61 checks; all passed. No `src/` file changed.
- Production build and prerender succeeded. Normal and reduced-motion paths,
  light/dark themes, crisis selection, candidate input, candidate tabs, unavailable
  tickers and offline retention passed Playwright checks.
- Desktop 1440px, mobile 390px and narrow mobile 360px: no horizontal overflow,
  no console errors, and no axe WCAG A/AA violations in the checked states.
- Standard Lighthouse mobile simulation on the local production build:
  **performance 96, accessibility 100, best practices 100**. FCP 1.8 seconds,
  LCP 2.6 seconds, TBT 0ms, CLS 0.002. This is a local lab result, not a guarantee
  of hosted field performance. The raw report is `screenshots/lighthouse.json`.
- A controlled GitHub transport simulation proved that new versions load all
  files from one immutable commit, update metrics without reload, preserve the
  selected ticker/crisis, and retain the last good version after API failure.
  This does not substitute for the missing real-token integration test.
- Dependency installation reports zero known vulnerabilities after updating the
  Lighthouse development dependency. Secrets are excluded from the client bundle
  and git; the read-only GitHub token is a server environment variable only.

## Screenshots

The first [GitHub Actions run](https://github.com/shivakasula/portfolio-allocation-capstone-project/actions/runs/35552702673)
also passed: fresh network export, Python checks, web tests, production build and
bot-authored data publication. Deployment was explicitly skipped because
`VERCEL_TOKEN` is missing. Repository references use the confirmed transferred
location under `shivakasula`; original commit authorship is preserved.

- `screenshots/desktop-viewport.png` and `screenshots/desktop.png`
- `screenshots/mobile-viewport.png` and `screenshots/mobile.png`
- `screenshots/small-mobile-viewport.png` and `screenshots/small-mobile.png`
- `screenshots/mobile-dark.png`
- `screenshots/browser-report.json` and `screenshots/live-report.json`

## External setup still required

1. Vercel accepted the upload but blocked the preview with `TEAM_ACCESS_REQUIRED`:
   the Git commit author lacks access to the selected team. The current git author
   is Shiva Kasula (`sahas.kasula@gmail.com`). The authorized Vercel login is
   `skasulas-4436`, target `sahas7/portfolio-optimization-engine`. Resolve the author
   association/team access in Vercel, then deploy again. No access check was bypassed.
   [Blocked deployment details](https://vercel.com/sahas7/portfolio-optimization-engine/F8cgvU1vPtfZLYbDudK1QhBHooPz).
   The allocated preview URL is not a verified usable preview.
2. Add `GITHUB_READ_TOKEN` to Preview and Production in that Vercel project;
   scope it to the capstone repo with Contents, Issues and Pull requests read-only.
   At the last check the project had no environment variables. The app accurately
   displays an exported snapshot until this connection succeeds.
3. Add `VERCEL_TOKEN` to the capstone repository's Actions secrets. The selected
   `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID` were configured successfully. The
   workflow publishes data but explicitly skips deployment if this token is absent.
4. A hosted live refresh and an automatically deployed push cycle cannot be verified
   until the preceding access and secret configuration are complete.

The other Vercel account and the selected project's existing Git integration were
left untouched. Ask which account to use for future deployments. Local `AGENTS.md`
and `Untitled` remain untracked and are excluded from all commits.
