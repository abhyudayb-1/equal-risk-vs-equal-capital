import React, { Suspense, useEffect, useRef, useState } from "react";
import { motion, useReducedMotion, animate } from "framer-motion";
import {
  Activity,
  ArrowDown,
  ArrowUpRight,
  Check,
  ChevronRight,
  Download,
  ExternalLink,
  GitBranch,
  Github,
  Moon,
  Sun,
  ShieldCheck,
  AlertCircle,
} from "lucide-react";
import { useProjectData, pct, num, date } from "./data";

import Chart from "./DeferredChart.jsx";
const ENGINE = "Risk parity + vol target";
const repoURL =
  "https://github.com/shivakasula/portfolio-allocation-capstone-project";
const cleanText = (text) => text.replace(/[`*]/g, "");

function Count({ value, percent = true, digits = 2 }) {
  const ref = useRef();
  const reduced = useReducedMotion();
  useEffect(() => {
    if (reduced || !Number.isFinite(value)) {
      ref.current.textContent = percent
        ? pct(value, digits)
        : num(value, digits);
      return;
    }
    const control = animate(0, value, {
      duration: 0.7,
      onUpdate: (v) => {
        if (ref.current)
          ref.current.textContent = percent ? pct(v, digits) : num(v, digits);
      },
    });
    return () => control.stop();
  }, [value, reduced, percent, digits]);
  return (
    <span ref={ref}>{percent ? pct(value, digits) : num(value, digits)}</span>
  );
}

const TabContext = React.createContext("overview");

const TABS = [
  ["overview", "Overview"],
  ["findings", "Findings"],
  ["charts", "The data"],
  ["benchmarks", "Benchmarks"],
  ["hardening", "Hardening"],
  ["method", "Method"],
];

function Section({ id, number, title, note, children }) {
  const active = React.useContext(TabContext);
  if (active !== id) return null;
  return (
    <section id={id} className="section reveal">
      <div className="section-head">
        <div>
          <span className="eyebrow">{number} / RESEARCH NOTEBOOK</span>
          <h2>{title}</h2>
        </div>
        {note && <p>{note}</p>}
      </div>
      {children}
    </section>
  );
}

function Table({ rows, columns, label }) {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>
      <table>
        <caption className="sr-only">{label}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={`${row.window || ""}:${row.name || row.strategy || i}`}>
              {columns.map((c, j) =>
                j === 0 ? (
                  <th scope="row" key={c.key}>
                    {c.format ? c.format(row[c.key]) : row[c.key]}
                  </th>
                ) : (
                  <td key={c.key}>
                    {c.format
                      ? c.format(row[c.key])
                      : String(row[c.key] ?? "Unavailable")}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Duplicated in the boot script in index.html, which has to run standalone in
// <head> before any module loads. Keep the two in step.
const THEME_KEY = "risk-theme";
const THEME_COLORS = { light: "#f7f9f8", dark: "#111815" };
const readBootTheme = () =>
  typeof document !== "undefined" &&
  document.documentElement.dataset.theme === "dark"
    ? "dark"
    : "light";

export default function App() {
  const state = useProjectData();
  // Seeded from the attribute the boot script in index.html already set, not
  // from localStorage directly: that script is the single place the stored
  // preference is read, and it runs before first paint. On the server there is
  // no document, so SSR always renders the light-mode default.
  const [theme, setTheme] = useState(readBootTheme);
  const [tab, setTabState] = useState("overview");
  const setTab = (next) => {
    setTabState(next);
    try {
      window.history.replaceState(
        null,
        "",
        next === "overview" ? location.pathname : `#${next}`,
      );
    } catch {}
  };
  useEffect(() => {
    const fromHash = () => {
      const h = location.hash.slice(1);
      if (TABS.some(([id]) => id === h)) setTabState(h);
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, []);
  const [crisis, setCrisis] = useState("full");
  const [metric, setMetric] = useState("equity");
  const [sort, setSort] = useState("cagr");
  const reduced = useReducedMotion();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    // Keep the browser chrome (address bar, task switcher) in step with the page.
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", THEME_COLORS[theme]);
  }, [theme]);
  // Written only here, on a real click - never on mount. A visitor who has
  // never touched the toggle leaves no stored preference, so "no preference"
  // stays distinguishable from "chose light".
  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {}
  };
  useEffect(() => {
    if (!state.data || reduced) return;
    let context,
      cancelled = false;
    const begin = () =>
      Promise.all([import("gsap"), import("gsap/ScrollTrigger")]).then(
        ([g, s]) => {
          if (cancelled) return;
          const gsap = g.gsap;
          gsap.registerPlugin(s.ScrollTrigger);
          context = gsap.context(() => {
            gsap.utils.toArray(".section-head").forEach((el) =>
              gsap.fromTo(
                el,
                { y: 12 },
                {
                  y: 0,
                  duration: 0.45,
                  scrollTrigger: { trigger: el, start: "top 94%", once: true },
                },
              ),
            );
          });
        },
      );
    globalThis.addEventListener("scroll", begin, { once: true, passive: true });
    return () => {
      cancelled = true;
      globalThis.removeEventListener("scroll", begin);
      context?.revert();
    };
  }, [Boolean(state.data), reduced]);
  if (!state.data)
    return (
      <main className="loading">
        <Activity size={32} />
        <h1>Risk Parity Lab</h1>
        <p role="status">
          {state.status === "error"
            ? state.error
            : "Loading the research export…"}
        </p>
        {state.status === "error" && (
          <button onClick={() => location.reload()}>Retry</button>
        )}
      </main>
    );
  const { results: r, history, manifest } = state.data;
  const engine = r.metrics.find((m) => m.name === ENGINE);
  const balanced = r.metrics.find((m) => m.name === "60/40");
  const selectedWindow = r.mechanisms.find((w) => w.name === crisis);
  const covid = r.mechanisms.find((w) => w.name.includes("COVID"));
  const shock = r.mechanisms.find((w) => w.start.startsWith("2022"));
  const covidResult = r.crisis.find(
    (c) => c.window === covid.name && c.strategy === ENGINE,
  );
  const worst = r.crisis
    .filter((c) => c.strategy === ENGINE)
    .reduce((a, b) => (a.max_drawdown < b.max_drawdown ? a : b));
  const dataOld =
    Date.now() - new Date(r.market_data_through).getTime() > 7 * 86400000;
  const gap = balanced.cagr - engine.cagr;
  const tableRows = [...r.metrics].sort((a, b) => b[sort] - a[sort]);
  const coreDiag = r.diagnostics;
  return (
    <>
      <a href="#main" className="skip-link">
        Skip to research
      </a>
      <header className="topbar">
        <a href="#main" className="brand">
          <Activity size={24} />
          <span>
            Risk Parity<span className="brand-lab"> / LAB</span>
          </span>
        </a>
        <div className="top-actions">
          <a
            className="icon-button"
            href={repoURL}
            target="_blank"
            rel="noreferrer"
            title="GitHub repository"
            aria-label="GitHub repository"
          >
            <Github size={19} />
          </a>
          {/*
            Renders both labels and lets CSS hide one, rather than branching on
            `theme`. That keeps the markup identical on the server and on a
            client with a stored dark preference, so hydration never mismatches
            and the boot script's theme survives first paint. `display: none`
            also drops the hidden half from the accessibility tree, so the
            accessible name is just the visible label - which is what makes the
            name contain the visible text, per WCAG 2.5.3.
          */}
          <button type="button" className="theme-toggle" onClick={toggleTheme}>
            <Moon size={17} className="when-light" aria-hidden="true" />
            <Sun size={17} className="when-dark" aria-hidden="true" />
            <span className="sr-only">Switch to </span>
            <span className="when-light">Dark Mode</span>
            <span className="when-dark">Light Mode</span>
          </button>
        </div>
      </header>
      <div className="tabbar" role="tablist" aria-label="Sections">
        {TABS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`tab-${id}`}
            aria-selected={tab === id}
            aria-controls="main"
            className="tab-button"
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <TabContext.Provider value={tab}>
        <main id="main" role="tabpanel" aria-labelledby={`tab-${tab}`}>
          {tab === "overview" && (
            <>
              <div className="statusbar">
                <span>
                  <span className={`status-dot ${state.status}`} />
                  {state.status === "live"
                    ? "LIVE REPOSITORY"
                    : state.status === "offline"
                      ? "OFFLINE SNAPSHOT"
                      : "EXPORTED SNAPSHOT"}
                </span>
                <span>
                  Last {state.checkedAt ? "checked" : "exported"}{" "}
                  {date(state.checkedAt || manifest.generated_at)} ·{" "}
                  {new Date(
                    state.checkedAt || manifest.generated_at,
                  ).toLocaleTimeString("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                    timeZone: "UTC",
                  })}{" "}
                  UTC
                </span>
                <a href={`${repoURL}/commit/${r.source_commit}`}>
                  <GitBranch size={13} />
                  {r.source_commit.slice(0, 7)}
                  <ArrowUpRight size={13} />
                </a>
              </div>
              {
                <div className="stale-banner" role="status">
                  <AlertCircle size={15} />
                  <span>
                    {state.error ||
                      (dataOld
                        ? "Market data is older than a week."
                        : "Research export.")}{" "}
                    Market observations through {date(r.market_data_through)}.
                  </span>
                </div>
              }
              <section className="hero">
                <motion.div
                  initial={false}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45 }}
                >
                  <div className="hero-meta">
                    <span className="eyebrow">
                      PORTFOLIO RESEARCH / PROJECT 16
                    </span>
                    <span className="edition">
                      {history.tags.at(-1)?.name} · Research edition
                    </span>
                  </div>
                  <h1>
                    Risk parity.
                    <br />
                    <span>The whole picture.</span>
                  </h1>
                  <div className="hero-bottom">
                    <p>
                      Equal risk. Unequal outcomes. Our volatility-targeted
                      portfolio {gap > 0 ? "trailed" : "outperformed"} a simple
                      60/40 allocation. Here is the evidence, including where
                      the model broke down.
                    </p>
                    <a
                      className="text-link"
                      href="#findings"
                      onClick={() => setTab("findings")}
                    >
                      Read the findings <ArrowDown size={16} />
                    </a>
                  </div>
                </motion.div>
                <div className="metric-strip">
                  <div>
                    <span>RISK PARITY CAGR</span>
                    <strong className="accent">
                      <Count value={engine.cagr} />
                    </strong>
                    <small>
                      {date(r.start)} – {date(r.market_data_through)}
                    </small>
                  </div>
                  <div>
                    <span>60/40 CAGR</span>
                    <strong>
                      <Count value={balanced.cagr} />
                    </strong>
                    <small>
                      {num(Math.abs(gap) * 100)} pp{" "}
                      {gap > 0 ? "ahead" : "behind"} annually
                    </small>
                  </div>
                  <div>
                    <span>REALIZED VOLATILITY</span>
                    <strong>
                      <Count value={engine.annual_volatility} />
                    </strong>
                    <small>
                      Against a {pct(r.config.TARGET_VOL, 0)} target
                    </small>
                  </div>
                  <div>
                    <span>MAXIMUM DRAWDOWN</span>
                    <strong className="negative">
                      <Count value={engine.max_drawdown} />
                    </strong>
                    <small>Deepest crisis: {worst.window}</small>
                  </div>
                </div>
                <div className="performance">
                  <div className="plot-title">
                    <div>
                      <span className="eyebrow">THE LONG VIEW</span>
                      <h2 className="plot-heading">
                        {metric === "equity"
                          ? "Growth of one dollar"
                          : "Distance from the high-water mark"}
                      </h2>
                    </div>
                    <div
                      className="segmented"
                      role="group"
                      aria-label="Performance measure"
                    >
                      <button
                        aria-pressed={metric === "equity"}
                        onClick={() => setMetric("equity")}
                      >
                        Growth
                      </button>
                      <button
                        aria-pressed={metric === "drawdowns"}
                        onClick={() => setMetric("drawdowns")}
                      >
                        Drawdown
                      </button>
                    </div>
                  </div>
                  <Suspense fallback={<div className="chart-skeleton" />}>
                    <Chart
                      rows={r[metric]}
                      keys={[ENGINE, "60/40"]}
                      crises={r.mechanisms}
                      percent={metric === "drawdowns"}
                      suffix={metric === "equity" ? "×" : ""}
                      height={270}
                      hero
                      summary="Financing charged to risk parity. The benchmark is unlevered. Shaded periods mark the configured crisis windows."
                    />
                  </Suspense>
                </div>
              </section>
            </>
          )}
          <Section
            id="findings"
            number="01"
            title="Two crises. Two different failures."
            note="The largest loss and the failure of diversification are not the same event."
          >
            <div className="findings">
              <article>
                <div className="finding-label">
                  <span className="eyebrow">{covid.name.toUpperCase()}</span>
                  <span>01</span>
                </div>
                <h3>
                  The calm before
                  <br />
                  the leverage.
                </h3>
                <strong className="finding-number negative">
                  {pct(covidResult.max_drawdown)}
                </strong>
                <p>
                  The book entered the shock at{" "}
                  <b>{num(covid.entry_leverage)}× leverage</b>. A trailing
                  estimator saw calm conditions; volatility arrived before the
                  overlay could reduce exposure.
                </p>
                <p className="meta">
                  Volatility targeting controls realized risk. It cannot
                  forecast a crash.
                </p>
                <a
                  href="#charts"
                  onClick={() => {
                    setCrisis(covid.name);
                    setTab("charts");
                  }}
                  className="text-link"
                >
                  Inspect the COVID window <ArrowUpRight size={16} />
                </a>
              </article>
              <article>
                <div className="finding-label">
                  <span className="eyebrow">{shock.name.toUpperCase()}</span>
                  <span>02</span>
                </div>
                <h3>
                  The hedge lost more
                  <br />
                  than the risk.
                </h3>
                <div className="sleeve-losses">
                  <span>
                    <small>TLT</small>
                    <strong>{pct(shock.sleeve_returns.TLT, 1)}</strong>
                  </span>
                  <span>
                    <small>SPY</small>
                    <strong>{pct(shock.sleeve_returns.SPY, 1)}</strong>
                  </span>
                </div>
                <p>
                  Rolling stock/bond correlation averaged{" "}
                  <b>{num(shock.correlation_mean)}</b>, positive on{" "}
                  {pct(shock.correlation_positive_fraction, 0)} of days.
                  Diversification weakened without a persistent positive
                  correlation.
                </p>
                <p className="meta">
                  Sleeve returns measured from the opening peak close, excluding
                  that day's return.
                </p>
                <a
                  href="#charts"
                  onClick={() => {
                    setCrisis(shock.name);
                    setTab("charts");
                  }}
                  className="text-link"
                >
                  Inspect the rate shock <ArrowUpRight size={16} />
                </a>
              </article>
            </div>
          </Section>
          <Section
            id="charts"
            number="02"
            title="Follow the risk, not just the capital."
            note="The same allocation, seen through money, risk, leverage and diversification."
          >
            <div className="chart-toolbar">
              <span className="meta">
                {selectedWindow
                  ? `${date(selectedWindow.start)} to ${date(selectedWindow.end)}`
                  : `${date(r.start)} to ${date(r.market_data_through)}`}
              </span>
              <label>
                Crisis window
                <select
                  aria-label="Crisis window"
                  value={crisis}
                  onChange={(e) => setCrisis(e.target.value)}
                >
                  <option value="full">Full history</option>
                  {r.mechanisms.map((w) => (
                    <option key={w.name}>{w.name}</option>
                  ))}
                </select>
              </label>
              <a
                href={
                  state.head
                    ? `/api/live?file=results&ref=${state.head}`
                    : "/data/results.json"
                }
                download="risk-parity-results.json"
                className="icon-button"
                title="Download exported data"
                aria-label="Download exported data"
              >
                <Download size={18} />
              </a>
            </div>
            <Suspense fallback={<div className="chart-skeleton" />}>
              <div className="chart-grid">
                <div>
                  <h3>Capital allocation</h3>
                  <Chart
                    rows={r.weights}
                    decisions
                    keys={r.config.UNIVERSE}
                    window={selectedWindow}
                    stack
                    percent
                    domain={[0, 1]}
                    summary="Share of money at each monthly rebalance. Low-volatility sleeves attract more capital."
                  />
                </div>
                <div>
                  <h3>Risk contribution</h3>
                  <Chart
                    rows={r.risk_contributions}
                    decisions
                    keys={r.config.UNIVERSE}
                    window={selectedWindow}
                    stack
                    percent
                    domain={[0, 1]}
                    summary="Share of modeled portfolio volatility. The solver equalizes risk across the sleeves."
                  />
                </div>
                <div className="full-width">
                  <h3>Leverage through the cycle</h3>
                  <Chart
                    rows={r.leverage}
                    decisions
                    keys={["leverage"]}
                    window={selectedWindow}
                    crises={r.mechanisms}
                    domain={[0, r.config.LEVERAGE_CAP[1]]}
                    suffix="×"
                    height={250}
                    summary={`Decision-date leverage, financed at DTB3. Allowed range ${num(r.config.LEVERAGE_CAP[0])}–${num(r.config.LEVERAGE_CAP[1])}×; decisions take effect after the close.`}
                  />
                </div>
                <div>
                  <h3>Stock / bond correlation</h3>
                  <Chart
                    rows={r.correlation}
                    keys={["correlation"]}
                    window={selectedWindow}
                    domain={[-1, 1]}
                    summary={`${r.config.CORRELATION_WINDOW_DAYS}-day rolling SPY/TLT correlation. A sign change alone does not explain the rate shock.`}
                  />
                </div>
                <div>
                  <h3>Diversification ratio</h3>
                  <Chart
                    rows={r.diversification}
                    keys={["ratio"]}
                    window={selectedWindow}
                    suffix="×"
                    summary="Weighted sleeve volatility divided by portfolio volatility. Lower values indicate less diversification."
                  />
                </div>
              </div>
            </Suspense>
          </Section>
          <Section
            id="benchmarks"
            number="03"
            title="A fair comparison can be uncomfortable."
            note="All strategies share the engine's covariance estimator and trading calendar."
          >
            <div className="table-toolbar">
              <span className="meta">
                Full-sample results · Financing included
              </span>
              <label>
                Sort by
                <select value={sort} onChange={(e) => setSort(e.target.value)}>
                  <option value="cagr">CAGR</option>
                  <option value="sharpe_ratio">Return / vol</option>
                  <option value="max_drawdown">Max drawdown</option>
                </select>
              </label>
            </div>
            <Table
              label="Full-sample benchmark comparison"
              rows={tableRows}
              columns={[
                { key: "name", label: "Strategy" },
                { key: "cagr", label: "CAGR", format: pct },
                { key: "annual_volatility", label: "Volatility", format: pct },
                { key: "sharpe_ratio", label: "Return / vol", format: num },
                { key: "max_drawdown", label: "Max drawdown", format: pct },
                {
                  key: "mean_leverage",
                  label: "Mean leverage",
                  format: (v) => num(v) + "×",
                },
              ]}
            />
            <p className="context-note">
              Return / vol uses a zero risk-free baseline for every row. It is
              not an excess-return Sharpe. Benchmarks are unlevered; the
              risk-parity overlay borrows at the bill rate.
            </p>
            <div className="financing-note">
              <span className="eyebrow">THE COST OF BORROWING</span>
              <p>
                Omitting financing lifts CAGR from{" "}
                <b>{pct(r.financing["Financing charged"].cagr)}</b> to{" "}
                <b>{pct(r.financing["Financing omitted"].cagr)}</b>:{" "}
                <strong>
                  {num(
                    100 *
                      (r.financing["Financing omitted"].cagr -
                        r.financing["Financing charged"].cagr),
                  )}{" "}
                  percentage points
                </strong>{" "}
                of annualized return that the funded strategy does not earn.
              </p>
            </div>
            <details>
              <summary>
                Crisis comparison and benchmark health{" "}
                <ChevronRight size={16} />
              </summary>
              <Table
                label="Crisis losses and bond exposures"
                rows={r.crisis}
                columns={[
                  { key: "window", label: "Crisis" },
                  { key: "strategy", label: "Strategy" },
                  { key: "max_drawdown", label: "Max drawdown", format: pct },
                  {
                    key: "capital_weight_bonds",
                    label: "TLT capital",
                    format: pct,
                  },
                  {
                    key: "mean_leverage",
                    label: "Leverage",
                    format: (v) => num(v) + "×",
                  },
                ]}
              />
              <pre>{JSON.stringify(r.benchmark_diagnostics, null, 2)}</pre>
            </details>
          </Section>
          <Section
            id="hardening"
            number="04"
            title="A result is only as good as its guardrails."
            note="Tests attack the assumptions that can produce plausible but wrong returns."
          >
            <div className="health-bar">
              <ShieldCheck size={24} />
              <div>
                <span className="eyebrow">ENGINE RUN HEALTH</span>
                <strong>{coreDiag.run_health}</strong>
              </div>
              <span>{r.leverage.length} rebalances</span>
              <span>{num(coreDiag.solver_failures, 0)} solver failures</span>
            </div>
            <div className="guardrails">
              <article>
                <Check />
                <h3>No future information</h3>
                <p>
                  Future price spikes, truncated history and changed financing
                  rates cannot move earlier decisions.
                </p>
              </article>
              <article>
                <Check />
                <h3>A test that can fail</h3>
                <p>
                  Removing the time guard must make the tests fail. Positive
                  controls prove the engine is responding to its input.
                </p>
              </article>
              <article>
                <Check />
                <h3>No same-day hindsight</h3>
                <p>
                  A choice at today's close cannot earn today's return. The
                  mutation test deliberately breaks that alignment.
                </p>
              </article>
            </div>
            <details>
              <summary>
                Complete engine diagnostics <ChevronRight size={16} />
              </summary>
              <pre>{r.run_summary}</pre>
            </details>
            <a
              className="text-link"
              href={`${repoURL}/blob/${r.source_commit}/tests/test_lookahead.py`}
            >
              Read the look-ahead tests <ArrowUpRight size={15} />
            </a>
          </Section>
          <Section
            id="method"
            number="05"
            title="The method. And its boundaries."
          >
            <div className="method-grid">
              <article>
                <span className="eyebrow">ESTIMATE</span>
                <h3>Recent history weighs more.</h3>
                <p>
                  Covariance uses a {r.config.COV_EWMA_HALFLIFE_DAYS}-day EWMA
                  halflife and {pct(r.config.COV_SHRINKAGE, 0)} shrinkage.
                  Correlation-space repair preserves individual sleeve
                  volatilities.
                </p>
              </article>
              <article>
                <span className="eyebrow">ALLOCATE</span>
                <h3>Balance risk contributions.</h3>
                <p>
                  The constrained solver targets equal contributions to
                  portfolio volatility, with independent oracle checks and
                  explicit convergence diagnostics.
                </p>
              </article>
              <article>
                <span className="eyebrow">SIZE & FINANCE</span>
                <h3>Scale the book, pay for it.</h3>
                <p>
                  A {r.config.VOL_EWMA_HALFLIFE_DAYS}-day volatility estimator
                  targets {pct(r.config.TARGET_VOL, 0)} volatility, with
                  leverage bounded at {num(r.config.LEVERAGE_CAP[0])}–
                  {num(r.config.LEVERAGE_CAP[1])}×.
                </p>
              </article>
            </div>
            <div className="limitations">
              <h3>What the model does not claim</h3>
              <p>
                No transaction costs, bid-ask spreads or borrowing spread above
                DTB3. Adjusted closes include distributions. History starts at{" "}
                {date(r.config.DATA_START_DATE)}, constrained by DBC's
                inception. The volatility overlay lags shocks, and the asset
                universe is fixed at four sleeves.
              </p>
            </div>
          </Section>
        </main>
      </TabContext.Provider>
      <footer>
        <a href="#main" className="brand">
          <Activity size={21} />
          Risk Parity / LAB
        </a>
        <p>Research with the uncomfortable results left in.</p>
        <a href={repoURL}>
          Source & methodology <ExternalLink size={14} />
        </a>
      </footer>
    </>
  );
}
