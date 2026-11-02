import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mkdir, writeFile } from "node:fs/promises";

const out = new URL("../screenshots/", import.meta.url);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: "chrome" });
const errors = [],
  reports = [];
const context = await browser.newContext({ reducedMotion: "reduce" });
const page = await context.newPage();
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
try {
  for (const [name, width, height] of [
    ["desktop", 1440, 1000],
    ["mobile", 390, 844],
    ["small-mobile", 360, 800],
  ]) {
    await page.setViewportSize({ width, height });
    await page.goto(process.env.QA_URL || "http://127.0.0.1:5173");
    await page
      .getByRole("heading", { name: "Equal risk. Equal capital." })
      .waitFor();
    await page.locator(".hero .performance").scrollIntoViewIfNeeded();
    await page.locator(".hero .recharts-line-curve").first().waitFor();
    for (const plot of await page
      .locator(".chart-grid > div")
      .all()) {
      await plot.scrollIntoViewIfNeeded();
      await page.waitForTimeout(150);
    }
    await page.evaluate(() => scrollTo(0, 0));
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({
      path: new URL(`${name}-viewport.png`, out).pathname,
    });
    await page.screenshot({
      path: new URL(`${name}.png`, out).pathname,
      fullPage: true,
    });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    );
    if (overflow) throw new Error(`Horizontal overflow at ${width}px`);
    const axe = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    reports.push({
      viewport: name,
      overflow,
      violations: axe.violations.map((v) => ({
        id: v.id,
        impact: v.impact,
        nodes: v.nodes.map((n) => n.target),
      })),
    });
  }
  await page.getByRole("button", { name: "Switch to Dark Mode" }).click();
  await page.screenshot({
    path: new URL("mobile-dark.png", out).pathname,
    fullPage: true,
  });
  reports.push({
    viewport: "mobile-dark",
    violations: (
      await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze()
    ).violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      nodes: v.nodes.map((n) => n.target),
    })),
  });
  await page
    .getByRole("combobox", { name: "Crisis window" })
    .selectOption({ label: "COVID (2020)" });
  await page.getByRole("button", { name: "NVDA", exact: true }).click();
  await page.getByRole("tab", { name: "Realized risk" }).click();
  if (
    !(
      await page
        .getByRole("region", {
          name: "Realized risk before and after the addition",
        })
        .innerText()
    ).includes("baseline")
  )
    throw new Error("Realized table is missing book labels");
  await page.getByRole("tab", { name: "Buy-and-hold comparison" }).click();
  await page
    .getByRole("region", { name: "Historical ticker comparison" })
    .waitFor();
  await page.getByLabel("Candidate ticker").fill("ZZZZ");
  await page
    .getByRole("button", { name: "Analyze ticker", exact: true })
    .click();
  await page.getByRole("heading", { name: "ZZZZ is not available" }).waitFor();
  await page.getByRole("button", { name: "AAPL", exact: true }).click();
  await context.setOffline(true);
  await page.getByText("OFFLINE SNAPSHOT", { exact: true }).waitFor();
  await context.setOffline(false);
  reports.push({
    interactions:
      "theme, crisis, ticker, realized, comparison, unavailable ticker, offline retention passed",
  });
  const animatedPage = await browser.newPage({
    reducedMotion: "no-preference",
    viewport: { width: 1440, height: 1000 },
  });
  animatedPage.on("pageerror", (e) => errors.push(e.message));
  animatedPage.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await animatedPage.goto(process.env.QA_URL || "http://127.0.0.1:5173");
  await animatedPage
    .getByRole("heading", { name: "Equal risk. Equal capital." })
    .waitFor();
  await animatedPage.locator("#findings").scrollIntoViewIfNeeded();
  await animatedPage.waitForTimeout(700);
  await animatedPage.locator("#benchmarks").scrollIntoViewIfNeeded();
  await animatedPage.waitForTimeout(700);
  await expectHeading(animatedPage);
  await animatedPage.close();
  reports.push({
    motion: "normal-motion scroll path and reduced-motion path both exercised",
  });
  await writeFile(
    new URL("browser-report.json", out),
    JSON.stringify({ errors, reports }, null, 2),
  );
  console.log(JSON.stringify({ errors, reports }, null, 2));
  if (errors.length || reports.some((r) => r.violations?.length))
    process.exitCode = 1;
} finally {
  await browser.close();
}

async function expectHeading(page) {
  if (
    !(await page
      .getByRole("heading", { name: "What happens when you add a stock?" })
      .isVisible())
  )
    throw new Error("Animated page lost its content");
}
