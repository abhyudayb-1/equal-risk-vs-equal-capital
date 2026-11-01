import { chromium, expect } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";

const files = Object.fromEntries(
  await Promise.all(
    ["manifest", "results", "history"].map(async (key) => [
      key,
      JSON.parse(
        await readFile(new URL(`../data/${key}.json`, import.meta.url), "utf8"),
      ),
    ]),
  ),
);
const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ reducedMotion: "reduce" });
let version = files.manifest.version,
  failed = false;
const head = "a".repeat(40),
  requests = [],
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.route("**/api/live*", async (route) => {
  const url = new URL(route.request().url());
  const file = url.searchParams.get("file");
  requests.push({ file, ref: url.searchParams.get("ref") });
  if (failed)
    return route.fulfill({
      json: {
        status: "stale",
        error: "Test rate limit; prior version retained",
        retryAt: Date.now() + 60_000,
      },
    });
  if (file) return route.fulfill({ json: files[file] });
  return route.fulfill({
    json: {
      status: "live",
      head,
      checkedAt: new Date().toISOString(),
      manifest: { ...files.manifest, version },
      activity: files.history.activity,
    },
  });
});
try {
  await page.goto(process.env.QA_URL || "http://127.0.0.1:5173");
  await expect(
    page.getByText("LIVE REPOSITORY", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "NVDA", exact: true }).click();
  await page
    .getByRole("combobox", { name: "Crisis window" })
    .selectOption({ label: "COVID (2020)" });
  // This deliberately changed value is test-only and never written to web/data.
  files.results.metrics[0].cagr = 0.1234;
  version = "test-new-version";
  await page.evaluate(() =>
    document.dispatchEvent(new Event("visibilitychange")),
  );
  await expect(page.locator(".metric-strip")).toContainText("12.34%");
  await expect(
    page.getByRole("button", { name: "NVDA", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("combobox", { name: "Crisis window" }),
  ).toHaveValue("COVID (2020)");
  for (const file of ["results", "history"])
    if (!requests.some((r) => r.file === file && r.ref === head))
      throw new Error(`Version update did not fetch ${file} at immutable ref`);
  failed = true;
  await page.evaluate(() =>
    document.dispatchEvent(new Event("visibilitychange")),
  );
  await expect(
    page.getByText("EXPORTED SNAPSHOT", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".metric-strip")).toContainText("12.34%");
  await expect(page.locator(".stale-banner")).toContainText("Test rate limit");
  if (errors.length) throw new Error(errors.join("\n"));
  const report = {
    passed: true,
    mode: "mocked GitHub transport; real exported initial payload",
    checks: [
      "version change loads all files from one immutable commit",
      "metrics update without reloading",
      "ticker and crisis choices survive refresh",
      "API failure retains the last successful version",
    ],
  };
  await writeFile(
    new URL("../screenshots/live-report.json", import.meta.url),
    JSON.stringify(report, null, 2),
  );
  console.log(report);
} finally {
  await browser.close();
}
