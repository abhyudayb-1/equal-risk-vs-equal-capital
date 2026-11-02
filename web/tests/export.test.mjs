import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const read = async (name) =>
  JSON.parse(
    await readFile(new URL(`../data/${name}.json`, import.meta.url), "utf8"),
  );
const [r, h, m] = await Promise.all(
  ["results", "history", "manifest"].map(read),
);

test("export preserves all daily returns, finite numbers and consistent provenance", () => {
  assert.equal(r.metrics[0].n_periods, r.equity.length);
  assert.equal(r.equity.length, r.drawdowns.length);
  assert.equal(m.source_commit, r.source_commit);
  assert.equal(m.market_data_through, r.equity.at(-1).date);
  assert.equal(r.leverage.length, r.weights.length);
  assert.equal(r.leverage.length, r.risk_contributions.length);
  assert.ok(
    r.start > r.weights[0].date,
    "first decision cannot earn its own day",
  );
  for (const metrics of r.metrics) {
    const last = r.equity.at(-1)[metrics.name];
    assert.ok(Math.abs(last - 1 - metrics.total_return) < 1e-10);
    assert.ok(
      Math.abs(
        Math.min(...r.drawdowns.map((d) => d[metrics.name])) -
          metrics.max_drawdown,
      ) < 1e-10,
    );
  }
});

test("allocation shares sum to one and decisions satisfy configured leverage bounds", () => {
  for (const row of [...r.weights, ...r.risk_contributions])
    assert.ok(
      Math.abs(r.config.UNIVERSE.reduce((sum, t) => sum + row[t], 0) - 1) <
        1e-6,
    );
  for (const row of r.leverage)
    assert.ok(
      row.leverage >= r.config.LEVERAGE_CAP[0] &&
        row.leverage <= r.config.LEVERAGE_CAP[1],
    );
});

// The candidate half of this test went with the position-analysis product.
// The step-numbering assertion is deliberately NOT restored as a contiguous
// 1..N range: steps 36-44 were removed from TASKS.md when the project was
// narrowed, so the exported phases are no longer gap-free and asserting they
// are would fail for the right reason in the wrong place. What still matters
// is that every step the export DOES carry points at a real commit.
test("every exported step points at a commit that exists in the log", () => {
  const steps = h.phases.flatMap((p) => p.steps);
  assert.ok(steps.length > 0, "export carried no steps at all");
  for (const step of steps)
    assert.ok(
      h.commits.some((c) => c.sha === step.commit?.sha),
      `step ${step.number} lacks source evidence`,
    );
});
