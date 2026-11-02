import test from "node:test";
import assert from "node:assert/strict";
import { windowRows } from "../src/chart-data.js";

test("crisis decision charts carry the entering book and never a future rebalance", () => {
  const rows = [
    { date: "2020-01-31", leverage: 2.3 },
    { date: "2020-02-28", leverage: 1.8 },
    { date: "2020-03-31", leverage: 0.6 },
  ];
  const window = { start: "2020-02-19", end: "2020-03-23" };
  const actual = windowRows(rows, window, true);
  assert.deepEqual(actual, [
    { date: window.start, leverage: 2.3 },
    rows[1],
    { date: window.end, leverage: 1.8 },
  ]);
  assert.equal(rows.length, 3);
  assert.deepEqual(
    windowRows(rows, window, false),
    [rows[1]],
    "observations must not be forward-filled like decisions",
  );
  assert.deepEqual(
    windowRows(rows, { start: "2019-01-01", end: "2019-02-01" }, true),
    [],
  );
});
