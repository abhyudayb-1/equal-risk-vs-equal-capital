import React, { useMemo } from "react";
import {
  Area,
  AreaChart,
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
  Legend,
} from "recharts";
import { useReducedMotion } from "framer-motion";
import { colors, pct, num, date } from "./data";
import { windowRows } from "./chart-data.js";

export default function Chart({
  rows,
  keys,
  percent = false,
  stack = false,
  crises = [],
  height = 290,
  domain,
  suffix = "",
  summary,
  window,
  name = "date",
  hero = false,
  decisions = false,
  labels = {},
}) {
  const reduced = useReducedMotion();
  const data = useMemo(
    () =>
      windowRows(rows, window, decisions).map((r) => ({
        ...r,
        stamp: name === "date" ? new Date(r.date).getTime() : r[name],
      })),
    [rows, window, name, decisions],
  );
  const format = (value) => (percent ? pct(value, 0) : num(value, 1) + suffix);
  const xFormat = (n) =>
    name === "date"
      ? window
        ? new Date(n).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            timeZone: "UTC",
          })
        : new Date(n).getUTCFullYear()
      : pct(n, 0);
  const Plot = stack ? AreaChart : LineChart;
  return (
    <figure className={`plot ${hero ? "hero-plot" : ""}`} aria-label={summary}>
      <div style={{ height, width: "100%", minWidth: 0 }}>
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <Plot
            data={data}
            margin={{ top: 12, right: 12, left: -12, bottom: 2 }}
            accessibilityLayer
          >
            <CartesianGrid
              vertical={false}
              stroke="var(--line)"
              strokeDasharray="3 5"
            />
            <XAxis
              dataKey="stamp"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={xFormat}
              tick={{ fill: "var(--muted)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              minTickGap={45}
            />
            <YAxis
              tickFormatter={format}
              domain={domain || ["auto", "auto"]}
              tick={{ fill: "var(--muted)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={64}
            />
            {name === "date" &&
              crises.map((c) => (
                <ReferenceArea
                  key={c.name}
                  x1={Math.max(
                    new Date(c.start).getTime(),
                    data[0]?.stamp || 0,
                  )}
                  x2={Math.min(
                    new Date(c.end).getTime(),
                    data.at(-1)?.stamp || 0,
                  )}
                  fill="var(--red)"
                  fillOpacity={0.06}
                  ifOverflow="hidden"
                />
              ))}
            {domain?.[0] < 0 && (
              <ReferenceLine
                y={0}
                stroke="var(--muted)"
                strokeDasharray="4 4"
              />
            )}
            <Tooltip
              labelFormatter={(n) =>
                name === "date"
                  ? date(new Date(n).toISOString())
                  : `Candidate allocation ${pct(n, 0)}`
              }
              formatter={(v, key) => [percent ? pct(v) : num(v) + suffix, key]}
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--line)",
                borderRadius: 4,
                color: "var(--ink)",
                fontSize: 12,
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
              iconType="plainline"
            />
            {keys.map((key, i) =>
              stack ? (
                <Area
                  key={key}
                  type={decisions ? "stepAfter" : "linear"}
                  dataKey={key}
                  name={labels[key] || key}
                  stackId="risk"
                  stroke={colors[i]}
                  fill={colors[i]}
                  fillOpacity={0.78}
                  isAnimationActive={!reduced}
                  animationDuration={450}
                />
              ) : (
                <Line
                  key={key}
                  type={decisions ? "stepAfter" : "linear"}
                  dataKey={key}
                  name={labels[key] || key}
                  stroke={colors[i]}
                  strokeWidth={hero && i === 0 ? 2.5 : 1.8}
                  dot={false}
                  isAnimationActive={!reduced}
                  animationDuration={450}
                />
              ),
            )}
          </Plot>
        </ResponsiveContainer>
      </div>
      <figcaption>{summary}</figcaption>
    </figure>
  );
}
