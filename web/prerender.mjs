import { build } from "vite";
import react from "@vitejs/plugin-react";
import { readFile, writeFile, readdir } from "node:fs/promises";

const read = async (name) =>
  JSON.parse(await readFile(`data/${name}.json`, "utf8"));
const [results, history, manifest] = await Promise.all(
  ["results", "history", "manifest"].map(read),
);
// Render the story from real export data at build time; daily series load separately.
const scalarMeta = (meta) =>
  Object.fromEntries(Object.entries(meta).filter(([, v]) => !Array.isArray(v)));
const snapshot = {
  results: {
    ...results,
    equity: [],
    drawdowns: [],
    correlation: [],
    diversification: [],
    weights: [],
    risk_contributions: [],
    diagnostics: scalarMeta(results.diagnostics),
    benchmark_diagnostics: Object.fromEntries(
      Object.entries(results.benchmark_diagnostics).map(([k, v]) => [
        k,
        scalarMeta(v),
      ]),
    ),
  },
  history: { ...history, commits: [] },
  manifest,
};
await build({
  configFile: false,
  plugins: [react()],
  build: { ssr: "src/server.jsx", outDir: ".prerender", emptyOutDir: true },
  logLevel: "warn",
});
{
  const { render } = await import("./.prerender/server.js");
  let html = await readFile("dist/index.html", "utf8");
  // Critical CSS is small and local. Inline it to remove a blocking request.
  const cssMatch = html.match(
    /<link rel="stylesheet" crossorigin href="([^"]+)">/,
  );
  if (cssMatch)
    html = html.replace(
      cssMatch[0],
      `<style>${await readFile("dist" + cssMatch[1], "utf8")}</style>`,
    );
  const assets = await readdir("dist/assets");
  const fonts = [
    "inter-latin-400-normal-",
    "inter-latin-500-normal-",
    "ibm-plex-mono-latin-400-normal-",
  ];
  html = html.replace(
    "</head>",
    fonts
      .map((prefix) =>
        assets.find((f) => f.startsWith(prefix) && f.endsWith(".woff2")),
      )
      .filter(Boolean)
      .map(
        (f) =>
          `<link rel="preload" href="/assets/${f}" as="font" type="font/woff2" crossorigin>`,
      )
      .join("") + "</head>",
  );
  const bootstrap = JSON.stringify(snapshot)
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
  await writeFile(
    "dist/index.html",
    html.replace(
      '<div id="root"></div>',
      `<div id="root">${render(snapshot)}</div><script>window.__PROJECT_SNAPSHOT__=${bootstrap};</script>`,
    ),
  );
}
