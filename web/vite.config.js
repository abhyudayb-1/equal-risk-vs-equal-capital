import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import live from "./api/live.js";

export default defineConfig(({ mode }) => {
  Object.assign(process.env, loadEnv(mode, process.cwd(), "GITHUB_"));
  return {
    plugins: [
      react(),
      tailwindcss(),
      {
        name: "local-data-and-api",
        configureServer(server) {
          server.middlewares.use(async (req, res, next) => {
            const url = new URL(req.url, "http://localhost");
            if (url.pathname === "/api/live") {
              res.status = (code) => {
                res.statusCode = code;
                return res;
              };
              res.json = (data) => {
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(data));
              };
              await live(req, res);
              return;
            }
            if (
              /^\/data\/(manifest|results|history)\.json$/.test(url.pathname)
            ) {
              try {
                res.setHeader("Content-Type", "application/json");
                res.end(await readFile(resolve("." + url.pathname)));
              } catch {
                res.statusCode = 404;
                res.end("Missing export");
              }
              return;
            }
            next();
          });
        },
        async generateBundle() {
          for (const name of ["manifest", "results", "history"]) {
            this.emitFile({
              type: "asset",
              fileName: `data/${name}.json`,
              source: await readFile(`data/${name}.json`),
            });
          }
        },
      },
    ],
    build: { rollupOptions: { output: { manualChunks: undefined } } },
  };
});
