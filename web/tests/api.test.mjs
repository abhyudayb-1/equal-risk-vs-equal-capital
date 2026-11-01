import test from "node:test";
import assert from "node:assert/strict";
import handler, { github } from "../api/live.js";

function response() {
  return {
    code: 200,
    headers: {},
    setHeader(k, v) {
      this.headers[k] = v;
    },
    status(n) {
      this.code = n;
      return this;
    },
    json(data) {
      this.data = data;
      return this;
    },
  };
}

test("missing configuration is explicit and never claims live", async () => {
  const old = process.env.GITHUB_READ_TOKEN;
  delete process.env.GITHUB_READ_TOKEN;
  const res = response();
  await handler({ method: "GET", url: "/api/live" }, res);
  assert.equal(res.data.status, "unconfigured");
  assert.match(res.headers["Cache-Control"], /private/);
  if (old) process.env.GITHUB_READ_TOKEN = old;
});

test("route rejects writes, arbitrary paths and mutable data refs", async () => {
  const old = process.env.GITHUB_READ_TOKEN;
  process.env.GITHUB_READ_TOKEN = "test-only";
  for (const [method, url, code] of [
    ["POST", "/api/live", 405],
    ["GET", "/api/live?file=../../secrets&ref=main", 400],
    ["GET", "/api/live?file=results&ref=main", 400],
  ]) {
    const res = response();
    await handler({ method, url }, res);
    assert.equal(res.code, code);
  }
  if (old) process.env.GITHUB_READ_TOKEN = old;
  else delete process.env.GITHUB_READ_TOKEN;
});

test("conditional upstream requests retain data on 304 and honor rate-limit reset", async () => {
  const originalFetch = globalThis.fetch,
    originalNow = Date.now;
  let now = 1_000_000,
    calls = 0;
  Date.now = () => now;
  globalThis.fetch = async (_url, opts) => {
    assert.equal(
      _url,
      `https://api.github.com/repos/${process.env.GITHUB_REPOSITORY || "shivakasula/portfolio-allocation-capstone-project"}/test-endpoint`,
    );
    calls++;
    if (calls === 1)
      return new Response(JSON.stringify({ version: "first" }), {
        status: 200,
        headers: { etag: '"v1"' },
      });
    assert.equal(opts.headers["If-None-Match"], '"v1"');
    return new Response(null, { status: 304 });
  };
  try {
    assert.deepEqual(await github("test-endpoint"), { version: "first" });
    await github("test-endpoint");
    assert.equal(calls, 1);
    now += 26_000;
    assert.deepEqual(await github("test-endpoint"), { version: "first" });
    assert.equal(calls, 2);
    globalThis.fetch = async () =>
      new Response(null, { status: 429, headers: { "retry-after": "120" } });
    await assert.rejects(
      github("limited-endpoint"),
      (e) => e.retryAt === now + 120_000,
    );
    globalThis.fetch = async () => {
      throw new Error("must not fetch during backoff");
    };
    await assert.rejects(github("other-endpoint"), /rate limit/);
  } finally {
    globalThis.fetch = originalFetch;
    Date.now = originalNow;
  }
});
