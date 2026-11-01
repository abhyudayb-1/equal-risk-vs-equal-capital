const cache = new Map();
const TTL = 25_000;
let blockedUntil = 0;
const files = new Set(["manifest", "results", "history"]);

export async function github(path, accept = "application/vnd.github+json") {
  const now = Date.now();
  const old = cache.get(path);
  if (old && now - old.time < TTL) return old.data;
  if (now < blockedUntil)
    throw Object.assign(
      new Error("GitHub rate limit; last successful data retained."),
      { retryAt: blockedUntil },
    );
  const headers = {
    Accept: accept,
    Authorization: `Bearer ${process.env.GITHUB_READ_TOKEN}`,
    "X-GitHub-Api-Version": "2022-11-28",
  };
  if (old?.etag) headers["If-None-Match"] = old.etag;
  const response = await fetch(
    `https://api.github.com/repos/${process.env.GITHUB_REPOSITORY || "shivakasula/portfolio-allocation-capstone-project"}/${path}`,
    { headers, signal: AbortSignal.timeout(8000) },
  );
  if (response.status === 304 && old) {
    old.time = now;
    return old.data;
  }
  if (response.status === 429 || response.status === 403) {
    const reset = Number(response.headers.get("x-ratelimit-reset")) * 1000;
    const retry = Number(response.headers.get("retry-after")) * 1000;
    blockedUntil = Math.max(now + (retry || 60_000), reset || 0);
    throw Object.assign(
      new Error("GitHub access or rate limit; retry scheduled."),
      { retryAt: blockedUntil },
    );
  }
  if (!response.ok)
    throw new Error(
      `GitHub returned ${response.status}; last successful data retained.`,
    );
  const data = await response.json();
  cache.set(path, { data, etag: response.headers.get("etag"), time: now });
  // Immutable data versions should not grow a warm function's memory without bound.
  if (cache.size > 40) cache.delete(cache.keys().next().value);
  return data;
}

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "private, no-store");
  if (req.method !== "GET")
    return res.status(405).json({ error: "Method not allowed" });
  if (!process.env.GITHUB_READ_TOKEN)
    return res
      .status(200)
      .json({
        status: "unconfigured",
        error:
          "Live GitHub connection is not configured. Showing the exported snapshot.",
      });
  const url = new URL(req.url, "http://localhost");
  const file = url.searchParams.get("file");
  try {
    if (file) {
      if (!files.has(file))
        return res.status(400).json({ error: "Unknown data file" });
      const ref = url.searchParams.get("ref");
      if (!ref || !/^[a-f0-9]{40}$/.test(ref))
        return res
          .status(400)
          .json({ error: "An immutable commit is required" });
      const data = await github(
        `contents/web/data/${file}.json?ref=${ref}`,
        "application/vnd.github.raw+json",
      );
      return res.status(200).json(data);
    }
    const branch = encodeURIComponent(process.env.GITHUB_DATA_BRANCH || "main");
    // Metadata and files are tied to the same immutable commit, never a moving branch.
    const commits = await github(`commits?sha=${branch}&per_page=12`);
    const head = commits[0].sha;
    const manifest = await github(
      `contents/web/data/manifest.json?ref=${head}`,
      "application/vnd.github.raw+json",
    );
    const pulls = await github("pulls?state=all&per_page=100&sort=updated");
    const issues = await github("issues?state=open&per_page=100");
    const tags = await github("tags?per_page=100");
    return res
      .status(200)
      .json({
        status: "live",
        checkedAt: new Date().toISOString(),
        head,
        manifest,
        activity: {
          commits: commits.map((c) => ({
            sha: c.sha,
            title: c.commit.message.split("\n")[0],
            date: c.commit.author.date,
            author: c.author?.login || c.commit.author.name,
            url: c.html_url,
          })),
          pulls,
          issues: issues.filter((i) => !i.pull_request),
          tags: tags.map((t) => ({
            name: t.name,
            sha: t.commit.sha,
            url: `https://github.com/${process.env.GITHUB_REPOSITORY || "shivakasula/portfolio-allocation-capstone-project"}/tree/${encodeURIComponent(t.name)}`,
          })),
        },
      });
  } catch (error) {
    return res
      .status(503)
      .json({
        status: "stale",
        error: error.message,
        retryAt: error.retryAt || Date.now() + 60_000,
      });
  }
}
