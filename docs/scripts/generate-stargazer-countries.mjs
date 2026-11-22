#!/usr/bin/env node
/**
 * Generator (weekly in .github/workflows/community-data.yml, not
 * pre-build): pages through every stargazer of
 * the deepeval repo with the GitHub GraphQL API, 100 per request, resolves
 * each free-text profile location to a country, and writes
 * `lib/generated/stargazer-countries.json`:
 *
 *   [ { "code": "US", "count": 2100 }, … ]
 *
 * sorted by count desc. Locations that do not resolve are skipped, so the
 * counts are a floor. Used by the community page globe (stars per country).
 *
 * It takes a few minutes for ~20k stars, which is why it is not part of
 * `prebuild`; the workflow commits the output to main once a week.
 *
 * Failure modes (exit 0): no token, or a request still failing after
 * retries, keeps the previous JSON file.
 *
 * Env:
 *   GITHUB_TOKEN  Required (the GraphQL API needs auth).
 *
 * Run: `npm run stargazer-countries`.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { resolve } from "./community/location-to-country.mjs";

const OUTPUT = "lib/generated/stargazer-countries.json";
const QUERY = `query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    stargazers(first: 100, after: $after) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { location }
    }
  }
}`;

function readGitConfig() {
  const src = readFileSync("lib/shared.ts", "utf8");
  const user = src.match(/user:\s*['"]([^'"]+)['"]/)?.[1];
  const repo = src.match(/repo:\s*['"]([^'"]+)['"]/)?.[1];
  if (!user || !repo) throw new Error("could not parse gitConfig from lib/shared.ts");
  return { user, repo };
}

function keep(reason) {
  console.warn(`[stargazer-countries] ${reason}`);
  if (existsSync(OUTPUT)) console.warn(`[stargazer-countries] keeping existing ${OUTPUT}.`);
  else {
    mkdirSync(join(OUTPUT, ".."), { recursive: true });
    writeFileSync(OUTPUT, "[]\n");
  }
}

async function page(token, vars, tries = 4) {
  for (let attempt = 1; ; attempt++) {
    const res = await fetch("https://api.github.com/graphql", {
      method: "POST",
      headers: {
        "User-Agent": "deepeval-docs-stargazer-countries",
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: QUERY, variables: vars }),
    });
    const body = res.ok ? await res.json() : null;
    const data = body?.data?.repository?.stargazers;
    if (data) return data;
    if (attempt >= tries) {
      throw new Error(`GraphQL ${res.status} ${body?.errors?.[0]?.message ?? res.statusText}`);
    }
    await new Promise((r) => setTimeout(r, 1500 * attempt));
  }
}

async function main() {
  const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
  if (!token) return keep("no GITHUB_TOKEN; skipping.");
  const { user, repo } = readGitConfig();

  const counts = new Map();
  let after = null;
  let seen = 0;
  let total = 0;
  try {
    for (;;) {
      const data = await page(token, { owner: user, name: repo, after });
      total = data.totalCount;
      for (const n of data.nodes) {
        const code = resolve(n?.location);
        if (code) counts.set(code, (counts.get(code) ?? 0) + 1);
      }
      seen += data.nodes.length;
      if (seen % 2000 < 100) console.log(`[stargazer-countries] ${seen} / ${total}`);
      if (!data.pageInfo.hasNextPage) break;
      after = data.pageInfo.endCursor;
    }
  } catch (e) {
    return keep(e.message);
  }

  const out = [...counts.entries()]
    .map(([code, count]) => ({ code, count }))
    .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code));
  mkdirSync(join(OUTPUT, ".."), { recursive: true });
  writeFileSync(OUTPUT, JSON.stringify(out, null, 2) + "\n");
  const located = out.reduce((s, e) => s + e.count, 0);
  console.log(`[stargazer-countries] ${located} of ${seen} stargazers in ${out.length} countries -> ${OUTPUT}.`);
}

main().catch((e) => {
  console.error(e);
  process.exit(0);
});
