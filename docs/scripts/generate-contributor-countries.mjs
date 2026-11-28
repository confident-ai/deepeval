#!/usr/bin/env node
/**
 * Build-time generator: reads lib/generated/repo-contributors.json, looks
 * up each contributor's free-text GitHub profile location, resolves it to
 * a country, and writes `lib/generated/contributor-countries.json`:
 *
 *   [
 *     { "code": "IN", "count": 24, "logins": ["a", "b", "c"] },
 *     …
 *   ]
 *
 * sorted by count desc. `logins` are that country's top contributors, at
 * most 3. Locations that do not resolve (empty, "Earth", "Remote") are
 * skipped, so the counts are a floor, not a census.
 *
 * Used by the community page globe: countries without a meetup are ranked
 * by these counts, and hovering a flag shows them.
 *
 * Failure modes (all exit 0 so a flaky network never blocks `next build`):
 *   - No token, rate-limited or non-200: keeps the previous JSON file if
 *     one exists, otherwise writes `[]`. The globe then shows meetups only.
 *
 * Env:
 *   GITHUB_TOKEN  Needed in practice: one request per contributor, more
 *                 than the 60/hr unauthenticated limit.
 *
 * Run: `npm run contributor-countries` (also runs pre-build).
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { resolve } from "./community/location-to-country.mjs";

const INPUT = "lib/generated/repo-contributors.json";
const OUTPUT = "lib/generated/contributor-countries.json";
const CONCURRENCY = 8;

function saveJson(path, obj) {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, JSON.stringify(obj, null, 2) + "\n");
}

function keepOrEmpty(reason) {
  console.warn(`[contributor-countries] ${reason}`);
  if (existsSync(OUTPUT)) {
    console.warn(`[contributor-countries] keeping existing ${OUTPUT}.`);
    return;
  }
  saveJson(OUTPUT, []);
}

// --- Fetch ----------------------------------------------------------------

async function fetchLocation(login, token) {
  const res = await fetch(`https://api.github.com/users/${encodeURIComponent(login)}`, {
    headers: {
      "User-Agent": "deepeval-docs-contributor-countries",
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
    },
  });
  // A deleted or renamed account has no profile; count it as unknown.
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GitHub API ${res.status} ${res.statusText} for ${login}`);
  return (await res.json()).location ?? null;
}

async function main() {
  const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
  if (!token) return keepOrEmpty("no GITHUB_TOKEN; skipping.");
  if (!existsSync(INPUT)) return keepOrEmpty(`${INPUT} missing; run repo-contributors first.`);

  const contributors = JSON.parse(readFileSync(INPUT, "utf8")).filter(
    (c) => !c.login.endsWith("[bot]")
  );
  const located = new Array(contributors.length);
  try {
    let next = 0;
    await Promise.all(
      Array.from({ length: CONCURRENCY }, async () => {
        while (next < contributors.length) {
          const i = next++;
          located[i] = resolve(await fetchLocation(contributors[i].login, token));
        }
      })
    );
  } catch (e) {
    return keepOrEmpty(e.message);
  }

  const byCode = new Map();
  contributors.forEach((c, i) => {
    const code = located[i];
    if (!code) return;
    const entry = byCode.get(code) ?? { code, count: 0, logins: [] };
    entry.count++;
    if (entry.logins.length < 3) entry.logins.push(c.login); // input is sorted by contributions
    byCode.set(code, entry);
  });
  const out = [...byCode.values()].sort((a, b) => b.count - a.count || a.code.localeCompare(b.code));
  saveJson(OUTPUT, out);
  const total = out.reduce((s, e) => s + e.count, 0);
  console.log(
    `[contributor-countries] ${total} of ${contributors.length} contributors in ${out.length} countries -> ${OUTPUT}.`
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(0);
});
