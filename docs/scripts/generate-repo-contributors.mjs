#!/usr/bin/env node
/**
 * Build-time generator: fetches the full contributor list for the
 * deepeval GitHub repo and writes `lib/generated/repo-contributors.json`.
 *
 * Used by the homepage "Built by amazing humans." section. Distinct from
 * `generate-contributors.mjs`, which produces a per-doc-page manifest
 * keyed by file path; this one is repo-wide.
 *
 *   [
 *     { "login": "penguine-ip", "avatarUrl": "…", "url": "…",
 *       "contributions": 4165 },
 *     …
 *   ]
 *
 * Bots (`…[bot]` login or non-`User` accounts) are excluded. The list
 * is sorted by contribution count desc, then login asc.
 *
 * Failure modes (all exit 0 so a flaky network never blocks `next build`):
 *   - Rate-limited / non-200: keeps the previous JSON file if one exists,
 *     otherwise writes `[]`. The component renders nothing when empty.
 *
 * Env:
 *   GITHUB_TOKEN  Optional; raises GitHub API rate limit from 60/hr to
 *                 5000/hr. Wire the built-in `GITHUB_TOKEN` in CI.
 *
 * Run: `npm run repo-contributors` (also runs pre-build).
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const OUTPUT = 'lib/generated/repo-contributors.json';
const PER_PAGE = 100;
const MAX_PAGES = 10; // 1000 contributors is plenty of headroom

function readGitConfig() {
  const src = readFileSync('lib/shared.ts', 'utf8');
  const user = src.match(/user:\s*['"]([^'"]+)['"]/)?.[1];
  const repo = src.match(/repo:\s*['"]([^'"]+)['"]/)?.[1];
  if (!user || !repo) throw new Error('could not parse gitConfig from lib/shared.ts');
  return { user, repo };
}

function saveJson(path, obj) {
  mkdirSync(join(path, '..'), { recursive: true });
  writeFileSync(path, JSON.stringify(obj, null, 2) + '\n');
}

function isBot(c) {
  return c?.type !== 'User' || (c?.login ?? '').endsWith('[bot]');
}

async function fetchPage(user, repo, page, token) {
  const headers = {
    'User-Agent': 'deepeval-docs-repo-contributors',
    Accept: 'application/vnd.github+json',
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  const url = `https://api.github.com/repos/${user}/${repo}/contributors?per_page=${PER_PAGE}&page=${page}`;
  const res = await fetch(url, { headers });
  if (!res.ok) {
    throw new Error(`GitHub API ${res.status} ${res.statusText} on page ${page}`);
  }
  return res.json();
}

async function main() {
  const { user, repo } = readGitConfig();
  const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;

  const all = [];
  try {
    for (let page = 1; page <= MAX_PAGES; page++) {
      const batch = await fetchPage(user, repo, page, token);
      if (!Array.isArray(batch) || batch.length === 0) break;
      all.push(...batch);
      if (batch.length < PER_PAGE) break;
    }
  } catch (e) {
    console.warn(`[repo-contributors] ${e.message}`);
    if (existsSync(OUTPUT)) {
      console.warn(`[repo-contributors] keeping existing ${OUTPUT}.`);
      return;
    }
    saveJson(OUTPUT, []);
    return;
  }

  const cleaned = all
    .filter((c) => c?.login && !isBot(c))
    .map((c) => ({
      login: c.login,
      avatarUrl: c.avatar_url,
      url: c.html_url,
      contributions: c.contributions ?? 0,
    }))
    .sort((a, b) => b.contributions - a.contributions || a.login.localeCompare(b.login));

  saveJson(OUTPUT, cleaned);
  console.log(`[repo-contributors] wrote ${cleaned.length} contributors -> ${OUTPUT}.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
