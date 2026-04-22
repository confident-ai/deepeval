#!/usr/bin/env node
/**
 * Build-time generator: walks `content/docs/**` and produces
 * `lib/generated/contributors.json`, keyed by each file's repo-relative
 * path. Values are sorted by commit count desc.
 *
 *   {
 *     "content/docs/getting-started.mdx": [
 *       { "login": "penguine-ip", "name": "Jeffrey Ip",
 *         "avatarUrl": "…", "url": "…", "commits": 12 },
 *       …
 *     ]
 *   }
 *
 * How the GitHub linking works: `git log` gives us (email, name) but no
 * GitHub handle. We resolve email→login via `GET /repos/{owner}/{repo}/
 * commits/{sha}` — GitHub does the email-to-user lookup server-side and
 * returns `author.login` / `author.avatar_url` / `author.html_url`. One
 * API call per unique email (not per commit), cached in
 * `lib/generated/.contributors-cache.json` so subsequent runs are ~free.
 *
 * Bots (`…[bot]` login suffix) are excluded. Commits whose email doesn't
 * resolve to a GitHub user are dropped — no point showing a ghost.
 *
 * Failure modes handled gracefully (all exit 0 so builds don't break):
 *   - Not in a git repo, or no commits touch content/docs: writes an
 *     empty `{}` and warns. The runtime component renders nothing for
 *     pages with no entry, so the UI silently no-ops.
 *   - GitHub API 403 / rate-limited: keeps existing cache entries,
 *     skips the uncached emails, warns.
 *
 * Env:
 *   GITHUB_TOKEN  Optional; bumps GitHub API rate limit from 60/hr to
 *                 5000/hr. In CI, wire the built-in `GITHUB_TOKEN`.
 *
 * Run: `npm run contributors`  (also runs pre-build).
 */
import { execSync } from 'node:child_process';
import { readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

// Sections that display a contributors list. Changelog and blog are
// intentionally omitted — blog posts have their own author byline, and
// changelog entries are attributed by release.
const CONTENT_DIRS = [
  'content/docs',
  'content/guides',
  'content/tutorials',
  'content/integrations',
];
const OUTPUT = 'lib/generated/contributors.json';
const CACHE = 'lib/generated/.contributors-cache.json';

// Cofounders — always credited on every page regardless of git history.
// Pinned to the top of each page's list in this order.
const PINNED_LOGINS = ['penguine-ip', 'kritinv'];

// Read repo coords from lib/shared.ts so there's one source of truth.
// Parsing literals avoids having to compile the TS file at script time.
function readGitConfig() {
  const src = readFileSync('lib/shared.ts', 'utf8');
  const user = src.match(/user:\s*['"]([^'"]+)['"]/)?.[1];
  const repo = src.match(/repo:\s*['"]([^'"]+)['"]/)?.[1];
  if (!user || !repo) throw new Error('could not parse gitConfig from lib/shared.ts');
  return { user, repo };
}

function tryExec(cmd) {
  try {
    return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return null;
  }
}

function inGitRepo() {
  return tryExec('git rev-parse --is-inside-work-tree') === 'true';
}

function walkMdx(dir, acc = []) {
  if (!existsSync(dir)) return acc;
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const s = statSync(full);
    if (s.isDirectory()) walkMdx(full, acc);
    else if (entry.endsWith('.mdx') || entry.endsWith('.md')) acc.push(full);
  }
  return acc;
}

// `git log --follow` so renames don't reset attribution. %x09 = tab, so
// we don't have to worry about author names containing our delimiter.
function gitCommitsForFile(file) {
  const out = tryExec(`git log --follow --format="%H%x09%ae%x09%an" -- "${file}"`);
  if (!out) return [];
  return out.split('\n').filter(Boolean).map((line) => {
    const [sha, email, name] = line.split('\t');
    return { sha, email: email.toLowerCase(), name };
  });
}

function loadCache() {
  if (!existsSync(CACHE)) return {};
  try { return JSON.parse(readFileSync(CACHE, 'utf8')); } catch { return {}; }
}

function saveJson(path, obj) {
  mkdirSync(join(path, '..'), { recursive: true });
  writeFileSync(path, JSON.stringify(obj, null, 2) + '\n');
}

async function resolveAuthor(sha, { user, repo, token }) {
  const headers = { 'User-Agent': 'deepeval-docs-contributors', Accept: 'application/vnd.github+json' };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`https://api.github.com/repos/${user}/${repo}/commits/${sha}`, { headers });
  if (res.status === 403 || res.status === 429) throw new Error(`rate_limited`);
  if (!res.ok) return null; // 404 (commit not on this remote yet) → treat as unresolvable
  const body = await res.json();
  const a = body?.author;
  if (!a?.login) return null; // commit exists but email isn't linked to a GH user
  return {
    login: a.login,
    // Prefer the commit-author display name; fall back to the GH user's
    // `name` field. The API's `user.name` endpoint would give us the
    // canonical one but costs another request — not worth it.
    name: body?.commit?.author?.name || a.login,
    avatarUrl: a.avatar_url,
    url: a.html_url,
  };
}

function isBot(author) {
  return author?.login?.endsWith('[bot]') || /\[bot\]$/.test(author?.name ?? '');
}

async function main() {
  if (!inGitRepo()) {
    console.warn('[contributors] not inside a git repo; writing empty manifest.');
    saveJson(OUTPUT, {});
    return;
  }

  const { user, repo } = readGitConfig();
  const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
  const cache = loadCache();

  const files = CONTENT_DIRS.flatMap((d) => walkMdx(d));
  if (files.length === 0) {
    console.warn(`[contributors] no MDX files found under ${CONTENT_DIRS.join(', ')}.`);
    saveJson(OUTPUT, {});
    return;
  }

  // First pass: gather per-file commit metadata (all local, no network).
  const perFile = new Map(); // relPath → Map<email, { name, commits, sha }>
  for (const file of files) {
    const rel = relative('.', file);
    const commits = gitCommitsForFile(file);
    if (commits.length === 0) continue;
    const byEmail = new Map();
    for (const c of commits) {
      const prev = byEmail.get(c.email) ?? { name: c.name, commits: 0, sha: c.sha };
      prev.commits += 1;
      byEmail.set(c.email, prev);
    }
    perFile.set(rel, byEmail);
  }

  // Second pass: resolve every unseen email to a GitHub user.
  const uniqueEmails = new Set();
  for (const byEmail of perFile.values()) for (const e of byEmail.keys()) uniqueEmails.add(e);

  let resolved = 0, skipped = 0, bot = 0, rateLimited = false;
  for (const email of uniqueEmails) {
    if (email in cache) continue;
    // Use any commit SHA associated with this email (they all resolve
    // to the same GH user for a given email).
    let sha;
    for (const byEmail of perFile.values()) {
      const entry = byEmail.get(email);
      if (entry) { sha = entry.sha; break; }
    }
    if (!sha) continue;
    try {
      const author = await resolveAuthor(sha, { user, repo, token });
      if (author && isBot(author)) { cache[email] = null; bot += 1; continue; }
      cache[email] = author;
      if (author) resolved += 1; else skipped += 1;
    } catch (e) {
      if (e.message === 'rate_limited') {
        rateLimited = true;
        console.warn('[contributors] GitHub API rate-limited; stopping resolution. Set GITHUB_TOKEN to raise the ceiling.');
        break;
      }
      console.warn(`[contributors] failed resolving ${email}: ${e.message}`);
      skipped += 1;
    }
  }

  saveJson(CACHE, cache);

  // Pinned logins → profile lookup from cache. We look up each pinned
  // login by scanning the cache for any email that resolved to it; that
  // way we don't have to burn extra API calls just to fetch cofounder
  // avatars. If a pinned login isn't in the cache yet (first run, or
  // they've never committed anywhere in this repo), we skip silently —
  // better than showing a half-rendered entry.
  const pinnedProfiles = PINNED_LOGINS.map((login) => {
    for (const entry of Object.values(cache)) {
      if (entry?.login === login) return entry;
    }
    console.warn(`[contributors] pinned login "${login}" not found in cache; skipping until they appear in git history.`);
    return null;
  }).filter(Boolean);

  // Third pass: materialize the manifest using the (now populated) cache.
  const manifest = {};
  for (const [rel, byEmail] of perFile) {
    const list = [];
    const seenLogins = new Set();
    for (const [email, meta] of byEmail) {
      const author = cache[email];
      if (!author) continue;
      // Same GH user may have pushed from multiple emails — collapse.
      if (seenLogins.has(author.login)) {
        const existing = list.find((x) => x.login === author.login);
        if (existing) existing.commits += meta.commits;
        continue;
      }
      seenLogins.add(author.login);
      list.push({ ...author, commits: meta.commits });
    }

    // Sort real committers by commit count, then alphabetical.
    list.sort((a, b) => b.commits - a.commits || a.login.localeCompare(b.login));

    // Prepend pinned cofounders. If a cofounder also shows up as a real
    // committer, keep their actual commit count but move them to the
    // pinned slot (preserves attribution honesty while guaranteeing
    // top-of-list placement).
    const pinnedList = [];
    for (const profile of pinnedProfiles) {
      const idx = list.findIndex((x) => x.login === profile.login);
      if (idx >= 0) {
        pinnedList.push(list[idx]);
        list.splice(idx, 1);
      } else {
        pinnedList.push({ ...profile, commits: 0 });
      }
    }

    const finalList = [...pinnedList, ...list];
    if (finalList.length > 0) manifest[rel] = finalList;
  }

  saveJson(OUTPUT, manifest);
  console.log(
    `[contributors] ${Object.keys(manifest).length} pages, ` +
    `resolved ${resolved} new author(s), skipped ${skipped}, bots filtered ${bot}` +
    (rateLimited ? ' (rate-limited; re-run with GITHUB_TOKEN)' : '') + '.'
  );
}

main().catch((e) => { console.error(e); process.exit(1); });
