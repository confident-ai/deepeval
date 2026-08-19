#!/usr/bin/env node
/**
 * Build-time generator: maps each docs URL to its frontmatter `languages`
 * so the client LanguageProvider can pick an SSR-safe default without
 * importing fumadocs server collections.
 *
 *   { "/integrations/frameworks/mastra": ["typescript"], ... }
 *
 * Walks the MDX trees that declare language support. Route-group folders
 * like `(rag)` are stripped from URLs the same way fumadocs does.
 *
 * Run: `npm run page-languages` (also runs pre-build).
 */
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  readdirSync,
  statSync,
  existsSync,
} from 'node:fs';
import { join, relative, sep } from 'node:path';

const OUTPUT = 'lib/generated/page-languages.json';

const SECTIONS = [
  ['content/docs', '/docs'],
  ['content/integrations', '/integrations'],
  ['content/guides', '/guides'],
  ['content/tutorials', '/tutorials'],
];

const ALLOWED = new Set(['python', 'typescript']);

function walk(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, out);
    else if (name.endsWith('.mdx') || name.endsWith('.md')) out.push(full);
  }
  return out;
}

function parseLanguages(src) {
  if (!src.startsWith('---')) return undefined;
  const end = src.indexOf('\n---', 3);
  if (end === -1) return undefined;
  const fm = src.slice(3, end);
  const match = fm.match(/^languages:\s*\[([^\]]*)\]/m);
  if (!match) return undefined;
  const langs = match[1]
    .split(',')
    .map((s) => s.trim().replace(/^['"]|['"]$/g, ''))
    .filter(Boolean);
  if (langs.length === 0) return undefined;
  for (const lang of langs) {
    if (!ALLOWED.has(lang)) {
      throw new Error(`unknown language "${lang}" in frontmatter`);
    }
  }
  return langs;
}

/** Mirror fumadocs URL shaping: drop route groups, map index → parent. */
function fileToUrl(sectionRoot, baseUrl, filePath) {
  let rel = relative(sectionRoot, filePath).split(sep).join('/');
  rel = rel.replace(/(^|\/)\([^/]+\)(?=\/)/g, '$1').replace(/^\//, '');
  rel = rel.replace(/\.mdx?$/, '');
  if (rel === 'index') return baseUrl;
  if (rel.endsWith('/index')) rel = rel.slice(0, -'/index'.length);
  return `${baseUrl}/${rel}`;
}

const map = {};

for (const [sectionRoot, baseUrl] of SECTIONS) {
  if (!existsSync(sectionRoot)) continue;
  for (const file of walk(sectionRoot)) {
    let langs;
    try {
      langs = parseLanguages(readFileSync(file, 'utf8'));
    } catch (err) {
      throw new Error(`${file}: ${err.message}`);
    }
    if (!langs) continue;
    const url = fileToUrl(sectionRoot, baseUrl, file);
    if (map[url] && map[url].join(',') !== langs.join(',')) {
      throw new Error(
        `URL collision for ${url}: ${map[url]} vs ${langs} (${file})`,
      );
    }
    map[url] = langs;
  }
}

const urls = Object.keys(map).sort();
const sorted = Object.fromEntries(urls.map((url) => [url, map[url]]));

mkdirSync(join(OUTPUT, '..'), { recursive: true });
writeFileSync(OUTPUT, `${JSON.stringify(sorted, null, 2)}\n`);
console.log(`[page-languages] ${urls.length} pages → ${OUTPUT}`);
