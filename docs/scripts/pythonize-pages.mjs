#!/usr/bin/env node
/**
 * Strip language-switching tags from the given pages, keeping the Python side,
 * and declare `languages: [python]` in frontmatter.
 *
 *   node scripts/pythonize-pages.mjs content/docs/getting-started.mdx ...
 *
 * Paths are relative to `docs/`. A page with no tags is still a valid target:
 * the frontmatter declaration alone takes it out of a TypeScript reader's
 * sidebar, which is what an index page whose children all became Python-only
 * needs.
 *
 * Deliberately line-based rather than an MDX AST rewrite: the blank lines around
 * a `<Case>` body are load-bearing (MDX folds a code fence into the preceding
 * paragraph without them) and a remark round-trip does not preserve them.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";

const DOCS = new URL("..", import.meta.url).pathname;
const KEEP = "python";
const DROP = "typescript";

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error("usage: pythonize-pages.mjs <path.mdx>... (relative to docs/)");
  process.exit(1);
}

const isTag = (line, re) => re.test(line.trim());
const caseId = (line) => line.trim().match(/^<Case\s+id="([^"]+)"\s*>$/)?.[1];
const onlyId = (line) => line.trim().match(/^<Only\s+id="([^"]+)"\s*>$/)?.[1];

/** Drop leading and trailing blank lines so the caller controls the spacing. */
function trimBlank(lines) {
  let a = 0;
  let b = lines.length;
  while (a < b && lines[a].trim() === "") a++;
  while (b > a && lines[b - 1].trim() === "") b--;
  return lines.slice(a, b);
}

/**
 * Replace each block-level `<Switch>` with its `<Case id="python">` body and
 * each block-level `<Only>` with its contents (or nothing, for TypeScript).
 * Recurses so a wrapper nested inside a kept body is handled too.
 */
function unwrapBlocks(lines) {
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (isTag(line, /^<Switch>$/)) {
      let depth = 1;
      let j = i + 1;
      while (j < lines.length && depth > 0) {
        if (isTag(lines[j], /^<Switch>$/)) depth++;
        else if (isTag(lines[j], /^<\/Switch>$/)) depth--;
        if (depth === 0) break;
        j++;
      }
      if (depth !== 0) throw new Error("unterminated <Switch>");
      out.push(...unwrapBlocks(pickCase(lines.slice(i + 1, j))));
      i = j;
      continue;
    }

    const only = onlyId(line);
    if (only) {
      let depth = 1;
      let j = i + 1;
      while (j < lines.length && depth > 0) {
        if (onlyId(lines[j])) depth++;
        else if (isTag(lines[j], /^<\/Only>$/)) depth--;
        if (depth === 0) break;
        j++;
      }
      if (depth !== 0) throw new Error("unterminated <Only>");
      if (only === KEEP) out.push(...unwrapBlocks(trimBlank(lines.slice(i + 1, j))));
      i = j;
      continue;
    }

    out.push(line);
  }
  return out;
}

/**
 * Body of a `<Switch>`: return the Python case's contents plus any content that
 * sat outside every `<Case>` (that part is shared across languages).
 */
function pickCase(body) {
  const out = [];
  let seenKeep = false;
  for (let i = 0; i < body.length; i++) {
    const id = caseId(body[i]);
    if (!id) {
      out.push(body[i]);
      continue;
    }
    let depth = 1;
    let j = i + 1;
    while (j < body.length && depth > 0) {
      if (caseId(body[j])) depth++;
      else if (isTag(body[j], /^<\/Case>$/)) depth--;
      if (depth === 0) break;
      j++;
    }
    if (depth !== 0) throw new Error("unterminated <Case>");
    if (id === KEEP) {
      if (out.length && out.at(-1).trim() !== "") out.push("");
      out.push(...trimBlank(body.slice(i + 1, j)));
      seenKeep = true;
    } else if (id !== DROP) {
      throw new Error(`unexpected <Case id="${id}">`);
    }
    i = j;
  }
  if (!seenKeep) throw new Error(`<Switch> with no <Case id="${KEEP}">`);
  return trimBlank(out);
}

/**
 * Inline forms, which sit mid-sentence or inside a JSX container. Applied per
 * line, after `unwrapBlocks` has taken care of every block-level wrapper.
 */
function unwrapInline(line) {
  return line
    .replace(/<Term\s+([^>]*?)\/>/g, (m, attrs) => {
      const py = attrs.match(/py="([^"]*)"/)?.[1];
      if (py === undefined) throw new Error(`<Term> without py: ${m}`);
      return `\`${py}\``;
    })
    .replace(new RegExp(`<Only\\s+id="${KEEP}"\\s*>(.*?)</Only>`, "g"), "$1")
    .replace(new RegExp(`<Only\\s+id="${DROP}"\\s*>.*?</Only>`, "g"), "")
    .replace(
      /<Switch>\s*(?:<Case\s+id="[^"]+"\s*>.*?<\/Case>\s*)+<\/Switch>/g,
      (block) => {
        const m = block.match(
          new RegExp(`<Case\\s+id="${KEEP}"\\s*>(.*?)</Case>`),
        );
        if (!m) throw new Error(`inline <Switch> without ${KEEP} case`);
        return m[1].trim();
      },
    );
}

/**
 * Set `languages: [python]` in frontmatter, replacing any existing declaration.
 *
 * Replacing rather than skipping is the whole point on a docs or integrations
 * page, where the field is required and so already reads `[python, typescript]`.
 * Leaving it would strip the TypeScript content while the page went on claiming
 * to serve TypeScript readers, which is the silent fallthrough LANGUAGES.md
 * treats as a bug.
 */
function declareLanguages(lines) {
  if (lines[0].trim() !== "---") throw new Error("no frontmatter");
  const end = lines.indexOf("---", 1);
  if (end === -1) throw new Error("unterminated frontmatter");

  const declaration = `languages: [${KEEP}]`;
  const offset = lines.slice(1, end).findIndex((l) => /^languages:/.test(l));
  if (offset === -1) {
    return [...lines.slice(0, end), declaration, ...lines.slice(end)];
  }
  const at = offset + 1;
  return [...lines.slice(0, at), declaration, ...lines.slice(at + 1)];
}

let changed = 0;
for (const rel of files) {
  const path = isAbsolute(rel) ? rel : join(DOCS, rel);
  const before = readFileSync(path, "utf8");
  let lines = unwrapBlocks(before.split("\n")).map(unwrapInline);
  lines = declareLanguages(lines);
  const after = lines.join("\n");
  if (after !== before) {
    writeFileSync(path, after);
    changed++;
  }
  console.log(`${after === before ? "unchanged" : "rewrote  "} ${rel}`);
}
console.log(`\n${changed}/${files.length} files rewritten`);
