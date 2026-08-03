#!/usr/bin/env node
/**
 * Build-time checks on `<Term py="…" ts="…"/>` across `content/`:
 *
 *   1. both spellings are present — MDX props are not typechecked;
 *   2. a Python spelling maps to exactly one TypeScript spelling site-wide,
 *      which is the one guarantee lost when the term registry went away;
 *   3. no `<Term>` sits inside a `<Case>` or `<Only>`, where the language is
 *      already decided.
 *
 * None of this can tell whether a spelling is *correct* — that still means
 * reading `typescript/src`.
 *
 * Tags inside fenced code blocks are ignored so a page can document the tag
 * itself without registering a mapping.
 *
 * Run: `npm run validate-terms` (also runs pre-build).
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const CONTENT = "content";
const ATTRIBUTE = /([a-zA-Z]+)="([^"]*)"/g;

/** `<Term/>` plus the wrappers it may not appear in, matched in source order. */
const TAG = /<(Term|Case|Only)\b([^>]*?)(\/?)>|<\/(Case|Only)>/gs;

function mdxFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return mdxFiles(path);
    return entry.name.endsWith(".mdx") ? [path] : [];
  });
}

/** Blank out fenced blocks, preserving offsets so line numbers stay true. */
function stripFences(source) {
  let fenced = false;
  return source
    .split("\n")
    .map((line) => {
      if (line.startsWith("```")) {
        fenced = !fenced;
        return "";
      }
      return fenced ? "" : line;
    })
    .join("\n");
}

function scan(file) {
  const source = stripFences(readFileSync(file, "utf8"));
  const terms = [];
  const open = [];

  for (const match of source.matchAll(TAG)) {
    const [tag, name, attributes = "", selfClosing, closing] = match;

    if (closing) {
      open.pop();
      continue;
    }

    const attrs = Object.fromEntries(
      [...attributes.matchAll(ATTRIBUTE)].map(([, key, value]) => [key, value]),
    );

    if (name !== "Term") {
      if (!selfClosing) open.push(`<${name} id="${attrs.id}">`);
      continue;
    }

    terms.push({
      py: attrs.py,
      ts: attrs.ts,
      tag,
      wrapper: open.at(-1),
      where: `${file}:${source.slice(0, match.index).split("\n").length}`,
    });
  }

  return terms;
}

const all = mdxFiles(CONTENT).flatMap(scan);

const malformed = all.filter(
  ({ py, ts }) => py === undefined || ts === undefined,
);
const wrapped = all.filter(({ wrapper }) => wrapper !== undefined);

const byPython = new Map();
for (const term of all) {
  if (term.py === undefined || term.ts === undefined) continue;
  if (!byPython.has(term.py)) byPython.set(term.py, new Map());
  const spellings = byPython.get(term.py);
  if (!spellings.has(term.ts)) spellings.set(term.ts, []);
  spellings.get(term.ts).push(term.where);
}

const conflicts = [...byPython].filter(([, spellings]) => spellings.size > 1);

if (malformed.length === 0 && wrapped.length === 0 && conflicts.length === 0) {
  console.log(
    `validate-terms: ${all.length} <Term> tags, ${byPython.size} distinct spellings, no conflicts.`,
  );
  process.exit(0);
}

const report = [];

if (malformed.length > 0) {
  report.push(`<Term> tags missing a spelling (${malformed.length}):`);
  for (const { tag, where } of malformed) report.push(`  ${where}  ${tag}`);
  report.push("");
  report.push(
    "Both `py` and `ts` are required; MDX props are not typechecked.",
  );
  report.push("");
}

if (wrapped.length > 0) {
  report.push(`<Term> tags inside a <Case> or <Only> (${wrapped.length}):`);
  for (const { tag, where, wrapper } of wrapped) {
    report.push(`  ${where}  in ${wrapper}`);
    report.push(`    ${tag}`);
  }
  report.push("");
  report.push(
    "The wrapper already fixes the language, so write the spelling for that",
  );
  report.push(
    "language as a plain code span. A one-case <Switch> makes this actively",
  );
  report.push(
    "wrong: it shows its Python case to TypeScript readers, and a <Term>",
  );
  report.push(
    "inside would render TypeScript text in a block labelled Python.",
  );
  report.push("");
}

if (conflicts.length > 0) {
  report.push(
    `Python spellings with more than one TypeScript spelling (${conflicts.length}):`,
  );
  for (const [py, spellings] of conflicts) {
    report.push(`  ${py}`);
    for (const [ts, wheres] of spellings) {
      report.push(
        `    ts="${ts}"  ${wheres.length} use${wheres.length === 1 ? "" : "s"}`,
      );
      for (const where of wheres) report.push(`      ${where}`);
    }
  }
  report.push("");
  report.push("Check the shipped SDK in `typescript/src` and make them agree.");
}

console.error(report.join("\n"));
process.exit(1);
