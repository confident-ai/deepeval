#!/usr/bin/env node
/**
 * Build-time checks on the language components across `content/`:
 *
 *   1. every `<Term py="…" ts="…"/>` carries both spellings — MDX props are not
 *      typechecked;
 *   2. a Python spelling maps to exactly one TypeScript spelling site-wide,
 *      which is the one guarantee lost when the term registry went away;
 *   3. no `<Term>` sits inside a `<Case>` or `<Only>`, where the language is
 *      already decided;
 *   4. every `<Switch>` has a `<Case>` for every language — one-sided content
 *      belongs in `<Only>`;
 *   5. no run of sibling `<Only>` blocks covers every language between them —
 *      that is a `<Switch>` written the long way;
 *   6. every `<NotImplemented>` names a language and a feature.
 *
 * The summary line reports the `<NotImplemented>` count, which is the parity
 * backlog: one per section an SDK is still missing.
 *
 * None of this can tell whether a spelling is *correct* — that still means
 * reading `typescript/src`.
 *
 * Tags inside fenced code blocks are ignored so a page can document the tags
 * themselves without registering a mapping.
 *
 * Run: `npm run validate-terms` (also runs pre-build).
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const CONTENT = "content";
const LANGUAGES_MODULE = "lib/lang/languages.ts";
const ATTRIBUTE = /([a-zA-Z]+)="([^"]*)"/g;

/** The language components, matched in source order. */
const TAG =
  /<(Term|Switch|Case|Only|NotImplemented)\b([^>]*?)(\/?)>|<\/(Switch|Case|Only|NotImplemented)>/gs;

/** The `LANGUAGES` keys, so adding a language stays a one-file edit. */
function languageIds() {
  const source = readFileSync(LANGUAGES_MODULE, "utf8");
  const body = source.match(/export const LANGUAGES = \{(.*?)\n\} as const/s);
  const ids = [
    ...(body?.[1] ?? "").matchAll(/^ {2}([a-z][a-zA-Z0-9]*):/gm),
  ].map(([, id]) => id);

  if (ids.length === 0) {
    throw new Error(`Found no languages in ${LANGUAGES_MODULE}.`);
  }
  return ids;
}

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

/**
 * Runs of `<Only>` siblings with nothing but whitespace between them. A run
 * covering every language is a `<Switch>`, so the reader's choice picks one
 * branch instead of the page happening to render exactly one of several
 * independent blocks.
 */
function onlyRuns(source, onlys) {
  const runs = [];
  let run = [];

  for (const block of [...onlys].sort((a, b) => a.start - b.start)) {
    const previous = run.at(-1);
    const sibling =
      previous !== undefined &&
      previous.depth === block.depth &&
      source.slice(previous.end, block.start).trim() === "";

    if (!sibling) {
      if (run.length > 1) runs.push(run);
      run = [];
    }
    run.push(block);
  }
  if (run.length > 1) runs.push(run);

  return runs.filter((blocks) => {
    const ids = new Set(blocks.map(({ id }) => id));
    return LANGUAGE_IDS.every((id) => ids.has(id));
  });
}

/** The parity backlog, as "N sections missing from TypeScript" per language. */
function gapSummary(gaps) {
  if (gaps.length === 0) return "no <NotImplemented> sections — SDKs at parity.";

  const counts = LANGUAGE_IDS.map((id) => ({
    id,
    count: gaps.filter((gap) => gap.id === id).length,
  })).filter(({ count }) => count > 0);

  const parts = counts.map(({ id, count }) => `${count} missing from ${id}`);
  return `${gaps.length} <NotImplemented> section${
    gaps.length === 1 ? "" : "s"
  } (${parts.join(", ")}).`;
}

function scan(file) {
  const source = stripFences(readFileSync(file, "utf8"));
  const where = (index) =>
    `${file}:${source.slice(0, index).split("\n").length}`;

  const terms = [];
  const switches = [];
  const onlys = [];
  const gaps = [];
  const open = [];

  for (const match of source.matchAll(TAG)) {
    const [tag, name, attributes = "", selfClosing, closing] = match;

    if (closing) {
      const frame = open.pop();
      if (frame?.name === "Only") {
        onlys.push({
          id: frame.id,
          start: frame.start,
          end: match.index + tag.length,
          depth: open.length,
          where: where(frame.start),
        });
      }
      continue;
    }

    const attrs = Object.fromEntries(
      [...attributes.matchAll(ATTRIBUTE)].map(([, key, value]) => [key, value]),
    );

    if (name === "Term") {
      // Only a Case or an Only fixes the language; a shared Term in a Switch is fine.
      const decided = open.findLast((frame) => frame.name !== "Switch");
      terms.push({
        py: attrs.py,
        ts: attrs.ts,
        tag,
        wrapper: decided && `<${decided.name} id="${decided.id}">`,
        where: where(match.index),
      });
      continue;
    }

    if (name === "NotImplemented") {
      gaps.push({
        id: attrs.id,
        feature: attrs.feature,
        tag,
        where: where(match.index),
      });
    }

    if (name === "Case") {
      open.findLast((frame) => frame.name === "Switch")?.cases.push(attrs.id);
    }

    if (name === "Switch") {
      const entry = {
        name,
        id: attrs.id,
        cases: [],
        start: match.index,
        where: where(match.index),
      };
      switches.push(entry);
      open.push(entry);
      continue;
    }

    if (!selfClosing) open.push({ name, id: attrs.id, start: match.index });
  }

  return { terms, switches, gaps, runs: onlyRuns(source, onlys) };
}

const LANGUAGE_IDS = languageIds();
const scanned = mdxFiles(CONTENT).map(scan);
const all = scanned.flatMap(({ terms }) => terms);

const malformed = all.filter(
  ({ py, ts }) => py === undefined || ts === undefined,
);
const wrapped = all.filter(({ wrapper }) => wrapper !== undefined);

const runs = scanned.flatMap(({ runs }) => runs);

const gaps = scanned.flatMap(({ gaps }) => gaps);
const unnamed = gaps.filter(
  ({ id, feature }) =>
    !LANGUAGE_IDS.includes(id) || feature === undefined || feature === "",
);

const partial = scanned
  .flatMap(({ switches }) => switches)
  .map((entry) => ({
    ...entry,
    missing: LANGUAGE_IDS.filter((id) => !entry.cases.includes(id)),
  }))
  .filter(({ missing }) => missing.length > 0);

const byPython = new Map();
for (const term of all) {
  if (term.py === undefined || term.ts === undefined) continue;
  if (!byPython.has(term.py)) byPython.set(term.py, new Map());
  const spellings = byPython.get(term.py);
  if (!spellings.has(term.ts)) spellings.set(term.ts, []);
  spellings.get(term.ts).push(term.where);
}

const conflicts = [...byPython].filter(([, spellings]) => spellings.size > 1);

if (
  malformed.length === 0 &&
  wrapped.length === 0 &&
  partial.length === 0 &&
  runs.length === 0 &&
  unnamed.length === 0 &&
  conflicts.length === 0
) {
  console.log(
    `validate-terms: ${all.length} <Term> tags, ${byPython.size} distinct spellings, no conflicts.`,
  );
  console.log(`validate-terms: ${gapSummary(gaps)}`);
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
  report.push("language as a plain code span.");
  report.push("");
}

if (partial.length > 0) {
  report.push(`<Switch> blocks missing a language (${partial.length}):`);
  for (const { where, cases, missing } of partial) {
    report.push(`  ${where}  no case for ${missing.join(", ")}`);
    if (cases.length > 0) report.push(`    has ${cases.join(", ")}`);
  }
  report.push("");
  report.push(
    "A <Switch> is for content each SDK has its own version of, so every",
  );
  report.push(
    "language needs a case. Content only one SDK has goes in <Only>, which",
  );
  report.push(
    "renders nothing for the others instead of showing them a block written",
  );
  report.push("for a language they are not reading.");
  report.push("");
}

if (runs.length > 0) {
  report.push(
    `Adjacent <Only> blocks covering every language (${runs.length}):`,
  );
  for (const blocks of runs) {
    report.push(`  ${blocks[0].where}  ${blocks.map(({ id }) => id).join(", ")}`);
    for (const { where, id } of blocks.slice(1)) {
      report.push(`    ${where}  id="${id}"`);
    }
  }
  report.push("");
  report.push(
    "Every language has its own version of this content, so it is a <Switch>",
  );
  report.push(
    "with a <Case> each. Side-by-side <Only> blocks read as unrelated content",
  );
  report.push(
    "that happens to sum to full coverage, and drift apart when one is edited.",
  );
  report.push("");
}

if (unnamed.length > 0) {
  report.push(`<NotImplemented> tags missing a prop (${unnamed.length}):`);
  for (const { tag, where } of unnamed) report.push(`  ${where}  ${tag}`);
  report.push("");
  report.push(
    `\`id\` must be one of ${LANGUAGE_IDS.join(", ")} and \`feature\` must name`,
  );
  report.push(
    "the missing API, so the count of these tags reads as a parity backlog.",
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
