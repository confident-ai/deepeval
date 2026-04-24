#!/usr/bin/env node
/**
 * Normalize Docusaurus-era admonition titles to remark-directive's
 * standard bracket syntax so `remark-directive` + `remarkAdmonitions`
 * can parse them:
 *
 *   :::note Login persistence   -->   :::note[Login persistence]
 *   :::tip DID YOU KNOW?        -->   :::tip[DID YOU KNOW?]
 *
 * Leaves the bare form untouched:
 *
 *   :::note
 *
 * Skips fenced code blocks so examples inside docs aren't rewritten.
 */
import { readdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";

const TYPES =
  "note|info|tip|success|important|warning|caution|danger|error|secondary";

// Match `:::<type> <label...>` at line start where label is not already
// wrapped in [ or {. Handle trailing whitespace / newline.
const PATTERN = new RegExp(
  String.raw`^(:::(?:${TYPES}))[ \t]+(?![\[\{])([^\n]+?)[ \t]*$`,
  "gm",
);

const FENCE = /```[\s\S]*?```/g;

function transform(src) {
  const chunks = [];
  let last = 0;
  for (const m of src.matchAll(FENCE)) {
    chunks.push({ text: src.slice(last, m.index), code: false });
    chunks.push({ text: m[0], code: true });
    last = m.index + m[0].length;
  }
  chunks.push({ text: src.slice(last), code: false });

  return chunks
    .map((c) =>
      c.code ? c.text : c.text.replace(PATTERN, (_m, head, label) => `${head}[${label}]`),
    )
    .join("");
}

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const s = statSync(full);
    if (s.isDirectory()) walk(full);
    else if (full.endsWith(".mdx")) processFile(full);
  }
}

let changed = 0;
function processFile(path) {
  const src = readFileSync(path, "utf8");
  const out = transform(src);
  if (out !== src) {
    writeFileSync(path, out);
    changed += 1;
    console.log("·", path);
  }
}

walk("content");
console.log(`\n${changed} file(s) updated.`);
