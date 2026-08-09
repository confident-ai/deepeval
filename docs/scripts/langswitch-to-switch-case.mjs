#!/usr/bin/env node
/**
 * One-off sweep: rewrite positional `<LangSwitch>` blocks into explicit
 * `<Switch><Case id="…">` ones.
 *
 * `<LangSwitch>` picked its child by index, so the two blocks had to appear in
 * LANGUAGE_IDS order and each side had to be a single element. This assigns the
 * same ids the runtime was already inferring, and reports any block whose fence
 * tag disagrees with its position — those were rendering under the wrong
 * language and need a human.
 *
 *   yarn node scripts/langswitch-to-switch-case.mjs
 */
import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

// Mirrors LANGUAGE_IDS in lib/lang/languages.ts, which is the order
// <LangSwitch> indexed its children by.
const LANGUAGE_IDS = ['python', 'typescript'];

const BLOCK = /^<LangSwitch>[ \t]*\n([\s\S]*?)^<\/LangSwitch>[ \t]*$/gm;
const FENCE = /^```(\w*)[^\n]*\n[\s\S]*?^```[ \t]*$/gm;

let changed = 0;
const problems = [];

function convert(inner, path) {
  const fences = [...inner.matchAll(FENCE)];

  if (fences.length !== LANGUAGE_IDS.length) {
    problems.push(`${path}: ${fences.length} fenced block(s), expected ${LANGUAGE_IDS.length}`);
    return null;
  }

  return fences
    .map((fence, index) => {
      const id = LANGUAGE_IDS[index];
      const tag = fence[1];
      if (LANGUAGE_IDS.includes(tag) && tag !== id) {
        problems.push(`${path}: fence #${index + 1} is \`${tag}\` but its position renders as ${id}`);
      }
      return `<Case id="${id}">\n\n${fence[0].trim()}\n\n</Case>`;
    })
    .join('\n\n');
}

function processFile(path) {
  const src = readFileSync(path, 'utf8');
  const out = src.replace(BLOCK, (block, inner) => {
    const body = convert(inner, path);
    return body ? `<Switch>\n\n${body}\n\n</Switch>` : block;
  });

  if (out !== src) {
    writeFileSync(path, out);
    changed += 1;
  }
}

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full);
    else if (full.endsWith('.mdx')) processFile(full);
  }
}

walk('content');
walk('snippets');

console.log(`${changed} file(s) updated.`);
if (problems.length > 0) {
  console.log(`\n${problems.length} block(s) need review:`);
  for (const problem of problems) console.log(`  - ${problem}`);
  process.exitCode = 1;
}
