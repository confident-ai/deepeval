#!/usr/bin/env node
/**
 * One-off sweep: strip `import ... from '@site/src/components/<X>'`
 * lines for components that are now globally registered in
 * components/mdx.tsx, AND ensure the import block is still separated
 * from MDX content by a blank line (MDX requires it).
 *
 *   yarn node scripts/strip-redundant-mdx-imports.mjs
 */
import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const REGISTERED = [
  'VideoDisplayer',
  'Equation',
  'MetricTagsDisplayer',
  'FeatureComparisonTable',
  'LinkCards',
  'TechStackCards',
  'CopyMarkdownButton',
  'FAQ',
  'Timeline',
];

// Strip a whole line (including its trailing newline) matching:
//   import ... from '@site/src/components/<REGISTERED>'
// Uses [ \t]* (not \s*) at the edges so greedy whitespace doesn't eat
// blank lines that separate the import block from MDX content.
const stripPattern = new RegExp(
  String.raw`^[ \t]*import[ \t]+[^;]+?from[ \t]+['"]@site/src/components/(?:${REGISTERED.join('|')})['"][ \t]*;?[ \t]*\r?\n`,
  'gm',
);

// Ensure an import line is followed by either another import/export or
// a blank line — insert a blank line otherwise. MDX requires this
// separation; our previous (too-greedy) regex occasionally ate it.
const ensureBlankPattern = /^((?:import|export)[^\n]*\n)(?!(?:import|export)\b|[ \t]*\r?\n)(?=\S)/gm;

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const s = statSync(full);
    if (s.isDirectory()) walk(full);
    else if (full.endsWith('.mdx')) processFile(full);
  }
}

let changed = 0;
function processFile(path) {
  const src = readFileSync(path, 'utf8');
  let out = src.replace(stripPattern, '');
  out = out.replace(ensureBlankPattern, '$1\n');
  if (out !== src) {
    writeFileSync(path, out);
    changed += 1;
    console.log('·', path);
  }
}

walk('content');
console.log(`\n${changed} file(s) updated.`);
