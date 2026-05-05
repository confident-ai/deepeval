#!/usr/bin/env node
/**
 * Migrate custom <Timeline> / <TimelineItem title="..."> to Fumadocs'
 * native <Steps> / <Step>. The step title becomes an h3 inside the
 * <Step> so Fumadocs' `.fd-step` counter styling applies naturally:
 *
 *   <Timeline>
 *   <TimelineItem title="Create the metric">
 *   body
 *   </TimelineItem>
 *   </Timeline>
 *
 *     ↓
 *
 *   <Steps>
 *   <Step>
 *   ### Create the metric
 *
 *   body
 *   </Step>
 *   </Steps>
 *
 *   yarn node scripts/timeline-to-steps.mjs
 */
import { readdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";

function transform(src) {
  let out = src;

  // <TimelineItem title="...">  →  <Step>\n### ...\n
  out = out.replace(
    /<TimelineItem\s+title\s*=\s*"([^"]*)"\s*>/g,
    (_m, title) => `<Step>\n### ${title}\n`,
  );
  // <TimelineItem title='...'>  →  <Step>\n### ...\n
  out = out.replace(
    /<TimelineItem\s+title\s*=\s*'([^']*)'\s*>/g,
    (_m, title) => `<Step>\n### ${title}\n`,
  );
  // <TimelineItem title={...}>  →  <Step>\n### {...}\n  (rare)
  out = out.replace(
    /<TimelineItem\s+title\s*=\s*\{([^}]*)\}\s*>/g,
    (_m, expr) => `<Step>\n### {${expr}}\n`,
  );
  // Bare <TimelineItem> (no title) → <Step>
  out = out.replace(/<TimelineItem\s*>/g, "<Step>");

  out = out.replace(/<\/TimelineItem>/g, "</Step>");
  out = out.replace(/<Timeline>/g, "<Steps>");
  out = out.replace(/<\/Timeline>/g, "</Steps>");

  return out;
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
