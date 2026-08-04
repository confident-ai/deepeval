import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const audit = JSON.parse(readFileSync(join(root, "languages-audit.json"), "utf8"));

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (entry.endsWith(".mdx")) out.push(full);
  }
  return out;
}

const count = (text, re) => (text.match(re) ?? []).length;

const actual = new Map();
for (const file of walk(join(root, "content"))) {
  const text = readFileSync(file, "utf8");
  const stats = {
    switch: count(text, /<Switch[\s>]/g),
    only: count(text, /<Only[\s>]/g),
    term: count(text, /<Term[\s>/]/g),
    pythonFences: count(text, /^```python/gm),
    tsFences: count(text, /^```(typescript|ts)\b/gm),
  };
  const fm = text.match(/^---\n([\s\S]*?)\n---/);
  const langLine = fm?.[1].match(/^languages:.*$/m)?.[0] ?? "";
  stats.languages = ["python", "typescript"].filter((l) => langLine.includes(l));
  actual.set(relative(root, file), stats);
}

const fields = ["switch", "only", "term", "pythonFences", "tsFences"];
let mismatched = 0;
const missing = [];

for (const page of audit.pages) {
  const now = actual.get(page.file);
  if (!now) {
    missing.push(page.file);
    continue;
  }
  const diffs = fields
    .filter((f) => now[f] !== page[f])
    .map((f) => `${f}: audit ${page[f]} -> now ${now[f]}`);
  if (page.languages.join(",") !== now.languages.join(","))
    diffs.push(`languages: audit [${page.languages}] -> now [${now.languages}]`);
  if (diffs.length) {
    mismatched++;
    console.log(`${page.file}\n  ${diffs.join("\n  ")}`);
  }
}

const auditFiles = new Set(audit.pages.map((p) => p.file));
const untracked = [...actual]
  .filter(([f, s]) => !auditFiles.has(f) && (s.switch || s.only || s.term))
  .map(([f]) => f);

console.log(`\naudit pages: ${audit.pages.length}`);
console.log(`mismatched:  ${mismatched}`);
console.log(`missing files: ${missing.length}${missing.length ? "\n  " + missing.join("\n  ") : ""}`);
console.log(
  `tagged pages absent from audit: ${untracked.length}${untracked.length ? "\n  " + untracked.join("\n  ") : ""}`,
);
