#!/usr/bin/env node
// Fails when the Python and TypeScript environment variable surfaces drift
// apart. Accepted differences live in env-parity-expected.json.
//
// Both sides are parsed with regexes rather than imported: settings.py would
// need a Python runtime, and schema.ts would pull in the whole SDK.

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");

const PYTHON_SETTINGS = path.join(
  repoRoot,
  "deepeval",
  "config",
  "settings.py",
);
const TS_SCHEMA = path.join(
  repoRoot,
  "typescript",
  "src",
  "config",
  "schema.ts",
);
const EXPECTED = path.join(here, "env-parity-expected.json");

function pythonFields() {
  const source = fs.readFileSync(PYTHON_SETTINGS, "utf-8");
  // Field declarations on the Settings class, e.g. `    OPENAI_API_KEY: Optional[str] = ...`
  return new Set(
    [...source.matchAll(/^ {4}([A-Z][A-Z0-9_]*)\s*:/gm)].map((m) => m[1]),
  );
}

function typescriptFields() {
  const source = fs.readFileSync(TS_SCHEMA, "utf-8");
  const marker = "export const settingsSchema = z.object({";
  const body = source.slice(source.indexOf(marker) + marker.length);
  return new Set(
    [...body.matchAll(/^ {2}([A-Z][A-Z0-9_]*):/gm)].map((m) => m[1]),
  );
}

/** Every string in the tree, ignoring `$comment` keys. */
function collectNames(node) {
  if (typeof node === "string") return [];
  if (Array.isArray(node)) return node.filter((v) => typeof v === "string");
  return Object.entries(node).flatMap(([key, value]) =>
    key === "$comment" ? [] : collectNames(value),
  );
}

function difference(a, b) {
  return [...a].filter((name) => !b.has(name)).sort();
}

function report(label, names) {
  console.error(`\n${label}`);
  for (const name of names) console.error(`  ${name}`);
}

const python = pythonFields();
const typescript = typescriptFields();
const expected = JSON.parse(fs.readFileSync(EXPECTED, "utf-8"));

const expectedPythonOnly = new Set(collectNames(expected.pythonOnly));
const expectedTypescriptOnly = new Set(collectNames(expected.typescriptOnly));

const actualPythonOnly = difference(python, typescript);
const actualTypescriptOnly = difference(typescript, python);

const undeclaredPythonOnly = actualPythonOnly.filter(
  (name) => !expectedPythonOnly.has(name),
);
const undeclaredTypescriptOnly = actualTypescriptOnly.filter(
  (name) => !expectedTypescriptOnly.has(name),
);
const stalePythonOnly = [...expectedPythonOnly]
  .filter((name) => !actualPythonOnly.includes(name))
  .sort();
const staleTypescriptOnly = [...expectedTypescriptOnly]
  .filter((name) => !actualTypescriptOnly.includes(name))
  .sort();

const failures =
  undeclaredPythonOnly.length +
  undeclaredTypescriptOnly.length +
  stalePythonOnly.length +
  staleTypescriptOnly.length;

if (failures === 0) {
  console.log(
    `Environment variables in sync: ${python.size} Python, ${typescript.size} TypeScript, ` +
      `${[...python].filter((n) => typescript.has(n)).length} shared, ` +
      `${actualPythonOnly.length + actualTypescriptOnly.length} accounted-for differences.`,
  );
  process.exit(0);
}

console.error("Python and TypeScript environment variables have drifted.");

if (undeclaredPythonOnly.length) {
  report(
    "Declared in Python but not TypeScript, and not listed in env-parity-expected.json.\n" +
      "Add them to typescript/src/config/schema.ts, or record why they are Python-only:",
    undeclaredPythonOnly,
  );
}
if (undeclaredTypescriptOnly.length) {
  report(
    "Declared in TypeScript but not Python, and not listed in env-parity-expected.json.\n" +
      "Add them to deepeval/config/settings.py, or record why they are TypeScript-only:",
    undeclaredTypescriptOnly,
  );
}
if (stalePythonOnly.length || staleTypescriptOnly.length) {
  report(
    "Listed in env-parity-expected.json but no longer a difference.\n" +
      "The gap closed, so remove these entries:",
    [...stalePythonOnly, ...staleTypescriptOnly],
  );
}

process.exit(1);
