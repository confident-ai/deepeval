#!/usr/bin/env node
/**
 * Build-time generator: captures the shipped SDK versions into
 * `lib/generated/sdk-versions.json` so the language selector can label each
 * option with the version a reader would actually install.
 *
 *   { "python": "4.1.5", "typescript": "0.1.32" }
 *
 * The repo declares its version in four independent places — three for Python
 * (`pyproject.toml`, `deepeval/_version.py`, `CITATION.cff`) and one for
 * TypeScript (`typescript/package.json`). Those stay the source of truth; this
 * only reads them, and fails if the three Python ones disagree, since a docs
 * site quoting a version nobody can install is worse than a red build.
 *
 * Outside a full checkout (a docs-only deploy) the sources are missing, so the
 * committed JSON is left in place and the build carries on.
 *
 * Run: `npm run sdk-versions` (also runs pre-build).
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const OUTPUT = 'lib/generated/sdk-versions.json';
const REPO = '..';

const PYTHON_SOURCES = [
  ['pyproject.toml', /^version\s*=\s*"([^"]+)"/m],
  ['deepeval/_version.py', /__version__[^=]*=\s*"([^"]+)"/],
  ['CITATION.cff', /^version:\s*(\S+)/m],
];

function read(file) {
  const path = join(REPO, file);
  return existsSync(path) ? readFileSync(path, 'utf8') : undefined;
}

function pythonVersion() {
  const found = [];

  for (const [file, pattern] of PYTHON_SOURCES) {
    const src = read(file);
    if (src === undefined) return undefined;
    const version = src.match(pattern)?.[1];
    if (!version) throw new Error(`could not parse a version out of ${file}`);
    found.push({ file, version });
  }

  const [first] = found;
  const drifted = found.filter((f) => f.version !== first.version);
  if (drifted.length > 0) {
    throw new Error(
      `Python version declarations disagree — ${found
        .map((f) => `${f.file} says ${f.version}`)
        .join(', ')}`,
    );
  }

  return first.version;
}

function typescriptVersion() {
  const src = read('typescript/package.json');
  if (src === undefined) return undefined;
  const { version } = JSON.parse(src);
  if (!version) throw new Error('typescript/package.json declares no version');
  return version;
}

const python = pythonVersion();
const typescript = typescriptVersion();

if (!python || !typescript) {
  if (!existsSync(OUTPUT)) {
    throw new Error(
      `SDK version sources are unreadable from ${process.cwd()} and there is no committed ${OUTPUT} to fall back on.`,
    );
  }
  console.warn(`[sdk-versions] sources unavailable, keeping ${OUTPUT}`);
  process.exit(0);
}

mkdirSync(join(OUTPUT, '..'), { recursive: true });
writeFileSync(OUTPUT, `${JSON.stringify({ python, typescript }, null, 2)}\n`);
console.log(`[sdk-versions] python ${python}, typescript ${typescript}`);
