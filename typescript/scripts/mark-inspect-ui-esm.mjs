// Node reads a .js file's module format from the nearest package.json. The root
// package has no "type", so dist/ is CommonJS — this marker carves out the
// inspect UI as ESM, which Ink requires.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const outDir = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "dist",
  "inspect",
  "ui",
);

mkdirSync(outDir, { recursive: true });
writeFileSync(
  join(outDir, "package.json"),
  `${JSON.stringify({ type: "module" }, null, 2)}\n`,
);
