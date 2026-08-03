// Dotenv autoloading, mirroring Python's `.env` -> `.env.local` layering: real
// environment variables always win, and `.env.local` beats `.env`. Merged by
// hand because dotenv's `override: false` would let `.env` shadow `.env.local`.

import * as fs from "fs";
import * as path from "path";

const DOTENV_FILES = [".env", ".env.local"];

let loadedFromFiles = new Set<string>();
let loadedPaths: string[] = [];

export function loadDotenvFiles(): void {
  const merged: Record<string, string> = {};
  const paths: string[] = [];

  for (const file of DOTENV_FILES) {
    const filePath = path.join(process.cwd(), file);
    if (!fs.existsSync(filePath)) continue;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { parse } = require("dotenv") as {
        parse: (src: string) => Record<string, string>;
      };
      Object.assign(merged, parse(fs.readFileSync(filePath, "utf-8")));
      paths.push(filePath);
    } catch {}
  }

  const applied = new Set<string>();
  for (const [key, value] of Object.entries(merged)) {
    if (process.env[key] === undefined) {
      process.env[key] = value;
      applied.add(key);
    }
  }

  loadedFromFiles = applied;
  loadedPaths = paths;
}

/** Whether a value in `process.env` came from a dotenv file we loaded. */
export function cameFromDotenv(key: string): boolean {
  return loadedFromFiles.has(key);
}

export function loadedDotenvPaths(): string[] {
  return [...loadedPaths];
}
