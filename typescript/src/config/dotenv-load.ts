// Dotenv autoloading, mirroring Python's `autoload_dotenv`: `.env` ->
// `.env.<APP_ENV>` -> `.env.local`, where later files win and real environment
// variables always win. Merged by hand because dotenv's `override: false`
// would let `.env` shadow the files that should outrank it.

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { APP_ENV, DEEPEVAL_DISABLE_DOTENV, ENV_DIR_PATH } from "@/constants";
import { envBool } from "@/env-flags";

let loadedFromFiles = new Set<string>();
let loadedPaths: string[] = [];

function readDotenvFile(filePath: string): Record<string, string> {
  if (!fs.existsSync(filePath)) return {};
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { parse } = require("dotenv") as {
      parse: (src: string) => Record<string, string>;
    };
    return parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return {};
  }
}

/** Python's `os.path.expanduser` + `os.path.expandvars`, narrowed to env vars. */
function expand(value: string): string {
  const expanded = value.replace(
    /\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))/g,
    (match, braced: string | undefined, bare: string | undefined) =>
      process.env[braced ?? bare ?? ""] ?? match,
  );
  return expanded.startsWith("~")
    ? path.join(os.homedir(), expanded.slice(1))
    : expanded;
}

/** Directory holding the dotenv files: `ENV_DIR_PATH`, else the CWD. */
export function resolveEnvDir(): string {
  const raw = process.env[ENV_DIR_PATH];
  return raw && raw.trim() ? expand(raw) : process.cwd();
}

function resolveAppEnv(envDir: string): string | undefined {
  const fromProcess = process.env[APP_ENV]?.trim();
  if (fromProcess) return fromProcess;
  // `.env.local` outranks `.env` when discovering which layer to load.
  for (const file of [".env.local", ".env"]) {
    const value = readDotenvFile(path.join(envDir, file))[APP_ENV]?.trim();
    if (value) return value;
  }
  return undefined;
}

/** The dotenv files in load order, lowest precedence first. */
export function dotenvSearchPaths(
  envDir: string = resolveEnvDir(),
  appEnv: string | undefined = resolveAppEnv(envDir),
): string[] {
  const paths = [path.join(envDir, ".env")];
  if (appEnv) paths.push(path.join(envDir, `.env.${appEnv}`));
  paths.push(path.join(envDir, ".env.local"));
  return paths;
}

export function loadDotenvFiles(): void {
  loadedFromFiles = new Set();
  loadedPaths = [];

  if (envBool(DEEPEVAL_DISABLE_DOTENV)) return;

  const envDir = resolveEnvDir();
  const appEnv = resolveAppEnv(envDir);

  // Seed the discovered APP_ENV; file values below may still override it.
  const merged: Record<string, string> = appEnv ? { [APP_ENV]: appEnv } : {};

  for (const filePath of dotenvSearchPaths(envDir, appEnv)) {
    if (!fs.existsSync(filePath)) continue;
    Object.assign(merged, readDotenvFile(filePath));
    loadedPaths.push(filePath);
  }

  for (const [key, value] of Object.entries(merged)) {
    if (process.env[key] === undefined) {
      process.env[key] = value;
      loadedFromFiles.add(key);
    }
  }
}

/** Whether a value in `process.env` came from a dotenv file we loaded. */
export function cameFromDotenv(key: string): boolean {
  return loadedFromFiles.has(key);
}

export function loadedDotenvPaths(): string[] {
  return [...loadedPaths];
}
