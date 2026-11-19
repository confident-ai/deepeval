// Resolves `DEEPEVAL_LOCAL_STORE` and gates the SQLite backend on the Node
// version that ships a stable `node:sqlite`. Kept free of any `node:sqlite`
// import so it can be evaluated (and rejected) on runtimes that lack it.

import { DEEPEVAL_LOCAL_STORE } from "@/constants";

export const LOCAL_STORE_JSON = "json";
export const LOCAL_STORE_SQLITE = "sqlite";
export type LocalStoreMode =
  | typeof LOCAL_STORE_JSON
  | typeof LOCAL_STORE_SQLITE;

/** First Node major whose `node:sqlite` is stable (no experimental warning). */
export const MIN_NODE_MAJOR_FOR_SQLITE = 24;

const SQLITE_ALIASES = new Set([LOCAL_STORE_SQLITE, "sqlite3", "db"]);

export class SqliteUnsupportedError extends Error {}

export function nodeMajor(version: string = process.versions.node): number {
  const major = Number.parseInt(version.split(".")[0] ?? "", 10);
  return Number.isNaN(major) ? 0 : major;
}

export function isSqliteSupported(
  version: string = process.versions.node,
): boolean {
  return nodeMajor(version) >= MIN_NODE_MAJOR_FOR_SQLITE;
}

export function sqliteUnsupportedMessage(
  version: string = process.versions.node,
): string {
  return (
    `${DEEPEVAL_LOCAL_STORE}=sqlite requires Node.js ${MIN_NODE_MAJOR_FOR_SQLITE} ` +
    `or newer (you are on v${version}), because the SQLite store is built on ` +
    "the built-in `node:sqlite` module. Upgrade Node, or unset " +
    `${DEEPEVAL_LOCAL_STORE} (or set it to json) to keep the JSON exports.`
  );
}

/** Throws `SqliteUnsupportedError` when this runtime cannot back the store. */
export function assertSqliteSupported(
  version: string = process.versions.node,
): void {
  if (!isSqliteSupported(version)) {
    throw new SqliteUnsupportedError(sqliteUnsupportedMessage(version));
  }
}

/**
 * Normalize a raw `DEEPEVAL_LOCAL_STORE` value. Returns `undefined` for
 * values that are neither json nor a sqlite alias.
 */
export function normalizeLocalStoreMode(
  value: string | undefined,
): LocalStoreMode | undefined {
  if (value === undefined) return undefined;
  const s = value.trim().toLowerCase();
  if (s === "" || s === LOCAL_STORE_JSON) return LOCAL_STORE_JSON;
  if (SQLITE_ALIASES.has(s)) return LOCAL_STORE_SQLITE;
  return undefined;
}

/**
 * `"json"` (default) or `"sqlite"` from the environment.
 *
 * Mirrors Python's `resolve_local_store_mode()`: an unknown value warns and
 * falls back to json so a typo never silently drops results. Asking for
 * sqlite on a Node older than {@link MIN_NODE_MAJOR_FOR_SQLITE} throws, since
 * there is no backend to fall back to that would honour the request.
 */
export function resolveLocalStoreMode(
  env: NodeJS.ProcessEnv = process.env,
): LocalStoreMode {
  const raw = env[DEEPEVAL_LOCAL_STORE];
  const mode = normalizeLocalStoreMode(raw);
  if (mode === undefined) {
    if (raw !== undefined && raw.trim() !== "") {
      console.warn(
        `Warning: unrecognised ${DEEPEVAL_LOCAL_STORE}=${JSON.stringify(raw)}; ` +
          "falling back to 'json'. Valid values: 'json', 'sqlite'.",
      );
    }
    return LOCAL_STORE_JSON;
  }
  if (mode === LOCAL_STORE_SQLITE) assertSqliteSupported();
  return mode;
}
