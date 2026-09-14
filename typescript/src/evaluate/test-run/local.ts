// Local test-run artifacts, mirroring Python's test run cache. The temp JSONL
// directory each worker appends to is deleted after wrap-up, so without these
// files nothing survives a run.

import * as fs from "fs";
import * as path from "path";
import {
  HIDDEN_DIR,
  LATEST_TEST_RUN_FILE,
  DEEPEVAL_RESULTS_FOLDER,
} from "@/constants";
import { isReadOnlyFileSystem } from "@/config/utils";
// `store.ts` only touches `node:sqlite` when a database is actually opened,
// so importing it here costs the JSON path nothing.
import { LOCAL_STORE_SQLITE, resolveLocalStoreMode } from "@/sqlite-store/mode";
import { resolveDbPath, writeTestRun } from "@/sqlite-store/store";
import type { PersistedCase } from "@/evaluate/confident";
import type { ProcessedHyperparameters } from "@/evaluate/hyperparameters";

export interface LocalTestRun {
  /** Confident AI link, once the run has been posted. */
  link: string | null;
  testRunId?: string | null;
  savedAt: string;
  runDuration: number;
  official: boolean;
  identifier?: string;
  hyperparameters?: ProcessedHyperparameters;
  testPassed: number;
  testFailed: number;
  cases: PersistedCase[];
}

function latestTestRunPath(): string {
  return path.join(process.cwd(), HIDDEN_DIR, LATEST_TEST_RUN_FILE);
}

export function saveLatestTestRun(run: LocalTestRun): string {
  const file = latestTestRunPath();
  if (isReadOnlyFileSystem()) return file;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(run, null, 2), "utf-8");
  return file;
}

export function readLatestTestRun(): LocalTestRun | null {
  try {
    const raw = fs.readFileSync(latestTestRunPath(), "utf-8");
    const parsed = JSON.parse(raw) as LocalTestRun;
    return Array.isArray(parsed.cases) ? parsed : null;
  } catch {
    return null;
  }
}

export function deleteLatestTestRun(): void {
  if (isReadOnlyFileSystem()) return;
  try {
    fs.rmSync(latestTestRunPath(), { force: true });
  } catch {}
}

function timestamp(): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const d = new Date();
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  );
}

/**
 * Returns the path written, or null when `DEEPEVAL_RESULTS_FOLDER` is unset or
 * the file system is read-only.
 */
export function exportTestRunJson(run: LocalTestRun): string | null {
  const folder = process.env[DEEPEVAL_RESULTS_FOLDER];
  if (!folder || folder.trim() === "") return null;
  if (isReadOnlyFileSystem()) return null;
  fs.mkdirSync(folder, { recursive: true });
  const file = path.join(folder, `test_run_${timestamp()}.json`);
  fs.writeFileSync(file, JSON.stringify(run, null, 2), "utf-8");
  return file;
}

export interface ExportedTestRun {
  /** `"json"` or `"sqlite"`. */
  mode: "json" | "sqlite";
  /** File written (json) or database path (sqlite). */
  path: string;
  /** Row id in `deepeval.db`; only set for sqlite. */
  runId?: number;
}

/**
 * Persist a finished run according to `DEEPEVAL_LOCAL_STORE`.
 *
 * - `json` (default): `exportTestRunJson`, i.e. a `test_run_<ts>.json` in
 *   `DEEPEVAL_RESULTS_FOLDER` when that is set; `null` otherwise.
 * - `sqlite`: appends the run to `deepeval.db` (in `DEEPEVAL_RESULTS_FOLDER`
 *   or the cache dir). Storage failures are reported as a warning and yield
 *   `null` so a locked or read-only database never fails the evaluation.
 *
 * Asking for sqlite on a Node older than 24 throws `SqliteUnsupportedError`
 * from `resolveLocalStoreMode` — that is a configuration error, not a
 * storage hiccup, so it is not swallowed here.
 */
export function exportTestRun(run: LocalTestRun): ExportedTestRun | null {
  if (resolveLocalStoreMode() !== LOCAL_STORE_SQLITE) {
    const file = exportTestRunJson(run);
    return file ? { mode: "json", path: file } : null;
  }

  if (isReadOnlyFileSystem()) return null;
  const dbPath = resolveDbPath();
  try {
    const runId = writeTestRun(run, dbPath);
    return { mode: "sqlite", path: dbPath, runId };
  } catch (e) {
    console.warn(
      `Warning: failed to save test run to ${dbPath}: ${(e as Error).message}`,
    );
    return null;
  }
}

export function summarizeCases(cases: PersistedCase[]): {
  testPassed: number;
  testFailed: number;
} {
  let testPassed = 0;
  let testFailed = 0;
  for (const { entry } of cases) {
    if (entry.success) testPassed += 1;
    else testFailed += 1;
  }
  return { testPassed, testFailed };
}
