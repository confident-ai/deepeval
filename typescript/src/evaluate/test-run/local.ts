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
