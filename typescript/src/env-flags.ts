// Readers for the env vars `deepeval test run` sets for its workers. Kept
// dependency-free so metrics and the evaluate pipeline can both use it without
// import cycles.

import { parseBool } from "@/config/utils";
import {
  DEEPEVAL_DISPLAY,
  DEEPEVAL_MAX_CONCURRENT,
  DEEPEVAL_VERBOSE_MODE,
  ENABLE_DEEPEVAL_CACHE,
  IGNORE_DEEPEVAL_ERRORS,
  SKIP_DEEPEVAL_MISSING_PARAMS,
} from "@/constants";

export function envBool(name: string): boolean | undefined {
  return parseBool(process.env[name]);
}

export function envNumber(name: string): number | undefined {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

/** Which test cases to print at the end of a run (Python's `--display`). */
export type TestRunDisplay = "all" | "passing" | "failing";

export function getDisplay(): TestRunDisplay {
  const raw = (process.env[DEEPEVAL_DISPLAY] || "").trim().toLowerCase();
  return raw === "passing" || raw === "failing" ? raw : "all";
}

export function isVerboseMode(): boolean {
  return envBool(DEEPEVAL_VERBOSE_MODE) ?? false;
}

export function shouldIgnoreErrors(): boolean {
  return envBool(IGNORE_DEEPEVAL_ERRORS) ?? false;
}

export function shouldSkipOnMissingParams(): boolean {
  return envBool(SKIP_DEEPEVAL_MISSING_PARAMS) ?? false;
}

export function shouldUseCache(): boolean {
  return envBool(ENABLE_DEEPEVAL_CACHE) ?? false;
}

export function getMaxConcurrent(): number {
  const value = envNumber(DEEPEVAL_MAX_CONCURRENT);
  return value && value > 0 ? Math.floor(value) : 100;
}

/** `Promise.all(items.map(fn))` bounded to `limit` in flight; order preserved. */
export async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (items.length <= 1 || limit <= 1) {
    const results: R[] = [];
    for (let i = 0; i < items.length; i++) results.push(await fn(items[i], i));
    return results;
  }

  const results = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, () =>
    (async () => {
      while (true) {
        const index = next++;
        if (index >= items.length) return;
        results[index] = await fn(items[index], index);
      }
    })(),
  );
  await Promise.all(workers);
  return results;
}
