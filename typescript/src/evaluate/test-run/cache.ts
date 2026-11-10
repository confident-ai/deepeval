import { expectationEvidence } from "@/evaluate/expectations";
// Metric result caching, the TS counterpart of Python's
// `deepeval/test_run/cache.py`. A result is reusable only when both the test
// case content and the metric's configuration are unchanged.

import * as fs from "fs";
import * as path from "path";
import type { ZodType } from "zod";
import { CACHE_FILE, HIDDEN_DIR } from "@/constants";
import { isReadOnlyFileSystem } from "@/config/utils";
import { getLogger } from "@/logger";
import { ConversationalTestCase, resolveRetrievalContext } from "@/test-case";
import type { BaseMetricCore } from "@/metrics/base-metrics";
import { toJsonSchema } from "@/models/utils";
import type { AnyTestCase, MetricData } from "@/evaluate/types";

const logger = getLogger("deepeval.cache");
const CACHE_VERSION = 1;

/**
 * The metric knobs that change a score.
 */
interface MetricConfiguration {
  evalMode?: string;
  threshold?: number | null;
  evaluationModel?: string;
  strictMode: boolean;
  includeReason: boolean;
  criteria?: string | null;
  evaluationSteps?: string[] | null;
  evaluationParams?: string[] | null;
  rubric?: unknown[] | null;
  expectedSchema?: Record<string, unknown> | null;
}

/** Read off a metric without importing the metric classes into the cache. */
interface MetricDefinition {
  criteria?: string;
  evaluationSteps?: string[];
  evaluationParams?: string[];
  rubric?: unknown[];
  expectedSchema?: ZodType;
}

interface CachedMetric {
  metricData: MetricData;
  configuration: MetricConfiguration;
}

interface CacheFile {
  version: number;
  testCases: Record<string, CachedMetric[]>;
}

function cachePath(): string {
  return path.join(process.cwd(), HIDDEN_DIR, CACHE_FILE);
}

let loaded: CacheFile | null = null;
const pending = new Map<string, CachedMetric[]>();

function emptyCache(): CacheFile {
  return { version: CACHE_VERSION, testCases: {} };
}

function readCache(): CacheFile {
  if (loaded) return loaded;
  try {
    const raw = fs.readFileSync(cachePath(), "utf-8");
    const parsed = JSON.parse(raw) as CacheFile;
    loaded =
      parsed.version === CACHE_VERSION && parsed.testCases
        ? parsed
        : emptyCache();
  } catch {
    loaded = emptyCache();
  }
  return loaded;
}

/** Identity for a test case: every field a metric can read. */
export function testCaseCacheKey(testCase: AnyTestCase): string {
  if (testCase.expectations)
    return JSON.stringify({
      expectations: testCase.expectations.toJSON(),
      evidence: expectationEvidence(testCase),
    });
  if (testCase instanceof ConversationalTestCase) {
    return JSON.stringify({
      turns: testCase.turns.map((turn) => ({
        role: turn.role,
        content: turn.content,
        retrievalContext: resolveRetrievalContext(turn.retrievalContext),
      })),
      scenario: testCase.scenario,
      expectedOutcome: testCase.expectedOutcome,
    });
  }
  return JSON.stringify({
    input: testCase.input,
    actualOutput: testCase.actualOutput,
    expectedOutput: testCase.expectedOutput,
    context: testCase.context,
    retrievalContext: resolveRetrievalContext(testCase.retrievalContext),
    toolsCalled: testCase.toolsCalled,
    expectedTools: testCase.expectedTools,
  });
}

function metricConfiguration(metric: BaseMetricCore): MetricConfiguration {
  const definition = metric as BaseMetricCore & MetricDefinition;
  return {
    evalMode: metric.evalMode,
    threshold: metric.threshold,
    evaluationModel: metric.evaluationModel,
    strictMode: metric.strictMode,
    includeReason: metric.includeReason,
    criteria: definition.criteria ?? null,
    evaluationSteps:
      definition.criteria == null ? (definition.evaluationSteps ?? null) : null,
    evaluationParams: definition.evaluationParams ?? null,
    rubric: definition.rubric ?? null,
    expectedSchema: definition.expectedSchema
      ? toJsonSchema(definition.expectedSchema)
      : null,
  };
}

/** Compares JSON values, so a configuration read back from disk still matches. */
function sameValue(a: unknown, b: unknown): boolean {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function sameConfiguration(
  a: MetricConfiguration,
  b: MetricConfiguration,
): boolean {
  return (
    a.evalMode === b.evalMode &&
    a.threshold === b.threshold &&
    a.evaluationModel === b.evaluationModel &&
    a.strictMode === b.strictMode &&
    a.includeReason === b.includeReason &&
    sameValue(a.criteria, b.criteria) &&
    sameValue(a.evaluationSteps, b.evaluationSteps) &&
    sameValue(a.evaluationParams, b.evaluationParams) &&
    sameValue(a.rubric, b.rubric) &&
    sameValue(a.expectedSchema, b.expectedSchema)
  );
}

export function getCachedMetricData(
  testCase: AnyTestCase,
  metric: BaseMetricCore,
): MetricData | null {
  const entries = readCache().testCases[testCaseCacheKey(testCase)];
  if (!entries) return null;
  const configuration = metricConfiguration(metric);
  const hit = entries.find(
    (entry) =>
      entry.metricData.name === metric.name &&
      sameConfiguration(entry.configuration, configuration),
  );
  return hit ? hit.metricData : null;
}

/** Buffers the result; {@link flushCache} does the single write per run. */
export function cacheMetricData(
  testCase: AnyTestCase,
  metric: BaseMetricCore,
  metricData: MetricData,
): void {
  if (metricData.error || metricData.skipped) return;
  const key = testCaseCacheKey(testCase);
  const entry: CachedMetric = {
    metricData,
    configuration: metricConfiguration(metric),
  };
  const existing = pending.get(key) ?? [];
  pending.set(key, [
    ...existing.filter(
      (candidate) =>
        !(
          candidate.metricData.name === metricData.name &&
          sameConfiguration(candidate.configuration, entry.configuration)
        ),
    ),
    entry,
  ]);
}

export function flushCache(): void {
  if (pending.size === 0) return;
  if (isReadOnlyFileSystem()) {
    pending.clear();
    return;
  }

  // Re-read: sibling vitest workers may have flushed since we loaded it.
  let onDisk: CacheFile;
  try {
    onDisk = JSON.parse(fs.readFileSync(cachePath(), "utf-8")) as CacheFile;
    if (onDisk.version !== CACHE_VERSION || !onDisk.testCases) {
      onDisk = emptyCache();
    }
  } catch {
    onDisk = emptyCache();
  }

  for (const [key, entries] of pending) {
    const merged = (onDisk.testCases[key] ?? []).filter(
      (candidate) =>
        !entries.some(
          (entry) =>
            entry.metricData.name === candidate.metricData.name &&
            sameConfiguration(entry.configuration, candidate.configuration),
        ),
    );
    onDisk.testCases[key] = [...merged, ...entries];
  }

  try {
    fs.mkdirSync(path.dirname(cachePath()), { recursive: true });
    fs.writeFileSync(cachePath(), JSON.stringify(onDisk), "utf-8");
    pending.clear();
    loaded = onDisk;
  } catch (error) {
    logger.warning(`Could not write ${CACHE_FILE}`, error as Error);
  }
}

export function clearCache(): void {
  pending.clear();
  loaded = null;
  if (isReadOnlyFileSystem()) return;
  try {
    fs.rmSync(cachePath(), { force: true });
  } catch {}
}

let exitHookRegistered = false;

/** Metrics also run outside `deepeval test run`, where nothing else flushes. */
export function ensureCacheFlushedOnExit(): void {
  if (exitHookRegistered) return;
  exitHookRegistered = true;
  process.once("exit", () => flushCache());
}
