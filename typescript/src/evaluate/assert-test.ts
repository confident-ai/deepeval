/**
 * assertTest() — the TypeScript equivalent of Python's `assert_test`.
 *
 * Runs a set of metrics against a single test case, records the outcome in
 * the shared {@link testRunManager}, and throws an `AssertionError`-style
 * error when one or more metrics fail.
 *
 * Unlike the Python version, this is a plain `async` function that works in
 * any runtime context — no pytest / Jest / Vitest adapter is required.
 *
 * @example
 * ```ts
 * import { assertTest, LLMTestCase } from "deepeval";
 * import { AnswerRelevancyMetric } from "deepeval/metrics";
 *
 * const metric = new AnswerRelevancyMetric({ threshold: 0.7 });
 * const testCase = new LLMTestCase({
 *   input: "What is the capital of France?",
 *   actualOutput: "Paris",
 * });
 *
 * await assertTest(testCase, [metric]);
 * // Throws AssertTestError if any metric fails.
 * ```
 */

import { BaseMetric } from "../metrics/base-metrics";
import { BaseConversationalMetric } from "../metrics/base-conversational-metric";
import { LLMTestCase } from "../test-case/llm-test-case";
import { ConversationalTestCase } from "../test-case/conversational-test-case";
import { runMetric, buildTestResult } from "./evaluate";
import { testRunManager } from "./test-run-manager";
import { MetricData } from "./types";
import { ErrorConfig, DEFAULT_ERROR_CONFIG } from "./configs";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Options accepted by {@link assertTest}. */
export interface AssertTestOptions {
  /**
   * Error-handling policy forwarded to the metric runner.
   *
   * Defaults match the Python `assert_test` behaviour:
   * - `ignoreErrors: false` — metric execution errors propagate.
   * - `skipOnMissingParams: false` — missing required params are an error.
   */
  errorConfig?: ErrorConfig;
  /**
   * When `true`, suppress the per-metric progress indicator.
   * Defaults to `true` (show indicator), mirroring Python's `assert_test`.
   */
  showIndicator?: boolean;
}

// ---------------------------------------------------------------------------
// Error class
// ---------------------------------------------------------------------------

/**
 * Thrown by {@link assertTest} when one or more metrics fail.
 *
 * Extends the built-in `Error` so it integrates cleanly with any testing
 * framework (`instanceof Error === true`).  The `failedMetrics` array
 * gives programmatic access to the individual metric results.
 */
export class AssertTestError extends Error {
  /** Every metric that did not pass. */
  readonly failedMetrics: MetricData[];

  constructor(failedMetrics: MetricData[]) {
    const parts = failedMetrics.map(
      (m) =>
        `${m.name} (score: ${m.score ?? "n/a"}, threshold: ${m.threshold}, ` +
        `strict: ${m.strictMode}, error: ${m.error ?? "n/a"}, reason: ${m.reason ?? "n/a"})`,
    );
    super(`Metrics: ${parts.join(", ")} failed.`);
    this.name = "AssertTestError";
    this.failedMetrics = failedMetrics;
  }
}

// ---------------------------------------------------------------------------
// Supported input types (mirrors Python's Union signature)
// ---------------------------------------------------------------------------

type AnyTestCase = LLMTestCase | ConversationalTestCase;
type AnyMetric = BaseMetric | BaseConversationalMetric;

// ---------------------------------------------------------------------------
// assertTest()
// ---------------------------------------------------------------------------

/**
 * Run `metrics` against `testCase`, record the outcome, and throw if any
 * metric fails.
 *
 * All metrics are executed **concurrently** using `Promise.all`, matching
 * Python's `run_async=True` default in `assert_test`.
 *
 * The result is recorded in the module-level {@link testRunManager} so
 * callers can aggregate multiple `assertTest` invocations with
 * `testRunManager.getSummary()`.
 *
 * @param testCase - The test case to evaluate.
 * @param metrics  - One or more metrics to run.
 * @param options  - Optional configuration (error policy, indicator display).
 *
 * @throws {AssertTestError} When one or more metrics fail or error
 *   (subject to `errorConfig.ignoreErrors`).
 * @throws The original error from a metric when `errorConfig.ignoreErrors`
 *   is `false` and the metric throws a non-`MissingTestCaseParamsError`.
 */
export async function assertTest(
  testCase: AnyTestCase,
  metrics: AnyMetric[],
  options: AssertTestOptions = {},
): Promise<void> {
  if (!testCase) {
    throw new Error("assertTest: 'testCase' must not be null or undefined.");
  }
  if (!Array.isArray(metrics) || metrics.length === 0) {
    throw new Error(
      "assertTest: 'metrics' must be a non-empty array of metrics.",
    );
  }

  // Build the effective error config (defaults match Python's assert_test).
  const errorCfg: Required<ErrorConfig> = {
    ...DEFAULT_ERROR_CONFIG,
    ...options.errorConfig,
  };

  // Honour the showIndicator option per metric (Python: show_indicator=True).
  const showIndicator = options.showIndicator ?? true;
  const originalShowIndicator = metrics.map((m) => m.showIndicator);
  metrics.forEach((m) => {
    m.showIndicator = showIndicator;
  });

  // Determine the index for naming purposes (count of previously recorded results).
  const index = testRunManager.getResults().length;

  let metricsData: MetricData[];
  try {
    // Run all metrics concurrently — mirrors Python's async default.
    metricsData = await Promise.all(
      metrics.map((m) => runMetric(m, testCase, errorCfg, () => {})),
    );
  } finally {
    // Always restore showIndicator, even if a metric throws.
    metrics.forEach((m, i) => {
      m.showIndicator = originalShowIndicator[i];
    });
  }

  // Build and record the test result.
  const testResult = buildTestResult(index, testCase, metricsData);
  testRunManager.record(testResult);

  // Collect failing metrics and throw if any exist.
  const failedMetrics = metricsData.filter((m) => {
    if (m.skipped) return false; // skipped metrics never cause a failure
    if (m.error !== undefined) return true; // errored metric = failure
    return !m.success;
  });

  if (failedMetrics.length > 0) {
    throw new AssertTestError(failedMetrics);
  }
}
