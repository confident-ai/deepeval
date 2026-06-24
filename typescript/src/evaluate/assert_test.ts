import {
  BaseMetric,
  BaseConversationalMetric,
} from "../metrics";
import {
  LLMTestCase,
  ConversationalTestCase,
} from "../test-case";
import { evaluate, runMetric, buildTestResult } from "./evaluate";
import {
  AsyncConfig,
  DisplayConfig,
  ErrorConfig,
  CacheConfig,
  DEFAULT_ASYNC_CONFIG,
  DEFAULT_DISPLAY_CONFIG,
  DEFAULT_ERROR_CONFIG,
  DEFAULT_CACHE_CONFIG,
} from "./configs";
import { MetricData, TestResult } from "./types";

/**
 * DeepEval assertion error with structured failure details.
 * Mirrors the `AssertionError` raised by Python's `assert_test()`.
 */
export class AssertionError extends Error {
  public failedMetrics: MetricData[];

  constructor(failedMetrics: MetricData[]) {
    const details = failedMetrics
      .map(
        (m) =>
          `  - ${m.name}: score=${m.score}, threshold=${m.threshold}` +
          (m.reason ? `, reason="${m.reason}"` : "") +
          (m.error ? `, error="${m.error}"` : ""),
      )
      .join("\n");

    super(
      `DeepEval assertion failed — ${failedMetrics.length} metric(s) did not pass:\n${details}`,
    );
    this.name = "DeepEvalAssertionError";
    this.failedMetrics = failedMetrics;
  }
}

/**
 * Assert that a test case passes the given metrics. When a golden is provided
 * instead of a test case, it is converted to an LLMTestCase.
 *
 * Mirrors Python's `deepeval.evaluate.assert_test`.
 *
 * @example
 * ```ts
 * import { assertTest, LLMTestCase, FaithfulnessMetric } from "deepeval";
 *
 * test("my llm test", async () => {
 *   const result = await assertTest({
 *     testCase: new LLMTestCase({
 *       input: "What is the capital of France?",
 *       actualOutput: "The capital of France is Paris.",
 *       retrievalContext: ["France is a country in Europe."],
 *     }),
 *     metrics: [new FaithfulnessMetric()],
 *   });
 *   // result contains testResult with pass/fail details
 * });
 * ```
 */
export async function assertTest(options: {
  testCase?: LLMTestCase | ConversationalTestCase;
  metrics?: (BaseMetric | BaseConversationalMetric)[];
}): Promise<TestResult> {
  const { testCase, metrics } = options;

  if (!testCase || !metrics || metrics.length === 0) {
    throw new Error(
      "assertTest requires both `testCase` (or `golden`) and `metrics`.",
    );
  }

  const asyncCfg: Required<AsyncConfig> = DEFAULT_ASYNC_CONFIG;
  const displayCfg: Required<DisplayConfig> = {
    ...DEFAULT_DISPLAY_CONFIG,
    showIndicator: false,
    printResults: false,
  };
  const errorCfg: Required<ErrorConfig> = DEFAULT_ERROR_CONFIG;
  const cacheCfg: Required<CacheConfig> = DEFAULT_CACHE_CONFIG;

  const result = await evaluate(
    [testCase],
    metrics,
    {
      asyncConfig: asyncCfg,
      displayConfig: displayCfg,
      errorConfig: errorCfg,
      cacheConfig: cacheCfg,
    },
  );

  const testResult = result.testResults[0];
  if (!testResult) {
    throw new Error("assertTest: no test results returned.");
  }

  const failedMetrics = (testResult.metricsData ?? []).filter(
    (m) => !m.skipped && !m.success,
  );

  if (failedMetrics.length > 0) {
    throw new AssertionError(failedMetrics);
  }

  return testResult;
}
