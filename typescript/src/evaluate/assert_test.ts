import { BaseMetric, BaseConversationalMetric } from "../metrics";
import type { LLMTestCase, ConversationalTestCase } from "../test-case";
import { runMetric } from "./evaluate";
import type { MetricData, TestResult } from "./types";
import { DEFAULT_ERROR_CONFIG } from "./configs";

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

export async function assertTest(options: {
  testCase?: LLMTestCase | ConversationalTestCase;
  metrics?: (BaseMetric | BaseConversationalMetric)[];
}): Promise<TestResult> {
  const { testCase, metrics } = options;

  if (!testCase || !metrics || metrics.length === 0) {
    throw new Error(
      "assertTest requires both `testCase` and `metrics`.",
    );
  }

  const metricsData: MetricData[] = [];

  for (const metric of metrics) {
    metric.showIndicator = false;
    const data = await runMetric(metric, testCase, DEFAULT_ERROR_CONFIG, () => {});
    metricsData.push(data);
  }

  const failedMetrics = metricsData.filter(
    (m) => !m.skipped && !m.success,
  );

  if (failedMetrics.length > 0) {
    throw new AssertionError(failedMetrics);
  }

  return {
    name: testCase instanceof Object ? (testCase as any).name ?? "assert_test" : "assert_test",
    success: true,
    metricsData,
    conversational: testCase.constructor?.name === "ConversationalTestCase",
  };
}
