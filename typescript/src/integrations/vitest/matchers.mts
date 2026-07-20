import type { LLMTestCase, ConversationalTestCase } from "../../test-case/index.js";
import { BaseMetric, BaseConversationalMetric } from "../../metrics/index.js";
import { buildTestResult } from "../../evaluate/evaluate.js";
import {
  evaluateAssertCase,
  buildFailureMessage,
  globalResultCollector,
} from "../../evaluate/assert-test/index.js";

type AnyTestCase = LLMTestCase | ConversationalTestCase;
type AnyMetric = BaseMetric | BaseConversationalMetric;

async function runMatcher(
  testCase: AnyTestCase,
  metrics: AnyMetric[],
): Promise<{ pass: boolean; failureMessage: string }> {
  const evaluated = await evaluateAssertCase(testCase, metrics);
  globalResultCollector.record(evaluated);
  const testResult = buildTestResult(
    0,
    evaluated.testCase,
    evaluated.metricsData,
  );
  return {
    pass: testResult.success,
    failureMessage: buildFailureMessage(evaluated.metricsData),
  };
}

/** `expect(testCase).toPassMetric(metric)` */
export async function toPassMetric(
  this: { isNot?: boolean },
  received: AnyTestCase,
  metric: AnyMetric,
) {
  const { pass, failureMessage } = await runMatcher(received, [metric]);
  return {
    pass,
    message: () =>
      pass
        ? `Expected the test case NOT to pass metric "${metric.name}", but it did.`
        : failureMessage,
  };
}

/** `expect(testCase).toPassAll([metricA, metricB])` */
export async function toPassAll(
  this: { isNot?: boolean },
  received: AnyTestCase,
  metrics: AnyMetric[],
) {
  const { pass, failureMessage } = await runMatcher(received, metrics);
  return {
    pass,
    message: () =>
      pass
        ? `Expected the test case NOT to pass all metrics, but it did.`
        : failureMessage,
  };
}
