import { Golden } from "../../dataset/index.js";
import { BaseMetric, BaseConversationalMetric } from "../../metrics/index.js";
import {
  runMetrics,
  type ToPassTarget,
} from "../../evaluate/test-run/index.js";

type AnyMetric = BaseMetric | BaseConversationalMetric;

function describeTarget(target: ToPassTarget): string {
  return target instanceof Golden ? "the trace" : "the test case";
}

/**
 * `expect(testCase).toPass([metricA, metricB])` — evaluate a test case.
 * `expect(golden).toPass([metric])` — evaluate the trace this test produced.
 */
export async function toPass(
  this: { isNot?: boolean },
  received: ToPassTarget,
  metrics: AnyMetric[] = [],
) {
  const { pass, failureMessage } = await runMetrics(received, metrics);
  return {
    pass,
    message: () =>
      pass
        ? `Expected ${describeTarget(received)} NOT to pass its metrics, but it did.`
        : failureMessage,
  };
}
