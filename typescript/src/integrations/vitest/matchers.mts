import { BaseMetric, BaseConversationalMetric } from "../../metrics/index.js";
import {
  runMetrics,
  type ToPassOptions,
  type ToPassTarget,
} from "../../evaluate/test-run/index.js";

type AnyMetric = BaseMetric | BaseConversationalMetric;

function describeTarget(target: ToPassTarget): string {
  return typeof target === "function"
    ? "the callback's trace"
    : "the test case";
}

/**
 * `expect(() => myAgent(input)).toPass([metric])` — run the callback and evaluate
 * the trace it produces. Add `{ golden }` for expected values.
 * `expect(testCase).toPass([metricA, metricB])` — evaluate a test case.
 */
export async function toPass(
  this: { isNot?: boolean },
  received: ToPassTarget,
  metrics: AnyMetric[] = [],
  options: ToPassOptions = {},
) {
  const { pass, failureMessage } = await runMetrics(received, metrics, options);
  return {
    pass,
    message: () =>
      pass
        ? `Expected ${describeTarget(received)} NOT to pass its metrics, but it did.`
        : failureMessage,
  };
}
