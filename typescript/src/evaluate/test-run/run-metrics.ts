import { ConversationalTestCase } from "../../test-case";
import { Golden } from "../../dataset";
import { BaseMetric, BaseConversationalMetric } from "../../metrics";
import { DeepEvalError } from "../../errors";
import { getCurrentTrace } from "../../tracing";
import { AnyTestCase, EvaluatedCase, MetricData } from "../types";
import { ErrorConfig } from "../configs";
import { runMetric, buildTestResult, metricMatchesCase } from "../evaluate";
import { evaluateTrace } from "../trace-eval";
import { buildFailureMessage } from "./errors";
import { globalResultCollector } from "./collector";
import { getLatestCapturedTrace } from "./trace-scope";

type AnyMetric = BaseMetric | BaseConversationalMetric;

/** Anything `expect(...).toPass()` can be called on. */
export type ToPassTarget = AnyTestCase | Golden;

/**
 * The verdict for one `toPass` call. Metric failures are reported here rather
 * than thrown — it is the test framework's job to decide what a failure means.
 * Genuine misuse (no metrics, wrong metric type, no trace) still throws.
 */
export interface MetricsOutcome {
  pass: boolean;
  failureMessage: string;
}

const STRICT_ERROR_CONFIG: Required<ErrorConfig> = {
  ignoreErrors: false,
  skipOnMissingParams: false,
};

export async function evaluateCase(
  testCase: AnyTestCase,
  metrics: AnyMetric[],
): Promise<EvaluatedCase> {
  if (!metrics || metrics.length === 0) {
    throw new DeepEvalError("toPass requires at least one metric.");
  }
  const mismatched = metrics.filter((m) => !metricMatchesCase(m, testCase));
  if (mismatched.length > 0) {
    const isConversational = testCase instanceof ConversationalTestCase;
    const caseKind = isConversational
      ? "a ConversationalTestCase (multi-turn)"
      : "an LLMTestCase (single-turn)";
    const expected = isConversational ? "multi-turn" : "single-turn";
    const names = mismatched.map((m) => m.name).join(", ");
    throw new DeepEvalError(
      `Metric(s) [${names}] cannot evaluate ${caseKind}: it requires ` +
        `${expected} metrics only. Do not mix single-turn and multi-turn ` +
        `metrics in a single toPass call.`,
    );
  }

  const start = Date.now();
  const metricsData = await Promise.all(
    metrics.map((m) => runMetric(m, testCase, STRICT_ERROR_CONFIG, () => {})),
  );
  return {
    testCase,
    metricsData,
    runDuration: (Date.now() - start) / 1000,
  };
}

/** Run `metrics` against an explicit test case. */
export async function runTestCaseMetrics(
  testCase: AnyTestCase,
  metrics: AnyMetric[],
): Promise<MetricsOutcome> {
  const evaluated = await evaluateCase(testCase, metrics);
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

/**
 * Evaluate the trace the current test just produced, filling any gaps in it
 * from `golden`. `metrics` are trace-level and add to whatever the spans
 * already carry, so passing none is valid.
 */
export async function runTraceMetrics(
  golden: Golden,
  metrics: AnyMetric[] = [],
): Promise<MetricsOutcome> {
  const conversational = metrics.filter(
    (m) => m instanceof BaseConversationalMetric,
  );
  if (conversational.length > 0) {
    const names = conversational.map((m) => m.name).join(", ");
    throw new DeepEvalError(
      `Metric(s) [${names}] cannot evaluate a trace: trace-level metrics must ` +
        `be single-turn.`,
    );
  }

  const trace = getLatestCapturedTrace() ?? getCurrentTrace();
  if (!trace) {
    throw new DeepEvalError(
      "expect(golden).toPass() must be called after invoking an @observe'd " +
        "function during a `deepeval test run` (no trace was captured for " +
        "this test).",
    );
  }

  if (trace.input == null) trace.input = golden.input;
  if (trace.output == null) trace.output = trace.rootSpans[0]?.output;
  if (trace.expectedOutput == null) trace.expectedOutput = golden.expectedOutput;
  if (trace.context == null) trace.context = golden.context;
  if (trace.retrievalContext == null)
    trace.retrievalContext = golden.retrievalContext;
  if (trace.expectedTools == null) trace.expectedTools = golden.expectedTools;

  if (metrics.length > 0) {
    trace.metrics = [...(trace.metrics ?? []), ...(metrics as BaseMetric[])];
  }

  const cases = await evaluateTrace(trace, {
    errorConfig: STRICT_ERROR_CONFIG,
  });
  for (const c of cases) globalResultCollector.record(c);
  const allMetrics: MetricData[] = cases.flatMap((c) => c.metricsData);
  return {
    pass: allMetrics.every((m) => m.skipped || m.success),
    failureMessage: buildFailureMessage(allMetrics),
  };
}

/**
 * Single entry point behind `expect(target).toPass(metrics)`. A `Golden` means
 * "evaluate the trace this test produced"; a test case is evaluated directly.
 */
export async function runMetrics(
  target: ToPassTarget,
  metrics: AnyMetric[] = [],
): Promise<MetricsOutcome> {
  return target instanceof Golden
    ? runTraceMetrics(target, metrics)
    : runTestCaseMetrics(target, metrics);
}
