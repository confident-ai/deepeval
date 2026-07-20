import { LLMTestCase, ConversationalTestCase } from "../../test-case";
import { Golden } from "../../dataset";
import { BaseMetric, BaseConversationalMetric } from "../../metrics";
import { DeepEvalError } from "../../errors";
import { getCurrentTrace } from "../../tracing";
import { AnyTestCase, EvaluatedCase, MetricData } from "../types";
import { ErrorConfig } from "../configs";
import { runMetric, buildTestResult, metricMatchesCase } from "../evaluate";
import { evaluateTrace } from "../trace-eval";
import { AssertionFailedError, buildFailureMessage } from "./errors";
import { globalResultCollector } from "./collector";
import { getLatestCapturedTrace } from "./trace-scope";

type AnyMetric = BaseMetric | BaseConversationalMetric;

export interface AssertTestTraceParams {
  golden: Golden;
  metrics?: BaseMetric[];
}

const STRICT_ERROR_CONFIG: Required<ErrorConfig> = {
  ignoreErrors: false,
  skipOnMissingParams: false,
};

function isTraceScoped(
  arg: AnyTestCase | AssertTestTraceParams,
): arg is AssertTestTraceParams {
  return (
    typeof arg === "object" &&
    arg !== null &&
    !(arg instanceof LLMTestCase) &&
    !(arg instanceof ConversationalTestCase) &&
    "golden" in arg
  );
}

export async function evaluateAssertCase(
  testCase: AnyTestCase,
  metrics: AnyMetric[],
): Promise<EvaluatedCase> {
  if (!metrics || metrics.length === 0) {
    throw new DeepEvalError("assertTest requires at least one metric.");
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
        `metrics in a single assertTest call.`,
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

async function assertTraceScoped(params: AssertTestTraceParams): Promise<void> {
  const trace = getLatestCapturedTrace() ?? getCurrentTrace();
  if (!trace) {
    throw new DeepEvalError(
      "assertTest({ golden }) must be called inside an @observe'd function " +
        "during a `deepeval test run` (no trace was captured for this test).",
    );
  }
  const { golden, metrics } = params;
  if (trace.input == null) trace.input = golden.input;
  if (trace.output == null) trace.output = trace.rootSpans[0]?.output;
  if (trace.expectedOutput == null)
    trace.expectedOutput = golden.expectedOutput;
  if (trace.context == null) trace.context = golden.context;
  if (trace.retrievalContext == null)
    trace.retrievalContext = golden.retrievalContext;
  if (trace.expectedTools == null) trace.expectedTools = golden.expectedTools;

  if (metrics && metrics.length > 0) {
    trace.metrics = [...(trace.metrics ?? []), ...metrics];
  }

  const cases = await evaluateTrace(trace, {
    errorConfig: STRICT_ERROR_CONFIG,
  });
  for (const c of cases) globalResultCollector.record(c);
  const allMetrics: MetricData[] = cases.flatMap((c) => c.metricsData);
  const success = allMetrics.every((m) => m.skipped || m.success);
  if (!success) {
    throw new AssertionFailedError(buildFailureMessage(allMetrics));
  }
}

export async function assertTest(
  testCase: LLMTestCase | ConversationalTestCase,
  metrics: AnyMetric[],
): Promise<void>;
export async function assertTest(params: AssertTestTraceParams): Promise<void>;
export async function assertTest(
  arg1: AnyTestCase | AssertTestTraceParams,
  arg2?: AnyMetric[],
): Promise<void> {
  if (isTraceScoped(arg1)) {
    await assertTraceScoped(arg1);
    return;
  }

  const evaluated = await evaluateAssertCase(arg1, arg2 ?? []);
  globalResultCollector.record(evaluated);
  const testResult = buildTestResult(
    0,
    evaluated.testCase,
    evaluated.metricsData,
  );
  if (!testResult.success) {
    throw new AssertionFailedError(buildFailureMessage(evaluated.metricsData));
  }
}
