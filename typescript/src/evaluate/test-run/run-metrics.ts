import { ConversationalTestCase } from "@/test-case";
import { Golden } from "@/dataset";
import { BaseMetric, BaseConversationalMetric } from "@/metrics";
import { DeepEvalError } from "@/errors";
import {
  getMaxConcurrent,
  mapWithConcurrency,
  shouldIgnoreErrors,
  shouldSkipOnMissingParams,
} from "@/env-flags";
import {
  AnyTestCase,
  EvaluatedCase,
  MetricData,
  aggregateSuccess,
} from "@/evaluate/types";
import { checkAtLeastOneMetricHasThreshold } from "@/metrics/utils";
import { ErrorConfig } from "@/evaluate/configs";
import {
  runMetric,
  buildTestResult,
  metricMatchesCase,
} from "@/evaluate/evaluate";
import {
  evaluateTrace,
  primaryTraceFor,
  turnTestCase,
} from "@/evaluate/trace-eval";
import { buildFailureMessage } from "@/evaluate/test-run/errors";
import { globalResultCollector } from "@/evaluate/test-run/collector";
import { collectTracesFrom } from "@/evaluate/test-run/trace-scope";
import { traceManager, type Trace } from "@/tracing/tracing";

type AnyMetric = BaseMetric | BaseConversationalMetric;

export type ToPassCallback = () => unknown;

/** Anything `expect(...).toPass()` can be called on. */
export type ToPassTarget = AnyTestCase | ToPassCallback;

/** Trace-level metrics judge a single turn, so they must be single-turn. */
function assertSingleTurn(metrics: AnyMetric[]): void {
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
}

/** Fill the trace's eval fields from the golden, leaving anything already set. */
function applyGoldenToTrace(trace: Trace, golden: Golden): void {
  if (trace.input == null) trace.input = golden.input;
  if (trace.output == null) trace.output = trace.rootSpans[0]?.output;
  if (trace.expectedOutput == null) trace.expectedOutput = golden.expectedOutput;
  if (trace.context == null) trace.context = golden.context;
  if (trace.retrievalContext == null)
    trace.retrievalContext = golden.retrievalContext;
  if (trace.expectedTools == null) trace.expectedTools = golden.expectedTools;
}

/**
 * The verdict for one `toPass` call. Metric failures are reported here rather
 * than thrown — it is the test framework's job to decide what a failure means.
 * Genuine misuse (no metrics, wrong metric type, no trace) still throws.
 */
export interface MetricsOutcome {
  pass: boolean;
  failureMessage: string;
}

/** Strict by default; `deepeval test run` flags relax it for the whole run. */
function runErrorConfig(): Required<ErrorConfig> {
  return {
    ignoreErrors: shouldIgnoreErrors(),
    skipOnMissingParams: shouldSkipOnMissingParams(),
  };
}

export async function evaluateCase(
  testCase: AnyTestCase,
  metrics: AnyMetric[],
): Promise<EvaluatedCase> {
  if (!metrics || metrics.length === 0) {
    throw new DeepEvalError("toPass requires at least one metric.");
  }
  checkAtLeastOneMetricHasThreshold(metrics);
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
  const errorConfig = runErrorConfig();
  const metricsData = await mapWithConcurrency(
    metrics,
    getMaxConcurrent(),
    (m) => runMetric(m, testCase, errorConfig, () => {}),
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
  const failureMessage = buildFailureMessage(evaluated.metricsData);

  // A flaky test case reports its failure without failing the test (Python's
  // assert_test warns rather than raising).
  if (!testResult.success && testCase.flaky) {
    console.warn(
      `Flaky test case failed (not reported as a failure): ${failureMessage}`,
    );
    return { pass: true, failureMessage };
  }

  return { pass: testResult.success, failureMessage };
}


// Evaluate the trace(s) produced by running `fn`.
export async function runCallbackMetrics(
  fn: ToPassCallback,
  metrics: AnyMetric[] = [],
  golden?: Golden,
): Promise<MetricsOutcome> {
  assertSingleTurn(metrics);

  const { traces } = await collectTracesFrom(fn);
  if (traces.length === 0) {
    throw new DeepEvalError(
      "expect(callback).toPass() ran the callback but no trace was produced. " +
        "Instrument the code under test (an `observe`d function, or a framework " +
        "integration) so there is something to evaluate.",
    );
  }

  const primary = primaryTraceFor(traces);
  if (traces.length > 1) {
    console.warn(
      `\n⚠ The callback produced ${traces.length} traces. Trace-level metrics ` +
        `judge the one carrying the turn's output; span-level metrics from the ` +
        `others are still evaluated.`,
    );
  }

  if (primary) {
    if (golden) applyGoldenToTrace(primary, golden);
    if (metrics.length > 0) {
      primary.metrics = [
        ...(primary.metrics ?? []),
        ...(metrics as BaseMetric[]),
      ];
      // Without a golden the trace supplies the test case, so a trace with no I/O
      // has nothing to judge. Say so instead of silently scoring nothing.
      const needsInput = !metrics.some((m) => m.requiresTrace);
      if (!golden && needsInput && primary.input == null) {
        throw new DeepEvalError(
          "expect(callback).toPass([...]) has trace-level metrics but the trace " +
            "has no input. Pass `{ golden }`, call `updateCurrentTrace({ input })`, " +
            "or use a trace-based metric.",
        );
      }
    }
  }

  const allMetrics: MetricData[] = [];
  let turnCase: EvaluatedCase | undefined;

  for (const trace of traces) {
    // Span metrics run for every trace — each is a real component of the turn —
    // but only the reported trace is judged against the golden.
    const cases = await evaluateTrace(trace, {
      errorConfig: runErrorConfig(),
      golden: trace === primary ? golden : undefined,
    });
    for (const c of cases) {
      if (c.isTraceScope && trace === primary) {
        turnCase = c;
        continue; // recorded below, with the trace attached
      }
      globalResultCollector.record({ ...c, displayOnly: true });
    }
    allMetrics.push(...cases.flatMap((c) => c.metricsData));
  }

  // Exactly one posted case per turn, carrying the trace
  if (primary) {
    const { confidentApiKey: _omit, ...traceApi } =
      traceManager.createTraceApi(primary);
    globalResultCollector.record({
      testCase: turnCase?.testCase ?? turnTestCase(primary, golden),
      metricsData: turnCase?.metricsData ?? [],
      runDuration: 0,
      trace: traceApi,
    });
  }
  return {
    pass: aggregateSuccess(allMetrics),
    failureMessage: buildFailureMessage(allMetrics),
  };
}

/** Options for the callback form of `toPass`. */
export interface ToPassOptions {
  golden?: Golden;
}

/**
 * Single entry point behind `expect(target).toPass(metrics)`. A callback is run
 * and the traces it produces are evaluated; a test case is evaluated directly.
 */
export async function runMetrics(
  target: ToPassTarget,
  metrics: AnyMetric[] = [],
  options: ToPassOptions = {},
): Promise<MetricsOutcome> {
  if (typeof target === "function") {
    return runCallbackMetrics(target as ToPassCallback, metrics, options.golden);
  }
  // `expect(myAgent(input))` instead of `expect(() => myAgent(input))`: the call
  // already started, so its traces are outside the window we capture.
  if (typeof (target as { then?: unknown })?.then === "function") {
    throw new DeepEvalError(
      "expect(...).toPass() received a promise. Pass the call as a callback — " +
        "`expect(() => myAgent(input))` — so it runs inside the assertion and " +
        "its trace can be captured.",
    );
  }
  // A `Golden` receiver is not a supported shape. `ToPassTarget` excludes it, but
  // vitest types `toPass` on `Assertion<T>` for any T, so this can only be caught
  // here — and falling through would hand a Golden to the test-case path.
  if (target instanceof Golden) {
    throw new DeepEvalError(
      "expect(golden).toPass() is not supported. Run the code inside the " +
        "assertion instead: `expect(() => myAgent(golden.input)).toPass(metrics, " +
        "{ golden })`, so the trace being judged is exactly the one your call " +
        "produced.",
    );
  }
  return runTestCaseMetrics(target, metrics);
}
