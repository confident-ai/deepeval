import { LLMTestCase } from "../test-case";
import { BaseSpan, Trace, traceManager } from "../tracing/tracing";
import { MetricData, EvaluatedCase } from "./types";
import { ErrorConfig, DEFAULT_ERROR_CONFIG } from "./configs";
import { runMetric } from "./evaluate";

/** Stringify a span's input/output the way the metrics expect (objects → JSON). */
function asString(v: unknown): string {
  if (v == null) return "None";
  return typeof v === "string" ? v : JSON.stringify(v);
}

/** Build a single-turn LLMTestCase from a span/trace scope's eval fields. */
function scopeToTestCase(scope: BaseSpan | Trace): LLMTestCase {
  return new LLMTestCase({
    input: asString(scope.input),
    actualOutput: asString(scope.output),
    expectedOutput: scope.expectedOutput,
    context: scope.context,
    retrievalContext: scope.retrievalContext,
    toolsCalled: scope.toolsCalled,
    expectedTools: scope.expectedTools,
  });
}

/** Depth-first list of every span under the given roots. */
function allSpans(roots: BaseSpan[]): BaseSpan[] {
  const out: BaseSpan[] = [];
  const walk = (s: BaseSpan) => {
    out.push(s);
    (s.children ?? []).forEach(walk);
  };
  roots.forEach(walk);
  return out;
}

export interface TraceEvalOptions {
  errorConfig?: ErrorConfig;
  /** Called after each metric measures (for progress bars). */
  onMetric?: () => void;
}

/** Number of metrics that `evaluateTrace` will actually run on this trace. */
export function countTraceMetrics(trace: Trace): number {
  let count = 0;
  const scopes: Array<BaseSpan | Trace> = [...allSpans(trace.rootSpans), trace];
  for (const scope of scopes) {
    const metrics = scope.metrics ?? [];
    if (metrics.length === 0) continue;
    const requiresTrace = metrics.some((m) => m.requiresTrace);
    if (scope.input == null && !requiresTrace) continue;
    count += metrics.length;
  }
  return count;
}

/**
 * Run locally-attached metrics over a completed trace (mirrors Python's agentic
 * executor). For the trace and each span carrying `metrics`, build an
 * `LLMTestCase` from its I/O, attach the serialized trace (`_traceDict`) when any
 * metric `requiresTrace`, then measure. Returns one `TestResult` per evaluated
 * scope, labelled by span/trace name.
 */
export async function evaluateTrace(
  trace: Trace,
  options: TraceEvalOptions = {},
): Promise<EvaluatedCase[]> {
  const errorCfg: Required<ErrorConfig> = {
    ...DEFAULT_ERROR_CONFIG,
    ...options.errorConfig,
  };
  const cases: EvaluatedCase[] = [];

  // Every span scope, then the trace scope (whose trace dict is the full tree).
  const scopes: Array<{ scope: BaseSpan | Trace; node: BaseSpan }> = [];
  for (const span of allSpans(trace.rootSpans)) {
    scopes.push({ scope: span, node: span });
  }
  if (trace.rootSpans[0]) {
    scopes.push({ scope: trace, node: trace.rootSpans[0] });
  }

  for (const { scope, node } of scopes) {
    const metrics = scope.metrics ?? [];
    if (metrics.length === 0) continue;

    const requiresTrace = metrics.some((m) => m.requiresTrace);
    if (scope.input == null && !requiresTrace) {
      // Metrics attached but no test-case data (forgot updateCurrentSpan).
      continue;
    }

    const testCase = scopeToTestCase(scope);
    if (scope.name) testCase.name = scope.name;
    if (requiresTrace) {
      testCase._traceDict = traceManager.createNestedSpansDict(node);
    }

    const metricsData: MetricData[] = [];
    for (const metric of metrics) {
      metricsData.push(
        await runMetric(metric, testCase, errorCfg, options.onMetric ?? (() => {})),
      );
    }
    scope.metricsData = metricsData; // also attach to the span/trace
    cases.push({ testCase, metricsData, runDuration: 0 });
  }
  return cases;
}
