import { LLMTestCase } from "../test-case";
import { asTestCaseString, asToolCalls } from "../test-case/utils";
import { Golden } from "../dataset/golden";
import {
  BaseSpan,
  Trace,
  TraceSpanStatus,
  traceManager,
} from "../tracing/tracing";
import { MetricData, EvaluatedCase, TestResult } from "./types";
import { ErrorConfig, DEFAULT_ERROR_CONFIG } from "./configs";
import { runMetric } from "./evaluate";

/** Stringify a span's input/output the way the metrics expect (objects → JSON). */
const asString = asTestCaseString;

// Build the trace-level test case from the golden, filling gaps from the trace.
function goldenToTraceTestCase(golden: Golden, trace: Trace): LLMTestCase {
  return new LLMTestCase({
    input: golden.input,
    actualOutput:
      trace.output != null
        ? asString(trace.output)
        : (golden.actualOutput ?? "None"),
    expectedOutput: trace.expectedOutput ?? golden.expectedOutput,
    context: trace.context ?? golden.context,
    retrievalContext: trace.retrievalContext ?? golden.retrievalContext,
    toolsCalled: asToolCalls(trace.toolsCalled ?? golden.toolsCalled),
    expectedTools: asToolCalls(trace.expectedTools ?? golden.expectedTools),
    // Carry the dataset the golden came from, so the posted run links back to it.
    _datasetAlias: golden._datasetAlias,
    _datasetId: golden._datasetId,
    _datasetRank: golden._datasetRank,
  });
}

/** Build a single-turn LLMTestCase from a span/trace scope's eval fields. */
function scopeToTestCase(scope: BaseSpan | Trace): LLMTestCase {
  return new LLMTestCase({
    input: asString(scope.input),
    actualOutput: asString(scope.output),
    expectedOutput: scope.expectedOutput,
    context: scope.context,
    retrievalContext: scope.retrievalContext,
    toolsCalled: asToolCalls(scope.toolsCalled),
    expectedTools: asToolCalls(scope.expectedTools),
  });
}

export function turnTestCase(trace: Trace, golden?: Golden): LLMTestCase {
  return golden ? goldenToTraceTestCase(golden, trace) : scopeToTestCase(trace);
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

export function isDuplicateOfCase(
  result: TestResult,
  mainResult?: TestResult,
): boolean {
  if (!mainResult) return false;
  if (result.input !== mainResult.input) return false;
  if (result.actualOutput !== mainResult.actualOutput) return false;

  const a = result.metricsData ?? [];
  const b = mainResult.metricsData ?? [];
  if (a.length !== b.length) return false;
  return a.every((m, i) => {
    const other = b[i];
    return (
      other !== undefined &&
      m.name === other.name &&
      m.score === other.score &&
      m.success === other.success
    );
  });
}


export function primaryTraceFor(traces: Trace[]): Trace | undefined {
  for (let i = traces.length - 1; i >= 0; i--) {
    const candidate = traces[i];
    const output = candidate.output ?? candidate.rootSpans?.[0]?.output;
    if (output != null) return candidate;
  }
  return traces[traces.length - 1];
}

export interface TraceEvalOptions {
  errorConfig?: ErrorConfig;
  /** Called after each metric measures (for progress bars). */
  onMetric?: () => void;
  golden?: Golden;
}

/** Number of metrics that `evaluateTrace` will actually run on this trace. */
export function countTraceMetrics(trace: Trace, golden?: Golden): number {
  let count = 0;
  for (const span of allSpans(trace.rootSpans)) {
    const metrics = span.metrics ?? [];
    if (metrics.length === 0) continue;
    const requiresTrace = metrics.some((m) => m.requiresTrace);
    if (span.input == null && !requiresTrace) continue;
    count += metrics.length;
  }
  const traceMetrics = trace.metrics ?? [];
  if (traceMetrics.length > 0) {
    const requiresTrace = traceMetrics.some((m) => m.requiresTrace);
    // A golden supplies the trace-level input, so the scope is always runnable.
    if (golden || trace.input != null || requiresTrace) {
      count += traceMetrics.length;
    }
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
  const scopes: Array<{
    scope: BaseSpan | Trace;
    node: BaseSpan;
    isTrace: boolean;
  }> = [];
  for (const span of allSpans(trace.rootSpans)) {
    scopes.push({ scope: span, node: span, isTrace: false });
  }
  if (trace.rootSpans[0]) {
    scopes.push({ scope: trace, node: trace.rootSpans[0], isTrace: true });
  }

  for (const { scope, node, isTrace } of scopes) {
    const metrics = scope.metrics ?? [];
    if (metrics.length === 0) continue;

    const requiresTrace = metrics.some((m) => m.requiresTrace);
    const golden = isTrace ? options.golden : undefined;
    if (scope.input == null && !requiresTrace && !golden) {
      if (!isTrace) {
        const span = scope as BaseSpan;
        span.status = TraceSpanStatus.ERRORED;
        span.error =
          "Span has metrics but no LLMTestCase. Are you sure you called " +
          "`updateCurrentSpan()`?";
      }
      // No bar advance: `countTraceMetrics` excludes these metrics from the total.
      continue;
    }

    const testCase = golden
      ? goldenToTraceTestCase(golden, trace)
      : scopeToTestCase(scope);
    if (scope.name) testCase.name = scope.name;
    if (requiresTrace) {
      testCase._traceDict = traceManager.createNestedSpansDict(node);
    }

    const metricsData: MetricData[] = [];
    for (const metric of metrics) {
      metricsData.push(
        await runMetric(
          metric,
          testCase,
          errorCfg,
          options.onMetric ?? (() => {}),
        ),
      );
    }
    scope.metricsData = metricsData; // also attach to the span/trace
    cases.push({
      testCase,
      metricsData,
      runDuration: 0,
      isTraceScope: isTrace,
    });
  }
  return cases;
}
