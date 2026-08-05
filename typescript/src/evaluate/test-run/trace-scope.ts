import { traceManager } from "../../tracing";
import type { Trace } from "../../tracing/tracing";

interface TraceCaptureStore {
  traces: Trace[];
  capturing: boolean;
  unsubscribe?: () => void;
  endEvaluation?: () => void;
}

const STORE_KEY = "__deepeval_trace_capture__";

function store(): TraceCaptureStore {
  const g = globalThis as Record<string, unknown>;
  if (!g[STORE_KEY]) {
    g[STORE_KEY] = { traces: [], capturing: false } as TraceCaptureStore;
  }
  return g[STORE_KEY] as TraceCaptureStore;
}

export function beginTraceCapture(): void {
  const s = store();
  s.traces = [];
  s.capturing = true;
  s.unsubscribe?.();
  s.unsubscribe = traceManager.addTraceCaptureSink((trace: Trace) => {
    store().traces.push(trace);
  });
  // Integrations consult this to materialise spans in-process for local eval.
  s.endEvaluation?.();
  s.endEvaluation = traceManager.beginEvaluation();
}

export function endTraceCapture(): void {
  const s = store();
  s.capturing = false;
  s.traces = [];
  s.unsubscribe?.();
  s.unsubscribe = undefined;
  s.endEvaluation?.();
  s.endEvaluation = undefined;
  traceManager.clearTraces();
}

export function isCapturingTraces(): boolean {
  return store().capturing;
}

export function getCapturedTraces(): Trace[] {
  return store().traces;
}

export function getLatestCapturedTrace(): Trace | undefined {
  const { traces } = store();
  return traces[traces.length - 1];
}

// Run `fn` and return the traces it produced.
export async function collectTracesFrom<T>(
  fn: () => T | Promise<T>,
): Promise<{ result: T; traces: Trace[] }> {
  const ownsScope = !isCapturingTraces();
  if (ownsScope) beginTraceCapture();
  try {
    const before = getCapturedTraces().length;
    const result = await fn();
    await traceManager.awaitSettled();
    // Slice rather than take the whole store: an outer scope may hold traces from
    // earlier assertions in the same test.
    return { result, traces: getCapturedTraces().slice(before) };
  } finally {
    // The returned Trace objects stay valid — this only stops capturing.
    if (ownsScope) endTraceCapture();
  }
}
