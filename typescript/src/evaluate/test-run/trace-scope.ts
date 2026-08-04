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
