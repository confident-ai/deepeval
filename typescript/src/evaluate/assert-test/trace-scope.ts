import { traceManager } from "../../tracing";
import type { Trace } from "../../tracing/tracing";

interface TraceCaptureStore {
  traces: Trace[];
  capturing: boolean;
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
  traceManager.setTraceCaptureSink((trace: Trace) => {
    store().traces.push(trace);
  });
}

export function endTraceCapture(): void {
  const s = store();
  s.capturing = false;
  s.traces = [];
  traceManager.setTraceCaptureSink(undefined);
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
