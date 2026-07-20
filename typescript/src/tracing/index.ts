export type { BaseApiSpan, TraceApi } from "./api";

export {
  updateCurrentSpan,
  updateCurrentTrace,
  updateRetrieverSpan,
  updateLlmSpan,
  SpanType,
  observe,
  traceManager,
  getCurrentSpan,
  getCurrentTrace,
  type Trace,
  type BaseSpan,
} from "./tracing";

export { setTracingContext } from "./trace-context";

export { evaluateThread, evaluateTrace, evaluateSpan } from "./offline-evals";
