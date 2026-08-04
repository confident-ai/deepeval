export { BaseApiSpan, TraceApi } from "./api";

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

export {
  resolveSpanRoute,
  shouldRouteToRest,
  resolveTraceForOtelSpan,
  isTraceOtelImplicit,
  markTraceOtelImplicit,
  ROUTE_TO_REST_ATTRIBUTE,
  type SpanRoute,
  type RestRoutingOptions,
} from "./otel-routing";

export {
  nextSpan,
  nextAgentSpan,
  nextLlmSpan,
  nextToolSpan,
  nextRetrieverSpan,
  popPendingFor,
  applyPendingToSpan,
  type PendingSpanParams,
  type PendingAgentSpanParams,
  type PendingLlmSpanParams,
  type PendingToolSpanParams,
  type PendingRetrieverSpanParams,
  type PendingPayload,
} from "./pending-context";

export { evaluateThread, evaluateTrace, evaluateSpan } from "./offline-evals";
