export { BaseApiSpan, TraceApi } from "@/tracing/api";

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
} from "@/tracing/tracing";

export { setTracingContext } from "@/tracing/trace-context";

export {
  resolveSpanRoute,
  shouldRouteToRest,
  resolveTraceForOtelSpan,
  isTraceOtelImplicit,
  markTraceOtelImplicit,
  ROUTE_TO_REST_ATTRIBUTE,
  type SpanRoute,
  type RestRoutingOptions,
} from "@/tracing/otel-routing";

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
} from "@/tracing/pending-context";

export { flushTraces, traceFlushEnabled } from "@/tracing/flush";

export {
  evaluateThread,
  evaluateTrace,
  evaluateSpan,
} from "@/tracing/offline-evals";
