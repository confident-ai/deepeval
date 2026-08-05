import {
  Trace,
  getCurrentTrace,
  setCurrentTrace,
  traceManager,
} from "./tracing";

// Where an OTel-based integration should send a span.
export type SpanRoute = "rest" | "otlp";

/** Marker attribute the interceptor stamps so later processors see the decision. */
export const ROUTE_TO_REST_ATTRIBUTE = "confident.internal.route_to_rest";

export interface RestRoutingOptions {
  isTestMode?: boolean;
}

/** Tag a trace as opened by an integration rather than by the caller. */
export function markTraceOtelImplicit(trace: Trace): void {
  trace._isOtelImplicit = true;
}

export function isTraceOtelImplicit(trace: Trace): boolean {
  return trace._isOtelImplicit === true;
}

/**
 * Decide the route for the span being started, first match winning:
 *
 * | Signal                                              | Route |
 * | --------------------------------------------------- | ----- |
 * | A real deepeval trace context (`observe`, `trace`)  | rest  |
 * | A local eval pipeline is active (`isEvaluating`)    | rest  |
 * | `isTestMode`                                        | rest  |
 * | none of the above                                   | otlp  |
 *
 * A trace this integration opened implicitly does NOT count as a real context:
 * the caller never asked for local behaviour, and promoting it would silently
 * change where their traces go. It still routes to REST while evaluating, because
 * the eval pipeline reads what the in-process path produces — over OTLP the trace
 * would vanish from the run.
 */
export function resolveSpanRoute(options: RestRoutingOptions = {}): SpanRoute {
  const currentTrace = getCurrentTrace();
  if (currentTrace && !isTraceOtelImplicit(currentTrace)) return "rest";
  if (traceManager.isEvaluating) return "rest";
  if (options.isTestMode) return "rest";
  return "otlp";
}

export function shouldRouteToRest(options: RestRoutingOptions = {}): boolean {
  return resolveSpanRoute(options) === "rest";
}

export function endOtelImplicitTrace(traceUuid: string): void {
  const trace = traceManager.getTraceByUuid(traceUuid);
  if (!trace || !isTraceOtelImplicit(trace)) return;

  traceManager.endTrace(traceUuid);

  // Leaving an ended trace in the async context would make the next root span
  // attach to a dead trace, whose spans are then silently dropped.
  if (getCurrentTrace()?.uuid === traceUuid) setCurrentTrace(null);
}

export function resolveTraceForOtelSpan(
  allowImplicit: boolean,
): Trace | undefined {
  const existing = getCurrentTrace();
  if (existing && traceManager.getTraceByUuid(existing.uuid)) return existing;
  if (!allowImplicit) return undefined;

  const implicit = traceManager.startNewTrace();
  markTraceOtelImplicit(implicit);
  // OTel calls `onStart` synchronously in the caller's async context, so entering
  // the context here makes the trace visible to the spans that follow.
  setCurrentTrace(implicit);
  return implicit;
}
