import {
  Trace,
  getCurrentTrace,
  setCurrentTrace,
  traceManager,
} from "./tracing";

/**
 * Where an OTel-based integration should send a span.
 *
 * `"rest"` means "materialise the trace in-process": build deepeval spans on the
 * trace manager, which either hands them to the local eval pipeline or posts them
 * over REST. `"otlp"` means "leave it to the OTLP exporter".
 *
 * Port of Python's `ContextAwareSpanProcessor._should_route_to_rest`. The span
 * interceptor never makes this decision — it only produces the `confident.*`
 * attributes both transports read.
 */
export type SpanRoute = "rest" | "otlp";

/** Marker attribute the interceptor stamps so later processors see the decision. */
export const ROUTE_TO_REST_ATTRIBUTE = "confident.internal.route_to_rest";

export interface RestRoutingOptions {
  /** Explicit override, e.g. a schema-asserting test harness. */
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

/** Convenience wrapper around {@link resolveSpanRoute}. */
export function shouldRouteToRest(options: RestRoutingOptions = {}): boolean {
  return resolveSpanRoute(options) === "rest";
}

/**
 * The trace an OTel integration should attach spans to, opening an implicit one
 * for a bare caller.
 *
 * `allowImplicit` should be true only for the root span of a trace — a child
 * span arriving with no context is an ordering bug, and inventing a trace for it
 * would split one logical trace across several.
 */
/**
 * End a trace this integration opened implicitly, once its root span ends.
 *
 * Nothing else will: a caller-owned trace is closed by whatever opened it
 * (`observe`, `setTracingContext`), but an implicit one has no owner, so without
 * this it stays active forever — never captured, never posted, and invisible to
 * local evaluation. A no-op for caller-owned traces.
 */
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
  // Only usable if it is still open. An implicit trace closed inside a nested
  // async context stays visible out here — clearing an AsyncLocalStorage store in
  // a child context does not propagate back to the parent — and adding a span to
  // an ended trace is rejected outright.
  if (existing && traceManager.getTraceByUuid(existing.uuid)) return existing;
  if (!allowImplicit) return undefined;

  const implicit = traceManager.startNewTrace();
  markTraceOtelImplicit(implicit);
  // OTel calls `onStart` synchronously in the caller's async context, so entering
  // the context here makes the trace visible to the spans that follow.
  setCurrentTrace(implicit);
  return implicit;
}
