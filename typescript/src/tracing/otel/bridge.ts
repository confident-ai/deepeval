/**
 * Collect confident-trace's spans as deepeval spans.
 *
 * The bridge claims every span of a trace deepeval consumes: one inside
 * `observe(...)`, during an evaluation, or under a trace-shape test, and
 * converts each as it ends. While an evaluation or a trace-shape test consumes
 * the trace, it also marks the spans so `RemoteSpanExporter` doesn't send them;
 * otherwise confident-trace exports them too.
 */

import type {
  Context,
  Span as OtelSpan,
  SpanContext,
} from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
  SpanExporter,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { type ExportResult, ExportResultCode } from "@opentelemetry/core";

import { ToolCall } from "@/test-case";
import {
  type AgentSpan,
  type BaseSpan,
  type Trace,
  getCurrentSpan,
  getCurrentTrace,
  isDescendant,
  SpanType,
  traceManager,
  TraceSpanStatus,
} from "@/tracing/tracing";
import {
  applyPendingToSpan,
  popPendingFor,
  popPendingSpanType,
  popPendingTrace,
} from "@/tracing/pending-context";
import { ConfidentAttr } from "@/tracing/attributes";
import {
  convertSpan,
  fillUnset,
  hrTimeToDate,
  isAiSpan,
  resolveSpanType,
} from "@/tracing/otel/converter";
import { latestUserText } from "@/tracing/otel/messages";

export const LOCAL_SPAN_ATTRIBUTE = "deepeval.local";
/** The scope of `observe` spans, which take content from the function they wrap. */
export const OBSERVE_SCOPE = "deepeval";

// The spans whose content describes the trace's own input and output.
const CONTENT_TYPES = new Set<string>([SpanType.AGENT, SpanType.LLM]);

interface Claim {
  traceUuid: string;
  /**
   * What deepeval code sees and updates until the span ends; every field
   * set on it overrides what the span recorded.
   */
  placeholder: BaseSpan;
  initial: Record<string, unknown>;
  /** The type an integration staged for the span, over the one it records. */
  spanType?: SpanType;
}

interface TraceState {
  /** True when the bridge opened the deepeval trace and so must end it. */
  owned: boolean;
  /** True when deepeval consumes the trace, so its spans aren't exported. */
  local: boolean;
  openSpans: number;
  /** Placeholders of the spans still open. */
  open: Set<string>;
  spans: Map<string, BaseSpan>;
  /** Spans that ended before their parent, by that parent's uuid. */
  orphans: Map<string, BaseSpan[]>;
  /** A dropped span's uuid, mapped to the parent its children attach to. */
  lifted: Map<string, string | undefined>;
  traceFields: Map<string, Partial<Trace>>;
  conversations: Set<string>;
  /** Trace fields the bridge derived, with the span time that chose each. */
  derived: Map<string, { time: number; value: unknown }>;
}

function key(context: SpanContext): string {
  return `${context.traceId}/${context.spanId}`;
}

export class DeepEvalSpanBridge implements SpanProcessor {
  // Every open span seen, so children inherit their parent's decision;
  // `null` marks a span left to confident-trace.
  private readonly claims = new Map<string, Claim | null>();
  private readonly traces = new Map<string, TraceState>();
  private closed = false;

  onStart(span: Span, _parentContext: Context): void {
    if (this.closed) return;
    let claim: Claim | null = null;
    try {
      claim = this.claim(span);
    } catch {
      claim = null;
    }
    this.claims.set(key(span.spanContext()), claim);
    if (claim && this.traces.get(claim.traceUuid)?.local)
      span.setAttribute(LOCAL_SPAN_ATTRIBUTE, true);
  }

  onEnd(span: ReadableSpan): void {
    const spanKey = key(span.spanContext());
    const claim = this.claims.get(spanKey);
    this.claims.delete(spanKey);
    if (!claim) return;
    const state = this.traces.get(claim.traceUuid);
    if (!state) return;
    try {
      this.collect(span, claim, state);
    } catch {
      // A span that can't be converted is left out; the trace goes on.
    }
    traceManager.removeSpan(claim.placeholder.uuid);
    state.open.delete(claim.placeholder.uuid);
    if (--state.openSpans > 0) return;
    try {
      if (state.owned) {
        this.traces.delete(claim.traceUuid);
        endOwnedTrace(claim.traceUuid, state);
      } else {
        endFinishedTrace(claim.traceUuid);
        // Later spans may still join the deepeval trace, and must see what
        // these derived for it; `open` drops the state once the trace ends.
        if (!traceManager.getTraceByUuid(claim.traceUuid))
          this.traces.delete(claim.traceUuid);
      }
    } catch {
      // Ending a trace must never break the application.
    }
  }

  async forceFlush(): Promise<void> {}

  async shutdown(): Promise<void> {
    this.closed = true;
  }

  /** The placeholder standing in for `span` while it is collected. */
  spanFor(span: OtelSpan | undefined): BaseSpan | undefined {
    if (!span) return undefined;
    return this.claims.get(key(span.spanContext()))?.placeholder;
  }

  /** Whether `span` is open and confident-trace will export it. */
  isExported(span: OtelSpan | undefined): boolean {
    if (!span) return false;
    const spanKey = key(span.spanContext());
    if (!this.claims.has(spanKey)) return false;
    const claim = this.claims.get(spanKey);
    return !claim || !this.traces.get(claim.traceUuid)?.local;
  }

  private claim(span: Span): Claim | null {
    const parent = span.parentSpanContext;
    if (parent && !parent.isRemote) {
      const parentClaim = this.claims.get(key(parent));
      if (parentClaim === null) return null;
      if (parentClaim) {
        let parentUuid = parentClaim.placeholder.uuid;
        // deepeval code opened inside the parent, such as an `observe`
        // function, encloses this span.
        const enclosing = getCurrentSpan();
        if (enclosing && isDescendant(enclosing, parentUuid))
          parentUuid = enclosing.uuid;
        return this.open(span, parentClaim.traceUuid, parentUuid);
      }
    }
    // A root span: the trace it starts takes what an integration staged.
    const staged = popPendingTrace();
    const scope = deepevalScope();
    let claim: Claim;
    if (scope) claim = this.open(span, scope.traceUuid, scope.spanUuid);
    else if (traceManager.isCapturingTraces)
      claim = this.open(span, span.spanContext().traceId, undefined, true);
    else {
      // On an exported root, ahead of the defaults confident-trace adds.
      span.setAttributes(traceFieldAttributes(staged));
      return null;
    }
    // Staged values are defaults: what deepeval code set on the trace wins.
    const trace = traceManager.getTraceByUuid(claim.traceUuid);
    if (trace) fillUnset(trace, staged);
    return claim;
  }

  private open(
    span: Span,
    traceUuid: string,
    parentUuid: string | undefined,
    owned = false,
  ): Claim {
    const staged = popPendingSpanType();
    const spanType =
      (staged as SpanType | undefined) ?? resolveSpanType(span, true);
    const placeholder = {
      uuid: span.spanContext().spanId,
      traceUuid,
      parentUuid,
      status: TraceSpanStatus.IN_PROGRESS,
      startTime: hrTimeToDate(span.startTime),
      name: span.name,
      type: spanType,
      children: [],
      ...(spanType === SpanType.AGENT
        ? { availableTools: [], agentHandoffs: [] }
        : {}),
      ...(spanType === SpanType.LLM ? { model: "" } : {}),
    } as BaseSpan;
    const initial = { ...placeholder };
    // `next*Span` staging, and what an integration staged under it.
    applyPendingToSpan(placeholder, popPendingFor(spanType));
    let state = this.traces.get(traceUuid);
    if (!state) {
      this.dropEnded();
      // In an evaluation this also captures the trace when it ends.
      if (owned) traceManager.startNewTrace(traceUuid);
      state = newTraceState(owned, traceManager.isCapturingTraces);
      this.traces.set(traceUuid, state);
    }
    state.openSpans++;
    state.open.add(placeholder.uuid);
    // An open span keeps its deepeval trace open, and parents deepeval spans.
    traceManager.addSpan(placeholder);
    return {
      traceUuid,
      placeholder,
      initial,
      ...(staged ? { spanType } : {}),
    };
  }

  /** Forget the deepeval traces that ended with none of their spans open. */
  private dropEnded(): void {
    for (const [traceUuid, state] of this.traces)
      if (state.openSpans === 0 && !traceManager.getTraceByUuid(traceUuid))
        this.traces.delete(traceUuid);
  }

  private collect(span: ReadableSpan, claim: Claim, state: TraceState): void {
    const trace = traceManager.getTraceByUuid(claim.traceUuid);
    // deepeval already ended the trace while this span was open.
    if (!trace) return;
    const { placeholder } = claim;
    const overrides: Record<string, unknown> = {};
    for (const [field, value] of Object.entries(placeholder))
      if (value !== claim.initial[field] && field !== "children")
        overrides[field] = value;
    addCollected(span, trace, state, {
      uuid: placeholder.uuid,
      parentUuid: placeholder.parentUuid,
      spanType: claim.spanType,
      overrides,
      // deepeval spans opened inside this one.
      children: placeholder.children,
    });
  }
}

// Traces and spans deepeval code dropped, by OTel id, until they export.
// Bounded, since a dropped trace may never end.
const MAX_DROPPED = 10_000;
const droppedTraces = new Set<string>();
const droppedSpans = new Set<string>();

/** Keep `span`, or its whole trace, out of the export. */
export function dropFromExport(
  span: OtelSpan | undefined,
  wholeTrace: boolean,
): void {
  if (!span) return;
  const context = span.spanContext();
  remember(
    wholeTrace ? droppedTraces : droppedSpans,
    wholeTrace ? context.traceId : key(context),
  );
}

function remember(dropped: Set<string>, id: string): void {
  dropped.add(id);
  if (dropped.size > MAX_DROPPED)
    dropped.delete(dropped.values().next().value!);
}

/** The sampling decision for a whole trace, the same for each of its spans. */
function sampled(traceId: string): boolean {
  const rate = traceManager.exportSamplingRate;
  return rate >= 1 || parseInt(traceId.slice(-8), 16) % 10_000 < rate * 10_000;
}

function exportable(span: ReadableSpan): boolean {
  const context = span.spanContext();
  if (span.attributes[LOCAL_SPAN_ATTRIBUTE] || !sampled(context.traceId))
    return false;
  if (droppedTraces.has(context.traceId)) return false;
  return !droppedSpans.delete(key(context));
}

/** `span` with deepeval's environment, if it starts a trace that set none. */
function withEnvironment(span: ReadableSpan): ReadableSpan {
  const parent = span.parentSpanContext;
  if (
    (parent && !parent.isRemote) ||
    span.attributes[ConfidentAttr.TRACE_ENVIRONMENT] !== undefined
  )
    return span;
  return Object.create(span, {
    attributes: {
      value: {
        ...span.attributes,
        [ConfidentAttr.TRACE_ENVIRONMENT]: traceManager.exportEnvironment,
      },
    },
  }) as ReadableSpan;
}

/** Exports the spans that aren't local, dropped or sampled out. */
export class RemoteSpanExporter implements SpanExporter {
  constructor(private readonly exporter: SpanExporter) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    const remote = spans.filter(exportable).map(withEnvironment);
    if (remote.length === 0)
      return resultCallback({ code: ExportResultCode.SUCCESS });
    this.exporter.export(remote, resultCallback);
  }

  forceFlush(): Promise<void> {
    return this.exporter.forceFlush?.() ?? Promise.resolve();
  }

  shutdown(): Promise<void> {
    return this.exporter.shutdown();
  }
}

/** The live deepeval trace, and span, that a span starting now joins. */
function deepevalScope(): { traceUuid: string; spanUuid?: string } | undefined {
  const span = getCurrentSpan();
  if (span && traceManager.getTraceByUuid(span.traceUuid))
    return { traceUuid: span.traceUuid, spanUuid: span.uuid };
  const trace = getCurrentTrace();
  if (trace && traceManager.getTraceByUuid(trace.uuid))
    return { traceUuid: trace.uuid };
  return undefined;
}

function newTraceState(owned: boolean, local = true): TraceState {
  return {
    owned,
    local,
    openSpans: 0,
    open: new Set(),
    spans: new Map(),
    orphans: new Map(),
    lifted: new Map(),
    traceFields: new Map(),
    conversations: new Set(),
    derived: new Map(),
  };
}

interface Collected {
  uuid: string;
  parentUuid: string | undefined;
  spanType?: SpanType;
  /** Fields set while the span was open, which win over what it recorded. */
  overrides?: Record<string, unknown>;
  children?: BaseSpan[];
}

/** Convert an ended span into `trace`, where its parent is or will be. */
function addCollected(
  span: ReadableSpan,
  trace: Trace,
  state: TraceState,
  collected: Collected,
): BaseSpan | undefined {
  const parentUuid = liftedParent(state, collected.parentUuid);
  const observed = span.instrumentationScope?.name === OBSERVE_SCOPE;
  // As confident-trace exports them, non-AI spans stay only as ancestors.
  if (
    !collected.spanType &&
    !isAiSpan(span) &&
    !state.orphans.has(collected.uuid) &&
    !collected.children?.length
  ) {
    state.lifted.set(collected.uuid, parentUuid);
    return undefined;
  }
  const converted = convertSpan(span, collected.spanType);
  const converts = converted.span;
  converts.traceUuid = trace.uuid;
  converts.parentUuid = parentUuid;
  Object.assign(converts, collected.overrides ?? {});
  // deepeval spans opened inside this one, then spans that ended before it.
  converts.children = [
    ...(collected.children ?? []),
    ...(state.orphans.get(collected.uuid) ?? []),
  ];
  for (const child of converts.children) child.parentUuid = converts.uuid;
  state.orphans.delete(collected.uuid);
  attach(trace, state, converts);
  if (converts.type === SpanType.AGENT) fillAgentContent(converts as AgentSpan);
  if (!converts.toolsCalled?.length) {
    // Like `observe`, a span lists the tools run under it; an integration's
    // agent also lists those its intermediate steps ran.
    const calls = toolCallsUnder(
      converts,
      converts.type === SpanType.AGENT && !observed,
    );
    if (calls.length > 0) converts.toolsCalled = calls;
  }
  if (converted.conversationId)
    state.conversations.add(converted.conversationId);
  if (state.owned) {
    state.traceFields.set(converts.uuid, converted.traceFields);
    return converts;
  }
  fillUnsetTrace(trace, converted.traceFields, !converts.parentUuid);
  if (!trace.threadId && converted.conversationId)
    trace.threadId = converted.conversationId;
  // `observe` gives its trace the root function's input and output.
  if (!observed) deriveContent(trace, state, converts);
  return converts;
}

/**
 * Build and end a deepeval trace from the ended spans of one trace, for a
 * framework that only reports spans once they finish.
 *
 * `trace` fields are set first, so the spans only fill what they leave
 * unset; `configure` sees each converted span.
 */
export function collectFinishedTrace(
  spans: ReadableSpan[],
  fields: Partial<Trace> = {},
  configure: (span: BaseSpan) => void = () => {},
): void {
  if (spans.length === 0) return;
  const traceUuid = spans[0].spanContext().traceId;
  const trace = traceManager.startNewTrace(traceUuid);
  Object.assign(trace, fields);
  const state = newTraceState(true);
  const ids = new Set(spans.map((span) => span.spanContext().spanId));
  for (const id of ids) state.open.add(id);
  // Children end before their parents, which adopt them.
  const byEnd = [...spans].sort(
    (a, b) =>
      hrTimeToDate(a.endTime).getTime() - hrTimeToDate(b.endTime).getTime(),
  );
  for (const span of byEnd) {
    const uuid = span.spanContext().spanId;
    state.open.delete(uuid);
    const parent = span.parentSpanContext?.spanId;
    const converted = addCollected(span, trace, state, {
      uuid,
      parentUuid: parent && ids.has(parent) ? parent : undefined,
    });
    if (converted) configure(converted);
  }
  endOwnedTrace(traceUuid, state);
}

const STAMPED_TRACE_STRINGS: Record<string, string> = {
  name: ConfidentAttr.TRACE_NAME,
  threadId: ConfidentAttr.TRACE_THREAD_ID,
  userId: ConfidentAttr.TRACE_USER_ID,
  testCaseId: ConfidentAttr.TRACE_TEST_CASE_ID,
  turnId: ConfidentAttr.TRACE_TURN_ID,
  metricCollection: ConfidentAttr.TRACE_METRIC_COLLECTION,
  environment: ConfidentAttr.TRACE_ENVIRONMENT,
};

/** Trace fields as the `confident.trace.*` attributes that carry them. */
export function traceFieldAttributes(
  values: Record<string, unknown>,
): Record<string, string | string[]> {
  const attributes: Record<string, string | string[]> = {};
  for (const [field, attribute] of Object.entries(STAMPED_TRACE_STRINGS)) {
    const value = values[field];
    if (typeof value === "string" && value) attributes[attribute] = value;
  }
  if (Array.isArray(values.tags))
    attributes[ConfidentAttr.TRACE_TAGS] = values.tags.map(String);
  if (values.metadata && typeof values.metadata === "object") {
    try {
      attributes[ConfidentAttr.TRACE_METADATA] = JSON.stringify(
        values.metadata,
      );
    } catch {
      // Metadata that can't be serialized isn't sent.
    }
  }
  return attributes;
}

function liftedParent(
  state: TraceState,
  parentUuid: string | undefined,
): string | undefined {
  while (parentUuid && state.lifted.has(parentUuid))
    parentUuid = state.lifted.get(parentUuid);
  return parentUuid;
}

function attach(trace: Trace, state: TraceState, span: BaseSpan): void {
  state.spans.set(span.uuid, span);
  const parentUuid = span.parentUuid;
  if (!parentUuid) {
    trace.rootSpans.push(span);
    return;
  }
  if (state.open.has(parentUuid)) {
    // The parent is still open: it adopts this span when it ends.
    const orphans = state.orphans.get(parentUuid) ?? [];
    orphans.push(span);
    state.orphans.set(parentUuid, orphans);
    return;
  }
  // A collected span, or a deepeval one such as an enclosing `observe`
  // function, open or already finished.
  const parent =
    state.spans.get(parentUuid) ??
    traceManager.getSpanByUuid(parentUuid) ??
    [...depthFirst(trace.rootSpans)].find((other) => other.uuid === parentUuid);
  if (parent) parent.children.push(span);
  else trace.rootSpans.push(span);
}

function fillUnsetTrace(
  trace: Trace,
  fields: Partial<Trace>,
  isRoot: boolean,
): void {
  // Only a trace's root names it.
  fillUnset(trace, isRoot ? fields : { ...fields, name: undefined });
}

function toolCallOf(span: BaseSpan): ToolCall {
  const input = span.input;
  return new ToolCall({
    name: span.name ?? "tool",
    ...((span as { description?: string }).description
      ? { description: (span as { description?: string }).description }
      : {}),
    ...(input && typeof input === "object" && !Array.isArray(input)
      ? { inputParameters: input }
      : input !== undefined
        ? { inputParameters: { input } }
        : {}),
    ...(span.output !== undefined ? { output: span.output } : {}),
  });
}

/**
 * Calls for the tool spans under `span`: its children or, with
 * `descendants`, every tool span below it that no nested agent owns.
 */
function toolCallsUnder(span: BaseSpan, descendants: boolean): ToolCall[] {
  const calls: ToolCall[] = [];
  for (const child of span.children) {
    if (child.type === SpanType.TOOL) calls.push(toolCallOf(child));
    else if (descendants && child.type !== SpanType.AGENT)
      calls.push(...toolCallsUnder(child, true));
  }
  return calls;
}

/** The model and agent spans under `span` that no nested agent owns. */
function contentSpans(span: BaseSpan): BaseSpan[] {
  const found: BaseSpan[] = [];
  for (const child of span.children) {
    if (CONTENT_TYPES.has(child.type)) found.push(child);
    if (child.type !== SpanType.AGENT) found.push(...contentSpans(child));
  }
  return found;
}

/**
 * The earliest input, narrowed to the latest user message, and the
 * latest-ending output among `spans`.
 */
function contentOf(spans: BaseSpan[]): [unknown, unknown] {
  const withInput = spans.filter((span) => span.input !== undefined);
  const withOutput = spans.filter((span) => span.output !== undefined);
  const earliest = withInput.sort(
    (a, b) => a.startTime.getTime() - b.startTime.getTime(),
  )[0];
  const latest = withOutput.sort(
    (a, b) => (b.endTime?.getTime() ?? 0) - (a.endTime?.getTime() ?? 0),
  )[0];
  return [
    earliest ? latestUserText(earliest.input) : undefined,
    latest?.output,
  ];
}

/** An agent that records no content takes it from its model calls and the agents it ran. */
function fillAgentContent(agent: AgentSpan): void {
  if (agent.input !== undefined && agent.output !== undefined) return;
  const [input, output] = contentOf(contentSpans(agent));
  if (agent.input === undefined) agent.input = input;
  if (agent.output === undefined) agent.output = output;
}

function* depthFirst(spans: BaseSpan[]): Generator<BaseSpan> {
  for (const span of spans) {
    yield span;
    yield* depthFirst(span.children);
  }
}

/** Derive what a deepeval trace's spans didn't set, keeping the best candidate. */
function deriveContent(trace: Trace, state: TraceState, span: BaseSpan): void {
  if (span.type === SpanType.TOOL) {
    derive(trace, state, "toolsCalled", Number.MAX_SAFE_INTEGER, [
      ...((state.derived.get("toolsCalled")?.value as ToolCall[]) ?? []),
      toolCallOf(span),
    ]);
    return;
  }
  if (!CONTENT_TYPES.has(span.type)) return;
  if (span.input !== undefined)
    derive(
      trace,
      state,
      "input",
      -span.startTime.getTime(),
      latestUserText(span.input),
    );
  if (span.output !== undefined)
    derive(trace, state, "output", span.endTime?.getTime() ?? 0, span.output);
}

function derive(
  trace: Trace,
  state: TraceState,
  field: "input" | "output" | "toolsCalled",
  time: number,
  value: unknown,
): void {
  const current = state.derived.get(field);
  const replaceable =
    (trace[field] === undefined && !current) ||
    (current && trace[field] === current.value && time >= current.time);
  if (!replaceable) return;
  state.derived.set(field, { time, value });
  (trace as unknown as Record<string, unknown>)[field] = value;
}

/** What the platform derives for a trace its spans didn't fill. */
function traceContent(spans: BaseSpan[], state: TraceState): Partial<Trace> {
  const candidates = spans.filter((span) => CONTENT_TYPES.has(span.type));
  const named =
    candidates.find((span) => span.type === SpanType.AGENT) ?? candidates[0];
  const [input, output] = contentOf(candidates);
  const toolCalls = spans
    .filter((span) => span.type === SpanType.TOOL)
    .map(toolCallOf);
  return {
    name: named?.name,
    input,
    output,
    // Ambiguous when the spans name different conversations.
    threadId:
      state.conversations.size === 1 ? [...state.conversations][0] : undefined,
    toolsCalled: toolCalls.length > 0 ? toolCalls : undefined,
  };
}

function endOwnedTrace(traceUuid: string, state: TraceState): void {
  const trace = traceManager.getTraceByUuid(traceUuid);
  if (!trace) return;
  if (trace.rootSpans.length === 0) {
    // Only non-AI spans ran, so there is nothing to evaluate or show.
    traceManager.discardTrace(traceUuid);
    return;
  }
  // Mirrors the platform: in depth-first order, later spans win. Values from
  // `updateCurrentTrace(...)` still win over all of them.
  const spans = [...depthFirst(trace.rootSpans)];
  const recorded: Partial<Trace> = {};
  for (const span of spans)
    Object.assign(recorded, state.traceFields.get(span.uuid) ?? {});
  fillUnset(trace, recorded);
  fillUnset(trace, traceContent(spans, state));
  if (trace.rootSpans.some((span) => span.status === TraceSpanStatus.ERRORED))
    trace.status = TraceSpanStatus.ERRORED;
  trace.startTime = new Date(
    Math.min(...trace.rootSpans.map((span) => span.startTime.getTime())),
  );
  traceManager.endTrace(traceUuid);
}

/** End a deepeval trace whose own spans all finished before the bridge's. */
function endFinishedTrace(traceUuid: string): void {
  const open = [...traceManager.getActiveSpans().values()].some(
    (span) => span.traceUuid === traceUuid,
  );
  if (!open && traceManager.getTraceByUuid(traceUuid))
    traceManager.endTrace(traceUuid);
}
