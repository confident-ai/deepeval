import { AsyncLocalStorage } from "async_hooks";

import { BaseMetric } from "@/metrics/base-metrics";
import { LLMTestCase, ToolCall } from "@/test-case";
import { Prompt } from "@/prompt";
import type { BaseSpan } from "@/tracing/tracing";
import { ConfidentAttr } from "@/tracing/attributes";

export interface PendingSpanParams {
  input?: any;
  output?: any;
  retrievalContext?: string[];
  context?: string[];
  expectedOutput?: string;
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  metadata?: Record<string, any>;
  name?: string;
  testCase?: LLMTestCase;
  metricCollection?: string;
  metrics?: BaseMetric[];
}

export interface PendingAgentSpanParams extends PendingSpanParams {
  availableTools?: string[];
  agentHandoffs?: string[];
}

export interface PendingLlmSpanParams extends PendingSpanParams {
  model?: string;
  inputTokenCount?: number;
  outputTokenCount?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  prompt?: Prompt;
}

export interface PendingToolSpanParams extends PendingSpanParams {
  description?: string;
}

export interface PendingRetrieverSpanParams extends PendingSpanParams {
  embedder?: string;
  topK?: number;
  chunkSize?: number;
}

export type PendingPayload = Record<string, any>;

/** Which typed slot a span competes for. `undefined` → base slot only. */
export type SlotKind = "agent" | "llm" | "tool" | "retriever";

class PendingSlot {
  payload?: PendingPayload;

  constructor(payload?: PendingPayload) {
    this.payload = payload;
  }
}

type SlotStore = {
  base?: PendingSlot;
  agent?: PendingSlot;
  llm?: PendingSlot;
  tool?: PendingSlot;
  retriever?: PendingSlot;
};

const pendingStore = new AsyncLocalStorage<SlotStore>();

export type AmbientPayloadReader = (
  kind: SlotKind | undefined,
) => PendingPayload | undefined;

let ambientReader: AmbientPayloadReader | undefined;

export function _setAmbientPayloadReader(
  reader: AmbientPayloadReader | undefined,
): void {
  ambientReader = reader;
}

function dropUndefined(params: Record<string, any>): PendingPayload {
  const out: PendingPayload = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function withSlot<T>(
  key: keyof SlotStore,
  params: Record<string, any>,
  fn: () => T | Promise<T>,
): Promise<T> {
  const store: SlotStore = {
    ...(pendingStore.getStore() ?? {}),
    [key]: new PendingSlot(dropUndefined(params)),
  };
  return pendingStore.run(store, async () => await fn());
}

export function nextSpan<T>(
  params: PendingSpanParams,
  fn: () => T | Promise<T>,
): Promise<T> {
  return withSlot("base", params, fn);
}

/** Set defaults for the next agent span (base + agent-specific fields). */
export function nextAgentSpan<T>(
  params: PendingAgentSpanParams,
  fn: () => T | Promise<T>,
): Promise<T> {
  return withSlot("agent", params, fn);
}

/** Set defaults for the next LLM span (base + LLM-specific fields). */
export function nextLlmSpan<T>(
  params: PendingLlmSpanParams,
  fn: () => T | Promise<T>,
): Promise<T> {
  return withSlot("llm", params, fn);
}

/** Set defaults for the next tool span (base + tool-specific fields). */
export function nextToolSpan<T>(
  params: PendingToolSpanParams,
  fn: () => T | Promise<T>,
): Promise<T> {
  return withSlot("tool", params, fn);
}

/** Set defaults for the next retriever span (base + retriever fields). */
export function nextRetrieverSpan<T>(
  params: PendingRetrieverSpanParams,
  fn: () => T | Promise<T>,
): Promise<T> {
  return withSlot("retriever", params, fn);
}

export function slotKindForSpanType(
  spanType: string | undefined,
): SlotKind | undefined {
  switch (spanType) {
    case "agent":
      return "agent";
    case "llm":
      return "llm";
    case "tool":
      return "tool";
    case "retriever":
      return "retriever";
    default:
      return undefined;
  }
}

/**
 * Consume the pending defaults for `spanType`, returning the merged
 * `{...ambient, ...base, ...typed}` payload — later layers win on overlap
 * ("more specific wins"):
 *
 *   1. **ambient** — scope-wide values from `setTracingContext(...)`, which apply
 *      to every matching span and are NOT consumed.
 *   2. **base** — a one-shot `nextSpan(...)` payload.
 *   3. **typed** — a one-shot `nextLlmSpan(...)`/etc. payload.
 *
 * The one-shot slots have their `payload` MUTATED to undefined rather than
 * replaced in the store; see {@link PendingSlot}.
 */
export function popPendingFor(spanType?: string): PendingPayload | undefined {
  const kindForAmbient = slotKindForSpanType(spanType);
  const ambient = ambientReader?.(kindForAmbient);
  const store = pendingStore.getStore();

  if (!store)
    return ambient && Object.keys(ambient).length > 0
      ? { ...ambient }
      : undefined;

  const merged: PendingPayload = { ...(ambient ?? {}) };

  const baseSlot = store.base;
  if (baseSlot?.payload) {
    Object.assign(merged, baseSlot.payload);
    baseSlot.payload = undefined;
  }

  const kind = slotKindForSpanType(spanType);
  if (kind) {
    const typedSlot = store[kind];
    if (typedSlot?.payload) {
      Object.assign(merged, typedSlot.payload);
      typedSlot.payload = undefined;
    }
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
}

/** What an OTel attribute is allowed to hold. */
type OtelAttributeValue = string | number | string[];

type AttributeEncoding = readonly [string, (value: any) => OtelAttributeValue];

const asText = (value: any): OtelAttributeValue =>
  typeof value === "string" ? value : JSON.stringify(value);
const asJson = (value: any): OtelAttributeValue => JSON.stringify(value);
const asStringList = (value: any): OtelAttributeValue =>
  (value as unknown[]).map(String);

/**
 * The `confident.*` attribute each staged field maps to, and how to encode it.
 *
 * Doubles as the allow-list the REST route filters on, so a field cannot be
 * taught to one route and forgotten on the other.
 */
const BASE_FIELD_ATTRS: Record<string, AttributeEncoding> = {
  input: [ConfidentAttr.SPAN_INPUT, asText],
  output: [ConfidentAttr.SPAN_OUTPUT, asText],
  retrievalContext: [ConfidentAttr.SPAN_RETRIEVAL_CONTEXT, asJson],
  context: [ConfidentAttr.SPAN_CONTEXT, asJson],
  expectedOutput: [ConfidentAttr.SPAN_EXPECTED_OUTPUT, asText],
  toolsCalled: [ConfidentAttr.SPAN_TOOLS_CALLED, asJson],
  expectedTools: [ConfidentAttr.SPAN_EXPECTED_TOOLS, asJson],
  metadata: [ConfidentAttr.SPAN_METADATA, asJson],
  name: [ConfidentAttr.SPAN_NAME, String],
  metricCollection: [ConfidentAttr.SPAN_METRIC_COLLECTION, String],
};

const TYPED_FIELD_ATTRS: Record<SlotKind, Record<string, AttributeEncoding>> = {
  agent: {
    availableTools: [ConfidentAttr.AGENT_AVAILABLE_TOOLS, asStringList],
    agentHandoffs: [ConfidentAttr.AGENT_AGENT_HANDOFFS, asStringList],
  },
  llm: {
    model: [ConfidentAttr.LLM_MODEL, String],
    inputTokenCount: [ConfidentAttr.LLM_INPUT_TOKEN_COUNT, Number],
    outputTokenCount: [ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT, Number],
    costPerInputToken: [ConfidentAttr.LLM_COST_PER_INPUT_TOKEN, Number],
    costPerOutputToken: [ConfidentAttr.LLM_COST_PER_OUTPUT_TOKEN, Number],
  },
  tool: {
    description: [ConfidentAttr.TOOL_DESCRIPTION, String],
  },
  retriever: {
    embedder: [ConfidentAttr.RETRIEVER_EMBEDDER, String],
    topK: [ConfidentAttr.RETRIEVER_TOP_K, Number],
    chunkSize: [ConfidentAttr.RETRIEVER_CHUNK_SIZE, Number],
  },
};

// `metrics` and `prompt` hold objects and so have no attribute of their own:
// metric instances only ever reach the REST route, and a `Prompt` is flattened
// into the four `confident.span.prompt_*` scalars instead.
const BASE_FIELDS: readonly string[] = [
  ...Object.keys(BASE_FIELD_ATTRS),
  "metrics",
];

// Guards against cross-type leakage (e.g. `embedder` landing on an LLM span).
const TYPED_FIELDS: Record<SlotKind, readonly string[]> = {
  agent: Object.keys(TYPED_FIELD_ATTRS.agent),
  llm: [...Object.keys(TYPED_FIELD_ATTRS.llm), "prompt"],
  tool: Object.keys(TYPED_FIELD_ATTRS.tool),
  retriever: Object.keys(TYPED_FIELD_ATTRS.retriever),
};

/**
 * Collapse a payload into the field values it effectively stages.
 *
 * `testCase` is unpacked first so that an individual field set in the same
 * payload wins over the one the test case supplied, mirroring
 * `updateCurrentSpan`.
 */
function resolvePendingFields(payload: PendingPayload): Record<string, any> {
  const resolved: Record<string, any> = {};

  const testCase = payload.testCase as LLMTestCase | undefined;
  if (testCase) {
    resolved.input = testCase.input;
    resolved.output = testCase.actualOutput;
    resolved.expectedOutput = testCase.expectedOutput;
    resolved.retrievalContext = testCase.retrievalContext;
    resolved.context = testCase.context;
    resolved.toolsCalled = testCase.toolsCalled;
    resolved.expectedTools = testCase.expectedTools;
  }

  for (const [key, value] of Object.entries(payload)) {
    if (key === "testCase" || value === undefined) continue;
    resolved[key] = value;
  }

  return resolved;
}

/**
 * Apply a popped payload to `span` in place.
 *
 * Mirrors `updateCurrentSpan` semantics for the base fields. Typed fields are
 * applied only when the span is of the matching type; mismatches are dropped
 * silently.
 */
export function applyPendingToSpan(
  span: BaseSpan,
  payload: PendingPayload | undefined,
): void {
  if (!payload) return;

  const target = span as unknown as Record<string, any>;
  const kind = slotKindForSpanType(span.type);
  const allowed = kind ? [...BASE_FIELDS, ...TYPED_FIELDS[kind]] : BASE_FIELDS;

  for (const [key, value] of Object.entries(resolvePendingFields(payload))) {
    if (!allowed.includes(key)) continue;
    target[key] = value;
  }
}

/**
 * Flatten a popped payload into `confident.*` OTel attributes.
 *
 * The OTLP counterpart to {@link applyPendingToSpan}: on that route no local
 * span object is ever built, so the attributes are the only carrier for
 * anything staged with `next*Span(...)`. A `Prompt` in particular cannot ride
 * in OTel attributes (primitives only), so it is flattened into the four
 * `confident.span.prompt_*` scalars the backend reads back to link the span to
 * its prompt version.
 *
 * `popPendingFor` drains the slot whether or not this can encode what it
 * finds, so every encodable field has to be handled here: a gap loses the
 * staged value outright rather than deferring it to a later span.
 */
export function pendingToOtelAttributes(
  payload: PendingPayload | undefined,
  spanType: string | undefined,
): Record<string, OtelAttributeValue> {
  const attrs: Record<string, OtelAttributeValue> = {};
  if (!payload) return attrs;

  const kind = slotKindForSpanType(spanType);
  const typedAttrs = kind ? TYPED_FIELD_ATTRS[kind] : {};
  const resolved = resolvePendingFields(payload);

  for (const [field, value] of Object.entries(resolved)) {
    // `setAttribute` drops null as readily as undefined, so an explicit null
    // cannot cross this route at all.
    if (value === undefined || value === null) continue;

    const encoding = BASE_FIELD_ATTRS[field] ?? typedAttrs[field];
    if (!encoding) continue;

    const [key, encode] = encoding;
    try {
      attrs[key] = encode(value);
    } catch {
      // Unencodable value (circular metadata, say). Skipping keeps the rest of
      // the payload intact rather than throwing out of `onStart`.
    }
  }

  if (kind === "llm") {
    const prompt = resolved.prompt as Prompt | undefined;
    if (prompt) {
      if (prompt._alias) attrs[ConfidentAttr.SPAN_PROMPT_ALIAS] = prompt._alias;
      if (prompt.hash)
        attrs[ConfidentAttr.SPAN_PROMPT_COMMIT_HASH] = prompt.hash;
      if (prompt.label) attrs[ConfidentAttr.SPAN_PROMPT_LABEL] = prompt.label;
      if (prompt.version)
        attrs[ConfidentAttr.SPAN_PROMPT_VERSION] = prompt.version;
    }
  }

  return attrs;
}

/**
 * Write a framework-derived value only where nothing is set already.
 *
 * Integrations extract their attributes at `onEnd`, long after anything staged
 * with `next*Span(...)` landed at `onStart`. On the OTLP route those staged
 * values live in the attributes themselves and there is no span object to fall
 * back on, so an unconditional write would erase what the user asked for.
 */
export function setDefaultSpanAttribute(
  attributes: Record<string, any>,
  key: string,
  value: OtelAttributeValue,
): void {
  if (attributes[key] === undefined) attributes[key] = value;
}

/** Test seam: drop every staged payload in the current scope. */
export function _clearPendingSlots(): void {
  const store = pendingStore.getStore();
  if (!store) return;
  for (const slot of Object.values(store)) {
    if (slot) slot.payload = undefined;
  }
}
