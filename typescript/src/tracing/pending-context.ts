import { AsyncLocalStorage } from "async_hooks";

import { BaseMetric } from "../metrics/base-metrics";
import { LLMTestCase, ToolCall } from "../test-case";
import { Prompt } from "../prompt";
import type { BaseSpan } from "./tracing";

/**
 * `next*Span`: declarative defaults for the NEXT span of a given type.
 *
 * Counterpart to `updateCurrentSpan(...)` for spans without a user-code seam —
 * i.e. spans the user never executes code inside, so `updateCurrentSpan` from
 * "their" body isn't reachable. The canonical case is an integration-emitted
 * agent / LLM span where the only callsite the user owns is the one wrapping
 * the framework call.
 *
 * Semantics (mirrors Python's `next_*_span` family):
 *   - One-shot: the payload is consumed by the FIRST span of the matching type
 *     created inside the active scope. Subsequent spans see an empty slot.
 *   - Per-type isolation: each helper writes to its own slot, so nesting
 *     `nextAgentSpan(..., () => nextLlmSpan(..., fn))` is unambiguous.
 *   - One-stop params: each helper accepts BASE fields (everything
 *     `updateCurrentSpan` takes) AND its type-specific fields in one call.
 *   - Consumer responsibility: integrations call `popPendingFor(...)` when they
 *     classify a fresh span and apply the payload to it. If nothing is
 *     listening the payload is discarded when the scope exits.
 */

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

/**
 * Mutable wrapper around a pending-defaults payload.
 *
 * Why a wrapper instead of putting the payload straight into the store: a
 * consumer often runs inside a NESTED `AsyncLocalStorage.run(...)` scope (the
 * framework opens its own context around the call). Replacing the store's
 * field from there would not be visible to the outer `next*Span` scope, so a
 * second framework call inside the same scope could re-consume a value that
 * was already drained. Store inheritance copies the REFERENCE to this wrapper,
 * so mutating `payload` is visible in both the consumer's sub-context and the
 * outer scope. (Same rationale as Python's `_PendingSlot`.)
 */
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

/**
 * Reads *ambient* defaults for a span kind — values that apply to every matching
 * span in scope rather than being consumed by the first one.
 *
 * This is the seam for `setTracingContext({ llmSpanContext, agentSpanContext })`,
 * which predates `next*Span` and is scope-wide rather than one-shot. Registering
 * it here means an integration only ever calls {@link popPendingFor} and gets
 * both mechanisms merged, with the one-shot payload winning.
 *
 * Injected rather than imported so this module stays free of a require cycle
 * (`trace-context` → `tracing` → `pending-context`).
 */
export type AmbientPayloadReader = (
  kind: SlotKind | undefined,
) => PendingPayload | undefined;

let ambientReader: AmbientPayloadReader | undefined;

/** @internal Registered by `trace-context` at module load. */
export function _setAmbientPayloadReader(
  reader: AmbientPayloadReader | undefined,
): void {
  ambientReader = reader;
}

/** Strip undefined values so consumers don't re-check every field. */
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

/**
 * Set base-span defaults for the next span of ANY type, for the duration of
 * `fn`. Use when the upcoming span's type doesn't matter or isn't known; for a
 * typed match use `nextAgentSpan` / `nextLlmSpan` / `nextToolSpan` /
 * `nextRetrieverSpan`.
 */
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

/**
 * Map a span's `type` onto the typed slot it consumes, if any.
 *
 * Compared against literals rather than the `SpanType` enum on purpose: the
 * enum lives in `tracing.ts`, which imports this module, so a value import
 * would create a require cycle. The enum's values ARE these strings.
 */
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

const BASE_FIELDS: readonly string[] = [
  "input",
  "output",
  "retrievalContext",
  "context",
  "expectedOutput",
  "toolsCalled",
  "expectedTools",
  "metadata",
  "name",
  "metricCollection",
  "metrics",
];

// Guards against cross-type leakage (e.g. `embedder` landing on an LLM span).
// Python uses `hasattr`; an allowlist is used here instead because spans are
// not always class instances — the `observe` path builds plain object literals,
// so an `in` check would silently drop legitimate fields.
const TYPED_FIELDS: Record<SlotKind, readonly string[]> = {
  agent: ["availableTools", "agentHandoffs"],
  llm: [
    "model",
    "inputTokenCount",
    "outputTokenCount",
    "costPerInputToken",
    "costPerOutputToken",
    "prompt",
  ],
  tool: ["description"],
  retriever: ["embedder", "topK", "chunkSize"],
};

/**
 * Apply a popped payload to `span` in place.
 *
 * Mirrors `updateCurrentSpan` semantics for the base fields — notably the
 * `testCase` unpacking path, which writes the test case's fields onto the span
 * and is overridden by any individual field set in the same payload. Typed
 * fields are applied only when the span is of the matching type; mismatches
 * are dropped silently.
 */
export function applyPendingToSpan(
  span: BaseSpan,
  payload: PendingPayload | undefined,
): void {
  if (!payload) return;

  const target = span as unknown as Record<string, any>;

  const testCase = payload.testCase as LLMTestCase | undefined;
  if (testCase) {
    target.input = testCase.input;
    target.output = testCase.actualOutput;
    target.expectedOutput = testCase.expectedOutput;
    target.retrievalContext = testCase.retrievalContext;
    target.context = testCase.context;
    target.toolsCalled = testCase.toolsCalled;
    target.expectedTools = testCase.expectedTools;
  }

  const kind = slotKindForSpanType(span.type);
  const allowed = kind ? [...BASE_FIELDS, ...TYPED_FIELDS[kind]] : BASE_FIELDS;

  for (const [key, value] of Object.entries(payload)) {
    if (key === "testCase" || value === undefined) continue;
    if (!allowed.includes(key)) continue;
    target[key] = value;
  }
}

/** Test seam: drop every staged payload in the current scope. */
export function _clearPendingSlots(): void {
  const store = pendingStore.getStore();
  if (!store) return;
  for (const slot of Object.values(store)) {
    if (slot) slot.payload = undefined;
  }
}
