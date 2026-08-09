import { AsyncLocalStorage } from "async_hooks";

import { BaseMetric } from "@/metrics/base-metrics";
import { LLMTestCase, ToolCall } from "@/test-case";
import { Prompt } from "@/prompt";
import type { BaseSpan } from "@/tracing/tracing";

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
