import { AsyncLocalStorage } from "async_hooks";

import {
  getCurrentTrace,
  setCurrentTrace,
  Trace,
  traceManager,
  updateCurrentTrace,
} from "@/tracing/tracing";

import { Prompt } from "@/prompt";
import { ToolCall } from "@/test-case";
import type { BaseMetric } from "@/metrics/base-metrics";
import { _setAmbientPayloadReader } from "@/tracing/pending-context";

export type LlmSpanContext = {
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  prompt?: Prompt;
  metrics?: BaseMetric[];
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  metricCollection?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  toolsMetricCollection?: string;
  toolsMetrics?: BaseMetric[];
  expectedOutput?: string;
  expectedTools?: ToolCall[];
  context?: string[];
  retrievalContext?: string[];
};

export type AgentSpanContext = {
  metrics?: BaseMetric[];
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  metricCollection?: string;
  expectedOutput?: string;
  expectedTools?: ToolCall[];
  context?: string[];
  retrievalContext?: string[];
};

const llmSpanContextStore = new AsyncLocalStorage<LlmSpanContext>();
export function getLlmContext(): LlmSpanContext | undefined {
  return llmSpanContextStore.getStore();
}
export function setLlmContext(context: LlmSpanContext | null): void {
  llmSpanContextStore.enterWith(context ?? {});
}

const agentSpanContextStore = new AsyncLocalStorage<AgentSpanContext>();
export function getAgentContext(): AgentSpanContext | undefined {
  return agentSpanContextStore.getStore();
}
export function setAgentContext(context: AgentSpanContext | null): void {
  agentSpanContextStore.enterWith(context ?? {});
}

_setAmbientPayloadReader((kind) => {
  const drop = (payload: Record<string, unknown>) => {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(payload)) {
      if (value !== undefined) out[key] = value;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  };

  if (kind === "llm") {
    const ctx = getLlmContext();
    if (!ctx) return undefined;
    return drop({
      metrics: ctx.metrics,
      expectedOutput: ctx.expectedOutput,
      expectedTools: ctx.expectedTools,
      context: ctx.context,
      retrievalContext: ctx.retrievalContext,
    });
  }

  if (kind === "tool") {
    // `toolsMetrics` is declared on the LLM context but targets the tool spans
    // in scope.
    const ctx = getLlmContext();
    if (!ctx) return undefined;
    return drop({ metrics: ctx.toolsMetrics });
  }

  if (kind === "agent") {
    const ctx = getAgentContext();
    if (!ctx) return undefined;
    return drop({
      metrics: ctx.metrics,
      expectedOutput: ctx.expectedOutput,
      expectedTools: ctx.expectedTools,
      context: ctx.context,
      retrievalContext: ctx.retrievalContext,
    });
  }

  return undefined;
});

/** Confident AI platform settings `setTracingContext` accepts and ignores. */
type DeprecatedTraceSettings = {
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  testCaseId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  testRunId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  turnId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  metricCollection?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  confidentApiKey?: string;
};

export async function setTracingContext<T>(
  opts: Partial<Trace> &
    DeprecatedTraceSettings & {
      llmSpanContext?: LlmSpanContext;
      agentSpanContext?: AgentSpanContext;
    },
  fn: () => Promise<T> | T,
): Promise<T> {
  let currentTrace = getCurrentTrace();
  // One left in context by an earlier call may have ended.
  if (!currentTrace || !traceManager.getTraceByUuid(currentTrace.uuid)) {
    currentTrace = traceManager.startNewTrace();
  }

  if (opts.metrics) {
    currentTrace.metrics = opts.metrics;
  }

  setCurrentTrace(currentTrace);
  updateCurrentTrace({
    name: opts.name,
    tags: opts.tags,
    metadata: opts.metadata,
    threadId: opts.threadId,
    userId: opts.userId,
    input: opts.input,
    output: opts.output,
    retrievalContext: opts.retrievalContext,
    context: opts.context,
    expectedOutput: opts.expectedOutput,
    toolsCalled: opts.toolsCalled,
    expectedTools: opts.expectedTools,
    metrics: opts.metrics,
  });

  if (opts.llmSpanContext) {
    setLlmContext(opts.llmSpanContext);
  }
  if (opts.agentSpanContext) {
    setAgentContext(opts.agentSpanContext);
  }

  // Loaded on first use, like `observe` does.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const setup = require("./otel/setup") as typeof import("./otel/setup");
  return await setup.exportedTraceContext(
    {
      name: opts.name,
      tags: opts.tags,
      metadata: opts.metadata,
      threadId: opts.threadId,
      userId: opts.userId,
      input: opts.input,
      output: opts.output,
      retrievalContext: opts.retrievalContext,
      context: opts.context,
      expectedOutput: opts.expectedOutput,
      toolsCalled: opts.toolsCalled,
      expectedTools: opts.expectedTools,
    },
    async () => await fn(),
  );
}
