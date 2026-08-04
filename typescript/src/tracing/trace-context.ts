import { AsyncLocalStorage } from "async_hooks";

import {
  getCurrentTrace,
  setCurrentTrace,
  Trace,
  traceManager,
  updateCurrentTrace,
} from "./tracing";

import { Prompt } from "../prompt";
import { ToolCall } from "../test-case";
import type { BaseMetric } from "../metrics/base-metrics";
import { _setAmbientPayloadReader } from "./pending-context";

export type LlmSpanContext = {
  prompt?: Prompt;
  metrics?: BaseMetric[];
  metricCollection?: string;
  toolsMetricCollection?: string;
  toolsMetrics?: BaseMetric[];
  expectedOutput?: string;
  expectedTools?: ToolCall[];
  context?: string[];
  retrievalContext?: string[];
};

export type AgentSpanContext = {
  metrics?: BaseMetric[];
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
      metricCollection: ctx.metricCollection,
      prompt: ctx.prompt,
      expectedOutput: ctx.expectedOutput,
      expectedTools: ctx.expectedTools,
      context: ctx.context,
      retrievalContext: ctx.retrievalContext,
    });
  }

  if (kind === "tool") {
    // `toolsMetricCollection`/`toolsMetrics` are declared on the LLM context but
    // target the tool spans in scope.
    const ctx = getLlmContext();
    if (!ctx) return undefined;
    return drop({
      metrics: ctx.toolsMetrics,
      metricCollection: ctx.toolsMetricCollection,
    });
  }

  if (kind === "agent") {
    const ctx = getAgentContext();
    if (!ctx) return undefined;
    return drop({
      metrics: ctx.metrics,
      metricCollection: ctx.metricCollection,
      expectedOutput: ctx.expectedOutput,
      expectedTools: ctx.expectedTools,
      context: ctx.context,
      retrievalContext: ctx.retrievalContext,
    });
  }

  return undefined;
});

export async function setTracingContext<T>(
  opts: Partial<Trace> & {
    llmSpanContext?: LlmSpanContext;
    agentSpanContext?: AgentSpanContext;
  },
  fn: () => Promise<T> | T,
): Promise<T> {
  let currentTrace = getCurrentTrace();
  if (!currentTrace) {
    currentTrace = traceManager.startNewTrace();
  }

  if (opts.metrics) {
    currentTrace.metrics = opts.metrics;
  }
  if (opts.metricCollection) {
    currentTrace.metricCollection = opts.metricCollection;
  }

  setCurrentTrace(currentTrace);
  updateCurrentTrace({
    name: opts.name,
    tags: opts.tags,
    metadata: opts.metadata,
    threadId: opts.threadId,
    userId: opts.userId,
    testCaseId: opts.testCaseId,
    turnId: opts.turnId,
    input: opts.input,
    output: opts.output,
    retrievalContext: opts.retrievalContext,
    context: opts.context,
    expectedOutput: opts.expectedOutput,
    toolsCalled: opts.toolsCalled,
    expectedTools: opts.expectedTools,
    metrics: opts.metrics,
    metricCollection: opts.metricCollection,
  });

  if (opts.llmSpanContext) {
    setLlmContext(opts.llmSpanContext);
  }
  if (opts.agentSpanContext) {
    setAgentContext(opts.agentSpanContext);
  }

  return await fn();
}
