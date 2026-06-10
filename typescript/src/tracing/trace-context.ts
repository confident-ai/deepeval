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

export type LlmSpanContext = {
  prompt?: Prompt;
  //   metrics?: BaseMetric[];
  metricCollection?: string;
  toolsMetricCollection?: string;
  expectedOutput?: string;
  expectedTools?: ToolCall[];
  context?: string[];
  retrievalContext?: string[];
};

export type AgentSpanContext = {
  //   metrics?: BaseMetric[];
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

  //   if (opts.metrics) {
  //     currentTrace.metrics = opts.metrics
  //   }
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
    //   metrics: opts.metrics,
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
