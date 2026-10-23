/**
 * deepeval's LangChain and LangGraph integration, built on confident-trace.
 *
 * `DeepEvalCallbackHandler` is confident-trace's LangChain handler, so it
 * traces every run it is passed to. It adds what only deepeval knows: the
 * trace's settings and metrics, and the metrics, metric collection and prompt
 * that a model's `metadata` names. The run you invoke is an agent span, which
 * `nextAgentSpan(...)` lands on.
 */

import type { BaseMetric } from "@/metrics/base-metrics";
import { recordTracingIntegration } from "@/telemetry";
import { Integration } from "@/tracing/integrations";
import {
  type PendingPayload,
  type StagedValues,
  withStaged,
} from "@/tracing/pending-context";
import {
  confidentTraceModule,
  ensureConfidentTrace,
} from "@/tracing/otel/setup";

type LangChainAdapter = typeof import("confident-trace/langchain");

const { ConfidentLangChainCallbackHandler } =
  confidentTraceModule<LangChainAdapter>("langchain");

export interface DeepEvalCallbackHandlerOptions {
  name?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  turnId?: string;
  metrics?: BaseMetric[];
  metricCollection?: string;
}

function field(metadata: unknown, key: string): any {
  return metadata && typeof metadata === "object"
    ? (metadata as Record<string, unknown>)[key]
    : undefined;
}

/** What a model's `metadata` stages for its LLM span. */
function llmFields(metadata: unknown): PendingPayload {
  return {
    metrics: field(metadata, "metrics"),
    metricCollection: field(metadata, "metricCollection"),
    prompt: field(metadata, "prompt"),
  };
}

export class DeepEvalCallbackHandler extends ConfidentLangChainCallbackHandler {
  constructor(options: DeepEvalCallbackHandlerOptions = {}) {
    ensureConfidentTrace();
    super({ rootSpanType: "agent" });
    recordTracingIntegration(Integration.LANGCHAIN);
    const trace: PendingPayload = { ...options };
    // confident-trace opens each run's span synchronously in its handler, so
    // what is staged around the call reaches that span.
    const staged = <A extends unknown[]>(
      start: (...args: A) => void,
      values: (args: A) => StagedValues,
    ) => {
      return (...args: A): void => {
        const parentRunId = args[3];
        withStaged({ ...values(args), ...(parentRunId ? {} : { trace }) }, () =>
          start(...args),
        );
      };
    };
    this.handleChainStart = staged(this.handleChainStart, () => ({}));
    this.handleToolStart = staged(this.handleToolStart, () => ({}));
    this.handleLLMStart = staged(this.handleLLMStart, (args) => ({
      span: llmFields(args[6]),
      spanKind: "llm",
    }));
    this.handleChatModelStart = staged(this.handleChatModelStart, (args) => ({
      span: llmFields(args[6]),
      spanKind: "llm",
    }));
    this.handleRetrieverStart = staged(this.handleRetrieverStart, (args) => ({
      span: {
        embedder: field(args[5], "ls_embedding_provider"),
        metricCollection: field(args[5], "metricCollection"),
      },
      spanKind: "retriever",
    }));
  }
}
