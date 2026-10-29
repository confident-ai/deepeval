/**
 * deepeval's OpenInference integration, built on confident-trace.
 *
 * Any `@arizeai/openinference-instrumentation-*` instrumentor traces on the
 * global tracer provider, which confident-trace runs once deepeval starts it.
 * Register your instrumentor after `instrumentOpenInference(...)`.
 */

import type { Span } from "@opentelemetry/sdk-trace-base";

import type { Prompt } from "@/prompt";
import { recordTracingIntegration } from "@/telemetry";
import { ConfidentAttr } from "@/tracing/attributes";
import { Integration } from "@/tracing/integrations";
import { SpanType, traceManager } from "@/tracing/tracing";
import { resolveSpanType } from "@/tracing/otel/converter";
import { SCOPE } from "@/tracing/otel/adapters/openinference";
import {
  ensureConfidentTrace,
  flushConfidentTrace,
  setTraceDefaults,
} from "@/tracing/otel/setup";

export interface OpenInferenceInstrumentationOptions {
  /** The Confident AI key to send traces with, over `CONFIDENT_API_KEY`. */
  apiKey?: string;
  name?: string;
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  turnId?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  environment?: string;
  metricCollection?: string;
  traceMetricCollection?: string;
  llmMetricCollection?: string;
  agentMetricCollection?: string;
  toolMetricCollectionMap?: Record<string, string>;
  prompt?: Prompt;
  /** @deprecated Set `CONFIDENT_OTEL_URL` instead. */
  otelEndpoint?: string;
  /** @deprecated Evaluations now collect spans locally on their own. */
  isTestMode?: boolean;
  /** @deprecated No longer used. */
  debug?: boolean;
}

let currentOptions: OpenInferenceInstrumentationOptions | null = null;

/** Trace whatever OpenInference instrumentors run for deepeval. */
export function instrumentOpenInference(
  options: OpenInferenceInstrumentationOptions = {},
): void {
  currentOptions = options;
  if (options.apiKey)
    traceManager.configure({ confidentApiKey: options.apiKey });
  if (options.otelEndpoint)
    process.env.CONFIDENT_OTEL_URL ??= options.otelEndpoint;
  ensureConfidentTrace();
  recordTracingIntegration(Integration.OPEN_INFERENCE);
  setTraceDefaults(
    { scope: SCOPE },
    {
      name: options.name,
      threadId: options.threadId,
      userId: options.userId,
      testCaseId: options.testCaseId,
      turnId: options.turnId,
      metadata: options.metadata,
      tags: options.tags,
      environment: options.environment,
      metricCollection:
        options.traceMetricCollection ?? options.metricCollection,
    },
    // The label also has confident-trace export these spans.
    { [ConfidentAttr.SPAN_INTEGRATION]: Integration.OPEN_INFERENCE },
    (span) => spanSettings(span, options),
  );
}

/** The metric collection and prompt that `options` give `span`. */
function spanSettings(
  span: Span,
  options: OpenInferenceInstrumentationOptions,
): Record<string, string> {
  const type = resolveSpanType(span, true);
  const toolName = span.attributes["tool.name"];
  const collection =
    type === SpanType.LLM
      ? options.llmMetricCollection
      : type === SpanType.AGENT
        ? options.agentMetricCollection
        : type === SpanType.TOOL && typeof toolName === "string"
          ? options.toolMetricCollectionMap?.[toolName]
          : undefined;
  const settings: Record<string, string> = {};
  if (collection) settings[ConfidentAttr.SPAN_METRIC_COLLECTION] = collection;
  const prompt = options.prompt;
  if (type === SpanType.LLM && prompt) {
    if (prompt._alias)
      settings[ConfidentAttr.SPAN_PROMPT_ALIAS] = prompt._alias;
    if (prompt.hash)
      settings[ConfidentAttr.SPAN_PROMPT_COMMIT_HASH] = prompt.hash;
    if (prompt.label) settings[ConfidentAttr.SPAN_PROMPT_LABEL] = prompt.label;
    if (prompt.version)
      settings[ConfidentAttr.SPAN_PROMPT_VERSION] = prompt.version;
  }
  return settings;
}

export function getCurrentOptions(): OpenInferenceInstrumentationOptions | null {
  return currentOptions;
}

/** Wait for the OpenInference spans to be sent. */
export async function forceFlush(): Promise<void> {
  await flushConfidentTrace(30000);
}
