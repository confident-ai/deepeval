/**
 * deepeval's OpenInference integration, built on confident-trace.
 *
 * Any `@arizeai/openinference-instrumentation-*` instrumentor traces on the
 * global tracer provider, which confident-trace runs once deepeval starts it.
 * Register your instrumentor after `instrumentOpenInference(...)`.
 */

import type { Prompt } from "@/prompt";
import { recordTracingIntegration } from "@/telemetry";
import { ConfidentAttr } from "@/tracing/attributes";
import { Integration } from "@/tracing/integrations";
import { SCOPE } from "@/tracing/otel/adapters/openinference";
import {
  ensureConfidentTrace,
  flushConfidentTrace,
  setTraceDefaults,
} from "@/tracing/otel/setup";

export interface OpenInferenceInstrumentationOptions {
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  apiKey?: string;
  name?: string;
  threadId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  userId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  testCaseId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  turnId?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  environment?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  metricCollection?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  traceMetricCollection?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  llmMetricCollection?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  agentMetricCollection?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  toolMetricCollectionMap?: Record<string, string>;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
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
  if (options.otelEndpoint)
    process.env.CONFIDENT_OTEL_URL ??= options.otelEndpoint;
  ensureConfidentTrace();
  recordTracingIntegration(Integration.OPEN_INFERENCE);
  setTraceDefaults(
    { scope: SCOPE },
    {
      name: options.name,
      threadId: options.threadId,
      metadata: options.metadata,
      tags: options.tags,
      environment: options.environment,
    },
    // The label also has confident-trace export these spans.
    { [ConfidentAttr.SPAN_INTEGRATION]: Integration.OPEN_INFERENCE },
  );
}

export function getCurrentOptions(): OpenInferenceInstrumentationOptions | null {
  return currentOptions;
}

/** Wait for the OpenInference spans to be sent. */
export async function forceFlush(): Promise<void> {
  await flushConfidentTrace(30000);
}
