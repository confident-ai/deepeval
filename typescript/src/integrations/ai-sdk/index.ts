/**
 * deepeval's AI SDK integration, built on confident-trace (AI SDK 7).
 *
 * Pass the tracer to `@ai-sdk/otel`'s integration:
 *
 *   const telemetry = {
 *     integrations: [new OpenTelemetry({ tracer: configureAiSdkTracing() })],
 *   };
 *   await generateText({ model, prompt, telemetry });
 */

import type { Tracer } from "@opentelemetry/api";

import { recordTracingIntegration } from "@/telemetry";
import { Integration } from "@/tracing/integrations";
import {
  confidentTraceModule,
  ensureConfidentTrace,
  flushConfidentTrace,
  setTraceDefaults,
} from "@/tracing/otel/setup";

type VercelAdapter = typeof import("confident-trace/vercel-ai");

// The label confident-trace gives the AI SDK's spans.
const INTEGRATION = "Vercel AI SDK";

export interface AiSdkInstrumentationOptions {
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  apiKey?: string;
  /** The name of every trace. */
  name?: string;
  environment?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  traceMetricCollection?: string;
  /** @deprecated Set `CONFIDENT_OTEL_URL` instead. */
  otelEndpoint?: string;
  /** @deprecated Evaluations now collect spans locally on their own. */
  isTestMode?: boolean;
  /** @deprecated No longer used. */
  debug?: boolean;
}

let currentOptions: AiSdkInstrumentationOptions | null = null;

/** A tracer that traces AI SDK calls for deepeval. */
export function configureAiSdkTracing(
  options: AiSdkInstrumentationOptions = {},
): Tracer {
  currentOptions = options;
  if (options.otelEndpoint)
    process.env.CONFIDENT_OTEL_URL ??= options.otelEndpoint;
  ensureConfidentTrace();
  recordTracingIntegration(Integration.AI_SDK);
  setTraceDefaults(
    { integration: INTEGRATION },
    { name: options.name, environment: options.environment },
  );
  return confidentTraceModule<VercelAdapter>(
    "vercel-ai",
  ).createVercelAITracer();
}

export function getCurrentOptions(): AiSdkInstrumentationOptions | null {
  return currentOptions;
}

/** Wait for the AI SDK's spans to be sent. */
export async function forceFlush(): Promise<void> {
  await flushConfidentTrace(30000);
}
