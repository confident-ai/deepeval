import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import {
  BatchSpanProcessor,
  SimpleSpanProcessor,
  SpanProcessor,
  ReadableSpan,
  Span,
  SpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { Context } from "@opentelemetry/api";
import { OpenInferenceSpanProcessor } from "./processor";
import { getSettings } from "../../config/settings";
import { ExportResult, ExportResultCode } from "@opentelemetry/core";
import { Prompt } from "../../prompt";

// OpenInference exporter filter to remove the parent Id for root spans
class OpenInferenceExporterWrapper implements SpanExporter {
  private aiSpanIds = new Set<string>();

  constructor(private readonly exporter: SpanExporter) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    const currentBatchIds: string[] = [];

    spans.forEach((span) => {
      const attrs = (span as any).attributes || {};
      if (
        attrs["confident.internal.is_oi_span"] ||
        attrs["openinference.span.kind"]
      ) {
        const id = span.spanContext().spanId;
        this.aiSpanIds.add(id);
        currentBatchIds.push(id);
      }
    });

    const sanitized = spans.map((span) => {
      let parentSpanContext = span.parentSpanContext;
      const attrs = (span as any).attributes || {};
      const kind = attrs["openinference.span.kind"];

      // 2. Identify potential root AI spans (LLM, AGENT, CHAIN)
      if (kind === "LLM" || kind === "AGENT" || kind === "CHAIN") {
        const parentId = span.parentSpanContext?.spanId;

        if (!parentId || !this.aiSpanIds.has(parentId)) {
          parentSpanContext = undefined;
        }
      }

      return Object.assign(Object.create(span), {
        parentSpanContext,
      });
    });

    try {
      this.exporter.export(sanitized as ReadableSpan[], resultCallback);
    } catch {
      resultCallback({ code: ExportResultCode.FAILED });
    }

    currentBatchIds.forEach((id) => this.aiSpanIds.delete(id));
  }

  shutdown(): Promise<void> {
    return this.exporter.shutdown();
  }

  forceFlush(): Promise<void> {
    return this.exporter.forceFlush?.() ?? Promise.resolve();
  }
}

// --- OpenInference Filter Processor ---
// Filters out non-OpenInference spans so foreign spans from other instrumentation
export class OpenInferenceFilterProcessor implements SpanProcessor {
  constructor(private readonly underlyingProcessor: SpanProcessor) {}

  forceFlush(): Promise<void> {
    return this.underlyingProcessor.forceFlush();
  }

  onStart(span: Span, parentContext: Context): void {
    this.underlyingProcessor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    const attrs = (span as any).attributes || {};
    // Also allow spans that were flagged by OpenInferenceSpanProcessor in onStart
    if (
      attrs["openinference.span.kind"] ||
      attrs["confident.internal.is_oi_span"]
    ) {
      this.underlyingProcessor.onEnd(span);
    }
  }

  shutdown(): Promise<void> {
    return this.underlyingProcessor.shutdown();
  }
}

export interface OpenInferenceInstrumentationOptions {
  apiKey?: string;
  otelEndpoint?: string;
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
  isTestMode?: boolean;
  debug?: boolean;
}

export function createOpenInferenceProcessors(
  options?: OpenInferenceInstrumentationOptions,
): SpanProcessor[] {
  const settings = getSettings();
  const apiKey =
    options?.apiKey ??
    (typeof process !== "undefined"
      ? process.env.CONFIDENT_API_KEY
      : undefined);

  if (!apiKey) {
    console.warn(
      "DeepEval: No API Key found. OpenInference tracing will be disabled.",
    );
    return [];
  }

  const baseUrl =
    options?.otelEndpoint ||
    settings.CONFIDENT_OTEL_URL ||
    "https://otel.confident-ai.com";

  const endpoint = baseUrl.endsWith("/")
    ? `${baseUrl}v1/traces`
    : `${baseUrl}/v1/traces`;

  const otlpExporter = new OpenInferenceExporterWrapper(
    new OTLPTraceExporter({
      url: endpoint,
      headers: { "x-confident-api-key": apiKey },
    }),
  );

  const baseExporterProcessor = options?.isTestMode
    ? new SimpleSpanProcessor(otlpExporter)
    : new BatchSpanProcessor(otlpExporter);

  return [
    new OpenInferenceSpanProcessor(options),
    new OpenInferenceFilterProcessor(baseExporterProcessor),
  ];
}

let _openInferenceTracerProvider: NodeTracerProvider | null = null;
let _currentOptions: OpenInferenceInstrumentationOptions | null = null;

export function instrumentOpenInference(
  options?: OpenInferenceInstrumentationOptions,
): void {
  _currentOptions = options || {};

  const processors = createOpenInferenceProcessors(_currentOptions);
  if (processors.length === 0) return;

  let environment = options?.environment;
  if (!environment && getSettings().CONFIDENT_TRACE_ENVIRONMENT) {
    environment = getSettings().CONFIDENT_TRACE_ENVIRONMENT;
  } else if (!environment) {
    environment = "development";
  }

  const provider = new NodeTracerProvider({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: "deepeval-ts-client",
      "deepeval.sdk.version": "typescript",
      "deepeval.environment": environment,
    }),
    spanProcessors: processors,
  });

  _openInferenceTracerProvider = provider;

  setupGracefulShutdown(provider, options?.debug);

  provider.register();

  if (options?.debug) {
    console.log("DeepEval OpenInference tracing configured:", {
      environment,
      name: options?.name,
      isTestMode: options?.isTestMode,
    });
  }
}

export function getCurrentOptions(): OpenInferenceInstrumentationOptions | null {
  return _currentOptions;
}

export function resetInstrumentation(): void {
  _currentOptions = null;
  _openInferenceTracerProvider = null;
}

function setupGracefulShutdown(provider: NodeTracerProvider, debug?: boolean) {
  let isShuttingDown = false;

  const shutdown = async (signal: string) => {
    if (isShuttingDown) return;
    isShuttingDown = true;

    if (debug) {
      console.log(`DeepEval: Received ${signal}, flushing traces...`);
    }

    try {
      const timeout = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Timeout")), 5000),
      );

      await Promise.race([provider.shutdown(), timeout]);

      if (debug) console.log("DeepEval: Traces flushed successfully.");
    } catch (err) {
      if (debug) console.warn("DeepEval: Failed to flush traces on exit", err);
    } finally {
      if (signal === "SIGINT" || signal === "SIGTERM") {
        process.exit(0);
      }
    }
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  process.on("beforeExit", async () => {
    await provider.forceFlush();
  });
}

export async function forceFlush(): Promise<void> {
  if (_openInferenceTracerProvider) {
    await _openInferenceTracerProvider.forceFlush();
  }
}

export { OpenInferenceSpanProcessor };
