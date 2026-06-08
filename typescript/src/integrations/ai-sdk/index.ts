import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import {
  BatchSpanProcessor,
  SimpleSpanProcessor,
  SpanProcessor,
  ReadableSpan,
  SpanExporter,
  Span,
} from "@opentelemetry/sdk-trace-base";
import { ExportResult, ExportResultCode } from "@opentelemetry/core";
import { Tracer, trace, Context } from "@opentelemetry/api";
import { DeepEvalSpanProcessor, ROOT_VERCEL_SPANS } from "./processor";
import { getSettings } from "../../config/settings";

// Creating a Wrapper for exporter to preserve parentIds of root spans
class DeepEvalExporterWrapper implements SpanExporter {
  private aiSpanIds = new Set<string>();

  constructor(private readonly exporter: SpanExporter) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    const currentBatchIds: string[] = [];

    // Track AI spans across batches
    spans.forEach((span) => {
      if (span.name.startsWith("ai.")) {
        const id = span.spanContext().spanId;
        this.aiSpanIds.add(id);
        currentBatchIds.push(id);
      }
    });

    const sanitized = spans.map((span) => {
      let parentSpanContext = span.parentSpanContext;

      if (ROOT_VERCEL_SPANS.has(span.name)) {
        const parentId = span.parentSpanContext?.spanId;

        // Only strip parent if it's not another AI span
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

// --- DeepEval Custom Export Filter ---
export class DeepEvalBatchFilterProcessor implements SpanProcessor {
  constructor(private readonly underlyingProcessor: SpanProcessor) {}

  forceFlush(): Promise<void> {
    return this.underlyingProcessor.forceFlush();
  }

  // We filter only the "ai." spans from Vercel AI SDK and abandon any and all spans that are not related to vercel's "ai." spans
  onStart(span: Span, parentContext: Context): void {
    const name = (span as any).name;
    if (name && name.startsWith("ai.")) {
      this.underlyingProcessor.onStart(span, parentContext);
    }
  }

  onEnd(span: ReadableSpan): void {
    if (span.name && span.name.startsWith("ai.")) {
      this.underlyingProcessor.onEnd(span);
    }
  }

  shutdown(): Promise<void> {
    return this.underlyingProcessor.shutdown();
  }
}

export interface AiSdkInstrumentationOptions {
  apiKey?: string;
  otelEndpoint?: string;
  name?: string;
  isTestMode?: boolean;
  debug?: boolean;
  environment?: string;
  traceMetricCollection?: string;
}

export function createDeepEvalProcessors(
  options?: AiSdkInstrumentationOptions,
): SpanProcessor[] {
  const settings = getSettings();
  const apiKey =
    options?.apiKey ??
    (typeof process !== "undefined"
      ? process.env.CONFIDENT_API_KEY
      : undefined);

  if (!apiKey) {
    console.warn(
      "DeepEval: No API Key found. AI SDK tracing will be disabled.",
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

  const otlpExporter = new DeepEvalExporterWrapper(
    new OTLPTraceExporter({
      url: endpoint,
      headers: { "x-confident-api-key": apiKey },
    }),
  );

  const baseExporterProcessor = options?.isTestMode
    ? new SimpleSpanProcessor(otlpExporter)
    : new BatchSpanProcessor(otlpExporter);

  return [
    new DeepEvalSpanProcessor(options),
    new DeepEvalBatchFilterProcessor(baseExporterProcessor),
  ];
}

let _deepevalAiSdkTracer: Tracer | null = null;
let _currentOptions: AiSdkInstrumentationOptions | null = null;
let _deepevalTracerProvider: NodeTracerProvider | null = null;

export function configureAiSdkTracing(
  options?: AiSdkInstrumentationOptions,
): Tracer | null {
  _currentOptions = options || {};

  if (_deepevalAiSdkTracer) {
    return _deepevalAiSdkTracer;
  }

  const processors = createDeepEvalProcessors(_currentOptions);
  if (processors.length === 0) return null;

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

  _deepevalTracerProvider = provider;

  setupGracefulShutdown(provider, options?.debug);

  provider.register();

  _deepevalAiSdkTracer = trace.getTracer("deepeval-ai-sdk");

  if (options?.debug) {
    console.log("DeepEval AI SDK tracing configured:", {
      environment,
      name: options?.name,
      isTestMode: options?.isTestMode,
    });
  }

  return _deepevalAiSdkTracer;
}

export function getCurrentOptions(): AiSdkInstrumentationOptions | null {
  return _currentOptions;
}

export function resetTracer(): void {
  _deepevalAiSdkTracer = null;
  _currentOptions = null;
  _deepevalTracerProvider = null;
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
  if (_deepevalTracerProvider) {
    await _deepevalTracerProvider.forceFlush();
  }
}

export { DeepEvalSpanProcessor };
