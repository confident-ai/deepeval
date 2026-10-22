/**
 * Run confident-trace as deepeval's tracing engine.
 *
 * deepeval starts confident-trace itself. `DeepEvalSpanBridge` collects the
 * spans of traces deepeval consumes in-process, and confident-trace exports
 * the rest to Confident AI over OTLP under deepeval's API key.
 */

import { trace as otelTrace } from "@opentelemetry/api";
import type { Context } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import type {
  ReadableSpan,
  Span,
  SpanExporter,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { type ExportResult, ExportResultCode } from "@opentelemetry/core";

import { getSettings } from "@/config/settings";
import type { LLMTestCase } from "@/test-case";
import { ConfidentAttr } from "@/tracing/attributes";
import { traceManager } from "@/tracing/tracing";
import { tracingEnabled } from "@/tracing/utils";
import {
  DeepEvalSpanBridge,
  RemoteSpanExporter,
  traceFieldAttributes,
} from "@/tracing/otel/bridge";
import { type ExportedScope, otelHooks } from "@/tracing/otel/hooks";

type ConfidentTraceModule = typeof import("confident-trace");

export const INSTALL_HINT =
  "confident-trace is not installed. Install it with `npm install confident-trace` (Node.js 22+).";

// Above confident-trace's 16 KiB default so evaluated content such as
// retrieval context survives capture, while OTLP batches stay small.
const MAX_CONTENT_BYTES = 64 * 1024;

let started:
  | {
      module: ConfidentTraceModule;
      runtime: ReturnType<ConfidentTraceModule["init"]>;
      bridge: DeepEvalSpanBridge;
    }
  | undefined;

/** The confident-trace package, which deepeval's integrations need. */
export function confidentTrace(): ConfidentTraceModule {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require("confident-trace") as ConfidentTraceModule;
  } catch (error) {
    throw new Error(INSTALL_HINT, { cause: error });
  }
}

/** A confident-trace subpath, such as `confident-trace/openai`. */
export function confidentTraceModule<T>(subpath: string): T {
  confidentTrace();
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require(`confident-trace/${subpath}`) as T;
}

/**
 * Start confident-trace once; returns whether it is running.
 *
 * Integrations attach confident-trace's manual adapters themselves, so no
 * automatic instrumentation or preload is needed. `exporter` replaces the
 * Confident AI exporter, and only applies to the call that starts it.
 */
export function ensureConfidentTrace(
  options: { exporter?: SpanExporter } = {},
): boolean {
  if (started) return started.runtime.active;
  const module = confidentTrace();
  const bridge = new DeepEvalSpanBridge();
  const runtime = module.init({
    instrumentations: [],
    exporter: new RemoteSpanExporter(
      options.exporter ?? new ConfidentOtlpExporter(),
    ),
    spanProcessors: [new ScopeDefaults(), bridge],
    maxContentBytes: MAX_CONTENT_BYTES,
  });
  started = { module, runtime, bridge };
  if (!runtime.active) {
    console.warn(
      "[DeepEval] confident-trace couldn't start, so integrations won't trace. " +
        "Another OpenTelemetry tracer provider may already be registered.",
    );
    return false;
  }
  otelHooks.collectedSpan = () => bridge.spanFor(otelTrace.getActiveSpan());
  otelHooks.collectedTrace = () => {
    const span = otelHooks.collectedSpan?.();
    return span ? traceManager.getTraceByUuid(span.traceUuid) : undefined;
  };
  otelHooks.updateExported = (scope, values) => {
    if (bridge.isExported(otelTrace.getActiveSpan()))
      sendExported(module, scope, values);
  };
  otelHooks.flush = (timeoutMillis) => runtime.flush(timeoutMillis);
  return true;
}

// The attributes integrations start their spans with, by the integration
// label confident-trace stamps or by instrumentation scope prefix.
const scopeDefaults: {
  integration?: string;
  scope?: string;
  attributes: Record<string, string | string[]>;
  forSpan?: (span: Span) => Record<string, string>;
}[] = [];

/**
 * Start every span of an integration (by label) or instrumentation scope (by
 * prefix) with these trace fields and `attributes`.
 *
 * `updateCurrentTrace(...)` still overrides them.
 */
export function setTraceDefaults(
  match: { integration?: string; scope?: string },
  values: Record<string, unknown>,
  attributes: Record<string, string> = {},
  forSpan?: (span: Span) => Record<string, string>,
): void {
  const index = scopeDefaults.findIndex(
    (entry) =>
      entry.integration === match.integration && entry.scope === match.scope,
  );
  if (index >= 0) scopeDefaults.splice(index, 1);
  scopeDefaults.push({
    ...match,
    attributes: { ...traceFieldAttributes(values), ...attributes },
    forSpan,
  });
}

/** Stamps each span with its integration's defaults that it doesn't set. */
class ScopeDefaults implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    if (scopeDefaults.length === 0) return;
    const scope = span.instrumentationScope?.name ?? "";
    const integration = span.attributes[ConfidentAttr.SPAN_INTEGRATION];
    for (const entry of scopeDefaults) {
      const matches = entry.integration
        ? entry.integration === integration
        : entry.scope !== undefined && scope.startsWith(entry.scope);
      if (!matches) continue;
      const attributes = { ...entry.attributes, ...entry.forSpan?.(span) };
      for (const [key, value] of Object.entries(attributes)) {
        const current = span.attributes[key];
        // confident-trace names an integration's root after the span itself,
        // which a configured name replaces.
        const automatic =
          key === ConfidentAttr.TRACE_NAME && current === span.name;
        if (current === undefined || automatic) span.setAttribute(key, value);
      }
    }
  }
  onEnd(_span: ReadableSpan): void {}
  async forceFlush(): Promise<void> {}
  async shutdown(): Promise<void> {}
}

const EXPORTED_SPAN_FIELDS = [
  "name",
  "input",
  "output",
  "metadata",
  "retrievalContext",
  "context",
  "expectedOutput",
  "toolsCalled",
  "expectedTools",
  "metricCollection",
];
const EXPORTED_TRACE_FIELDS = [
  ...EXPORTED_SPAN_FIELDS,
  "tags",
  "userId",
  "threadId",
  "turnId",
  "testCaseId",
];

/** Pass an update to confident-trace for the exported span or trace. */
function sendExported(
  module: ConfidentTraceModule,
  scope: ExportedScope,
  values: Record<string, any>,
): void {
  const { testCase, ...rest } = values as { testCase?: LLMTestCase };
  const unpacked: Record<string, any> = testCase
    ? {
        input: testCase.input,
        output: testCase.actualOutput,
        expectedOutput: testCase.expectedOutput,
        retrievalContext: testCase.retrievalContext,
        context: testCase.context,
        toolsCalled: testCase.toolsCalled,
        expectedTools: testCase.expectedTools,
      }
    : {};
  const accepted =
    scope === "span" ? EXPORTED_SPAN_FIELDS : EXPORTED_TRACE_FIELDS;
  const fields: Record<string, any> = {};
  for (const [key, value] of Object.entries({ ...unpacked, ...rest }))
    if (value !== undefined && accepted.includes(key)) fields[key] = value;
  if (Object.keys(fields).length === 0) return;
  try {
    if (scope === "span") module.updateSpan(fields);
    else module.updateTrace(fields);
  } catch {
    // confident-trace rejects values it can't carry; user code goes on.
  }
}

/** Wait up to `timeoutMillis` for confident-trace to export its spans. */
export function flushConfidentTrace(timeoutMillis: number): Promise<boolean> {
  return otelHooks.flush?.(timeoutMillis) ?? Promise.resolve(true);
}

/** Stop confident-trace. Tests only: confident-trace can't start again. */
export async function shutdownConfidentTrace(): Promise<void> {
  await started?.runtime.shutdown();
}

/**
 * OTLP export to Confident AI under the API key resolved at export time.
 *
 * Resolving per export lets a key set after start-up take effect. As with
 * trace posting, nothing is sent while tracing is off or no key is set.
 */
export class ConfidentOtlpExporter implements SpanExporter {
  private exporter?: { key: string; exporter: SpanExporter };

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    const apiKey = traceManager.exportApiKey;
    if (!tracingEnabled() || !apiKey)
      return resultCallback({ code: ExportResultCode.SUCCESS });
    this.exporterFor(apiKey).export(spans, resultCallback);
  }

  async forceFlush(): Promise<void> {
    await this.exporter?.exporter.forceFlush?.();
  }

  async shutdown(): Promise<void> {
    await this.exporter?.exporter.shutdown();
  }

  private exporterFor(apiKey: string): SpanExporter {
    if (this.exporter?.key !== apiKey) {
      void this.exporter?.exporter.shutdown();
      const base = getSettings().CONFIDENT_OTEL_URL.replace(/\/+$/, "");
      this.exporter = {
        key: apiKey,
        exporter: new OTLPTraceExporter({
          url: `${base}/v1/traces`,
          headers: { "x-confident-api-key": apiKey },
        }),
      };
    }
    return this.exporter.exporter;
  }
}
