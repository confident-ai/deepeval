/**
 * Run confident-trace as deepeval's tracing engine.
 *
 * deepeval starts confident-trace itself. `DeepEvalSpanBridge` collects the
 * spans of traces deepeval consumes in-process, and confident-trace exports
 * the rest to Confident AI over OTLP under deepeval's API key.
 */

import {
  context as otelContext,
  createContextKey,
  trace as otelTrace,
} from "@opentelemetry/api";
import type { Context, Span as OtelSpan } from "@opentelemetry/api";
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
import type { Prompt } from "@/prompt";
import { ConfidentAttr } from "@/tracing/attributes";
import { type StagedValues, withStaged } from "@/tracing/pending-context";
import { type BaseSpan, traceManager } from "@/tracing/tracing";
import { tracingEnabled } from "@/tracing/utils";
import {
  DeepEvalSpanBridge,
  OBSERVE_SCOPE,
  RemoteSpanExporter,
  dropFromExport,
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
  otelHooks.drop = dropExported;
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

/**
 * Start an `observe` span under the current one, not made active, with
 * `staged` applied to the deepeval span it becomes.
 *
 * Returns nothing when confident-trace isn't running; the placeholder only
 * when deepeval collects the span.
 */
export function startObservedSpan(
  name: string,
  spanType: string,
  staged: StagedValues,
): { span: OtelSpan; placeholder?: BaseSpan } | undefined {
  if (!ensureConfidentTrace() || !started) return undefined;
  const bridge = started.bridge;
  const span = withStaged(staged, () =>
    otelTrace
      .getTracer(OBSERVE_SCOPE)
      .startSpan(name, { attributes: { [ConfidentAttr.SPAN_TYPE]: spanType } }),
  );
  return { span, placeholder: bridge.spanFor(span) };
}

// confident-trace's marker for a trace's entry, whose span takes the trace's
// own fields. OTel context keys are global symbols, so this is the same key.
const ENTRY_KEY = createContextKey("confident-trace.entry-operation");

/**
 * The active context with `span` active, and the trace's entry when no span
 * opened under it is yet, so a confident-trace span inside `observe` doesn't
 * claim the trace.
 */
export function observedContext(span: OtelSpan): Context {
  const active = otelContext.active();
  const ctx = otelTrace.setSpan(active, span);
  return active.getValue(ENTRY_KEY) ? ctx : ctx.setValue(ENTRY_KEY, { span });
}

/** Export span or trace `values` on `span`, if confident-trace exports it. */
export function exportOn(
  span: OtelSpan,
  scope: ExportedScope,
  values: Record<string, unknown>,
): void {
  otelContext.with(otelTrace.setSpan(otelContext.active(), span), () =>
    otelHooks.updateExported?.(scope, values),
  );
}

/** Whether `span` already carries `attribute`. */
export function carries(span: OtelSpan, attribute: string): boolean {
  const attributes = (
    span as unknown as { attributes?: Record<string, unknown> }
  ).attributes;
  return attributes?.[attribute] !== undefined;
}

/** Keep the active span, or its whole trace, out of the export. */
export function dropExported(wholeTrace: boolean): void {
  dropFromExport(otelTrace.getActiveSpan(), wholeTrace);
}

/** Run `fn` with trace fields every span that starts a trace in it exports. */
export function exportedTraceContext<T>(
  values: Record<string, unknown>,
  fn: () => T,
): T {
  if (!ensureConfidentTrace() || !started) return fn();
  const fields = exportable(values, EXPORTED_TRACE_FIELDS);
  if (Object.keys(fields).length === 0) return fn();
  try {
    return started.module.traceContext(fields, fn);
  } catch (error) {
    // confident-trace validates the fields before running `fn`.
    if (error instanceof TypeError) return fn();
    throw error;
  }
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

const CONTENT_FIELDS = [
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
const EXPORTED_SPAN_FIELDS = [
  ...CONTENT_FIELDS,
  "model",
  "provider",
  "inputTokenCount",
  "outputTokenCount",
  "costPerInputToken",
  "costPerOutputToken",
];
const EXPORTED_TRACE_FIELDS = [
  ...CONTENT_FIELDS,
  "tags",
  "userId",
  "threadId",
  "turnId",
  "testCaseId",
];
const TOOL_FIELDS = ["toolsCalled", "expectedTools"];
const COUNT_FIELDS = ["inputTokenCount", "outputTokenCount"];

// Span fields the platform reads under deepeval's own attributes, which
// confident-trace has none for.
const PLATFORM_SPAN_ATTRIBUTES: Record<string, string> = {
  availableTools: ConfidentAttr.AGENT_AVAILABLE_TOOLS,
  agentHandoffs: ConfidentAttr.AGENT_AGENT_HANDOFFS,
  description: ConfidentAttr.TOOL_DESCRIPTION,
  embedder: ConfidentAttr.RETRIEVER_EMBEDDER,
  topK: ConfidentAttr.RETRIEVER_TOP_K,
  chunkSize: ConfidentAttr.RETRIEVER_CHUNK_SIZE,
  integration: ConfidentAttr.SPAN_INTEGRATION,
};
const PROMPT_ATTRIBUTES: Record<string, string> = {
  _alias: ConfidentAttr.SPAN_PROMPT_ALIAS,
  version: ConfidentAttr.SPAN_PROMPT_VERSION,
  label: ConfidentAttr.SPAN_PROMPT_LABEL,
  hash: ConfidentAttr.SPAN_PROMPT_COMMIT_HASH,
};

/** `values` as confident-trace takes them, limited to `accepted`. */
function exportable(
  values: Record<string, any>,
  accepted: string[],
): Record<string, any> {
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
  const fields: Record<string, any> = {};
  for (const [key, value] of Object.entries({ ...unpacked, ...rest })) {
    if (value === undefined || value === null || !accepted.includes(key))
      continue;
    if (TOOL_FIELDS.includes(key))
      // Plain objects, since confident-trace doesn't serialize class instances.
      fields[key] = (value as unknown[]).map((call) =>
        JSON.parse(JSON.stringify(call)),
      );
    else if (COUNT_FIELDS.includes(key))
      fields[key] = Math.trunc(Number(value));
    else fields[key] = value;
  }
  return fields;
}

function platformSpanAttributes(
  values: Record<string, any>,
): Record<string, string | number | string[]> {
  const attributes: Record<string, string | number | string[]> = {};
  for (const [key, attribute] of Object.entries(PLATFORM_SPAN_ATTRIBUTES)) {
    const value = values[key];
    if (Array.isArray(value)) attributes[attribute] = value.map(String);
    else if (typeof value === "string" || Number.isFinite(value))
      attributes[attribute] = value;
  }
  const prompt = values.prompt as Prompt | undefined;
  for (const [key, attribute] of Object.entries(PROMPT_ATTRIBUTES)) {
    const value = prompt?.[key as keyof Prompt];
    if (typeof value === "string") attributes[attribute] = value;
  }
  return attributes;
}

// The Confident AI key a trace exports under, when it isn't the default.
const MAX_ROUTED_TRACES = 10_000;
const traceKeys = new Map<string, string>();

function routeTrace(span: OtelSpan, apiKey: string): void {
  traceKeys.set(span.spanContext().traceId, apiKey);
  if (traceKeys.size > MAX_ROUTED_TRACES)
    traceKeys.delete(traceKeys.keys().next().value!);
}

/** Pass an update to confident-trace for the exported span or trace. */
function sendExported(
  module: ConfidentTraceModule,
  scope: ExportedScope,
  values: Record<string, any>,
): void {
  const span = otelTrace.getActiveSpan();
  if (!span) return;
  const fields = exportable(
    values,
    scope === "span" ? EXPORTED_SPAN_FIELDS : EXPORTED_TRACE_FIELDS,
  );
  try {
    if (Object.keys(fields).length > 0) {
      if (scope === "span") module.updateSpan(fields);
      else module.updateTrace(fields);
    }
  } catch {
    // confident-trace rejects values it can't carry; user code goes on.
  }
  if (scope === "span") {
    span.setAttributes(platformSpanAttributes(values));
    return;
  }
  if (typeof values.testRunId === "string")
    span.setAttribute(ConfidentAttr.TRACE_TEST_RUN_ID, values.testRunId);
  if (typeof values.confidentApiKey === "string")
    routeTrace(span, values.confidentApiKey);
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
 * Resolving per export lets a key set after start-up take effect, and a trace
 * may carry its own. Nothing is sent while tracing is off or for a trace with
 * no key.
 */
export class ConfidentOtlpExporter implements SpanExporter {
  private readonly exporters = new Map<string, SpanExporter>();

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (!tracingEnabled() || !traceManager.exportEnabled)
      return resultCallback({ code: ExportResultCode.SUCCESS });
    const defaultKey = traceManager.exportApiKey;
    const byKey = new Map<string, ReadableSpan[]>();
    for (const span of spans) {
      const apiKey = traceKeys.get(span.spanContext().traceId) ?? defaultKey;
      if (apiKey) byKey.set(apiKey, [...(byKey.get(apiKey) ?? []), span]);
    }
    if (byKey.size === 0)
      return resultCallback({ code: ExportResultCode.SUCCESS });
    let pending = byKey.size;
    let failed: ExportResult | undefined;
    for (const [apiKey, keyed] of byKey)
      this.exporterFor(apiKey).export(keyed, (result) => {
        if (result.code !== ExportResultCode.SUCCESS) failed ??= result;
        if (--pending === 0)
          resultCallback(failed ?? { code: ExportResultCode.SUCCESS });
      });
  }

  async forceFlush(): Promise<void> {
    await Promise.all(
      [...this.exporters.values()].map((exporter) => exporter.forceFlush?.()),
    );
  }

  async shutdown(): Promise<void> {
    await Promise.all(
      [...this.exporters.values()].map((exporter) => exporter.shutdown()),
    );
    this.exporters.clear();
  }

  private exporterFor(apiKey: string): SpanExporter {
    let exporter = this.exporters.get(apiKey);
    if (!exporter) {
      const base = getSettings().CONFIDENT_OTEL_URL.replace(/\/+$/, "");
      exporter = new OTLPTraceExporter({
        url: `${base}/v1/traces`,
        headers: { "x-confident-api-key": apiKey },
      });
      this.exporters.set(apiKey, exporter);
    }
    return exporter;
  }
}
