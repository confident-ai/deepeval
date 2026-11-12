/**
 * deepeval's Mastra integration, built on confident-trace.
 *
 * `DeepEvalExporter` is confident-trace's Mastra exporter: register it with
 * Mastra's observability. Mastra reports spans once they end, so during an
 * evaluation the exporter builds each finished trace locally for deepeval;
 * otherwise it sends the spans to Confident AI.
 */

import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { type ExportResult, ExportResultCode } from "@opentelemetry/core";

import type { Prompt } from "@/prompt";
import { recordTracingIntegration } from "@/telemetry";
import { ConfidentAttr } from "@/tracing/attributes";
import { Integration } from "@/tracing/integrations";
import { type Trace, traceManager } from "@/tracing/tracing";
import type { Environment } from "@/tracing/utils";
import {
  attrJsonObject,
  attrStringList,
} from "@/tracing/otel/attribute-values";
import {
  collectFinishedTrace,
  traceFieldAttributes,
} from "@/tracing/otel/bridge";
import { GenAIAttr } from "@/tracing/otel/genai";
import {
  ConfidentOtlpExporter,
  confidentTraceModule,
  ensureConfidentTrace,
} from "@/tracing/otel/setup";

type MastraAdapter = typeof import("confident-trace/mastra");

const { ConfidentMastraExporter } =
  confidentTraceModule<MastraAdapter>("mastra");

export interface DeepEvalExporterConfig {
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  apiKey?: string;
  environment?: string;
  name?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  threadId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  userId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  testCaseId?: string;
  /** @deprecated Ignored: a Confident AI platform setting. Don't build on it. */
  turnId?: string;
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
  /** @deprecated No longer used. */
  debug?: boolean;
  traceCaptureSink?: (trace: Trace) => void;
}

// Traces still waiting for their root span, at most.
const MAX_PENDING_TRACES = 1024;

// Per-request `tracingOptions.metadata` keys read as trace fields. `userId`,
// `resourceId`, `testCaseId` and `turnId` are deprecated platform settings:
// dropped, not kept as metadata.
const RESERVED_TRACE_META_KEYS = new Set<string>([
  "userId",
  "threadId",
  "sessionId",
  "resourceId",
  "traceName",
  "testCaseId",
  "turnId",
]);

/** What one run's `tracingOptions` set for its trace, from its root span. */
function perRequestFields(root: ReadableSpan): Partial<Trace> {
  const attributes = root.attributes;
  const meta = attrJsonObject(attributes, ConfidentAttr.TRACE_METADATA) ?? {};
  const fields: Partial<Trace> = {};

  const threadId =
    meta.threadId ?? meta.sessionId ?? attributes[GenAIAttr.CONVERSATION_ID];
  if (threadId) fields.threadId = String(threadId);

  const tags = attrStringList(attributes, ConfidentAttr.TRACE_TAGS);
  if (tags && tags.length > 0) fields.tags = tags;
  if (meta.traceName) fields.name = String(meta.traceName);

  const custom: Record<string, any> = {};
  for (const [key, value] of Object.entries(meta)) {
    if (!RESERVED_TRACE_META_KEYS.has(key) && value !== undefined) {
      custom[key] = value;
    }
  }
  if (Object.keys(custom).length > 0) fields.metadata = custom;

  return fields;
}

export class DeepEvalExporter extends ConfidentMastraExporter {
  private mastra?: { observability?: { flush?: () => Promise<void> } };
  private readonly config: DeepEvalExporterConfig;

  constructor(config: DeepEvalExporterConfig = {}) {
    // Package content settings, such as the larger content cap, apply.
    ensureConfidentTrace();
    super({ exporter: new FinishedTraces(config) });
    this.config = config;
    if (config.environment)
      traceManager.configure({
        environment: config.environment as Environment,
      });
    if (config.traceCaptureSink)
      traceManager.addTraceCaptureSink(config.traceCaptureSink);
    // Mastra delivers spans on its own bus, after `await agent.generate(...)`
    // may have returned. Evaluations settle on Mastra's flush first.
    traceManager.addSettleHook(async () => {
      await (this.mastra?.observability?.flush?.() ?? this.flush());
    });
    recordTracingIntegration(Integration.MASTRA);
  }

  override init(options: {
    config?: { serviceName?: string };
    mastra?: { observability?: { flush?: () => Promise<void> } };
  }): void {
    super.init(options);
    this.mastra = options.mastra;
    this.config.name ??= options.config?.serviceName;
  }
}

/** Groups Mastra's finished spans by trace until the root span ends. */
class FinishedTraces implements SpanExporter {
  private readonly pending = new Map<string, ReadableSpan[]>();
  private readonly remote = new ConfidentOtlpExporter();

  constructor(private readonly config: DeepEvalExporterConfig) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    const exported: ReadableSpan[] = [];
    for (const span of spans) {
      const traceId = span.spanContext().traceId;
      const trace = this.pending.get(traceId) ?? [];
      trace.push(span);
      this.pending.set(traceId, trace);
      const parent = span.parentSpanContext;
      if (parent && !parent.isRemote) continue;
      this.pending.delete(traceId);
      const fields = this.traceFields(span);
      if (traceManager.isCapturingTraces) this.collect(trace, fields);
      else
        exported.push(...trace.map((item) => this.withSettings(item, fields)));
    }
    while (this.pending.size > MAX_PENDING_TRACES)
      this.pending.delete(this.pending.keys().next().value!);
    if (exported.length === 0)
      return resultCallback({ code: ExportResultCode.SUCCESS });
    this.remote.export(exported, resultCallback);
  }

  async forceFlush(): Promise<void> {
    await this.remote.forceFlush();
  }

  async shutdown(): Promise<void> {
    await this.remote.shutdown();
  }

  /** The exporter's trace settings, which a run's `tracingOptions` override. */
  private traceFields(root: ReadableSpan): Partial<Trace> {
    const { config } = this;
    const request = perRequestFields(root);
    const fields: Partial<Trace> = {
      name: config.name,
      tags: config.tags,
      threadId: config.threadId,
      ...request,
      metadata:
        config.metadata || request.metadata
          ? { ...(config.metadata ?? {}), ...(request.metadata ?? {}) }
          : undefined,
    };
    for (const key of Object.keys(fields) as (keyof Trace)[])
      if (fields[key] === undefined) delete fields[key];
    return fields;
  }

  private collect(spans: ReadableSpan[], fields: Partial<Trace>): void {
    collectFinishedTrace(spans, fields);
  }

  /**
   * `span` with this exporter's trace settings, the run's own included, as the
   * attributes that carry them, replacing what Mastra recorded for the trace.
   */
  private withSettings(
    span: ReadableSpan,
    fields: Partial<Trace>,
  ): ReadableSpan {
    if (span.parentSpanContext && !span.parentSpanContext.isRemote) return span;
    const trace = traceFieldAttributes(fields);
    if (Object.keys(trace).length === 0) return span;
    return Object.create(span, {
      attributes: { value: { ...span.attributes, ...trace } },
    }) as ReadableSpan;
  }
}
