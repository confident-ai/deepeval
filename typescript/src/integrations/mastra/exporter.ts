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
import {
  type BaseSpan,
  type LlmSpan,
  SpanType,
  type Trace,
  type TraceManagerConfig,
  traceManager,
} from "@/tracing/tracing";
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
  apiKey?: string;
  environment?: string;
  name?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  turnId?: string;
  metricCollection?: string;
  traceMetricCollection?: string;
  llmMetricCollection?: string;
  agentMetricCollection?: string;
  toolMetricCollectionMap?: Record<string, string>;
  prompt?: Prompt;
  /** @deprecated No longer used. */
  debug?: boolean;
  traceCaptureSink?: (trace: Trace) => void;
}

// Traces still waiting for their root span, at most.
const MAX_PENDING_TRACES = 1024;

// Per-request `tracingOptions.metadata` keys read as trace fields.
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

  const userId = meta.userId ?? meta.resourceId;
  if (userId) fields.userId = String(userId);

  const tags = attrStringList(attributes, ConfidentAttr.TRACE_TAGS);
  if (tags && tags.length > 0) fields.tags = tags;
  if (meta.traceName) fields.name = String(meta.traceName);
  if (meta.testCaseId) fields.testCaseId = String(meta.testCaseId);
  if (meta.turnId) fields.turnId = String(meta.turnId);

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
    const managed: TraceManagerConfig = {};
    if (config.apiKey) managed.confidentApiKey = config.apiKey;
    if (config.environment)
      managed.environment = config.environment as Environment;
    if (Object.keys(managed).length > 0) traceManager.configure(managed);
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
      userId: config.userId,
      testCaseId: config.testCaseId,
      turnId: config.turnId,
      metricCollection: config.traceMetricCollection ?? config.metricCollection,
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

  private metricCollectionFor(
    type: string | undefined,
    name: string | undefined,
  ): string | undefined {
    const { config } = this;
    if (type === SpanType.LLM) return config.llmMetricCollection;
    if (type === SpanType.AGENT) return config.agentMetricCollection;
    if (type === SpanType.TOOL && name)
      return config.toolMetricCollectionMap?.[name];
    return undefined;
  }

  private collect(spans: ReadableSpan[], fields: Partial<Trace>): void {
    collectFinishedTrace(spans, fields, (span: BaseSpan) => {
      span.metricCollection ??= this.metricCollectionFor(span.type, span.name);
      const prompt = this.config.prompt;
      if (span.type === SpanType.LLM && prompt) {
        const llm = span as LlmSpan;
        llm.prompt = prompt;
        llm.promptAlias = prompt._alias ?? undefined;
        llm.promptCommitHash = prompt.hash ?? undefined;
        llm.promptVersion = prompt.version ?? undefined;
        llm.promptLabel = prompt.label ?? undefined;
      }
    });
  }

  /**
   * `span` with this exporter's settings as the attributes that carry them:
   * span settings fill what the span leaves unset, and trace settings, the
   * run's own included, replace what Mastra recorded for the trace.
   */
  private withSettings(
    span: ReadableSpan,
    fields: Partial<Trace>,
  ): ReadableSpan {
    const attributes: Record<string, unknown> = {};
    const trace =
      !span.parentSpanContext || span.parentSpanContext.isRemote
        ? traceFieldAttributes(fields)
        : {};
    const type = span.attributes[ConfidentAttr.SPAN_TYPE];
    const toolName = span.attributes[GenAIAttr.TOOL_NAME];
    const collection = this.metricCollectionFor(
      typeof type === "string" ? type : undefined,
      typeof toolName === "string" ? toolName : span.name,
    );
    if (collection)
      attributes[ConfidentAttr.SPAN_METRIC_COLLECTION] = collection;
    const prompt = this.config.prompt;
    if (type === SpanType.LLM && prompt) {
      if (prompt._alias)
        attributes[ConfidentAttr.SPAN_PROMPT_ALIAS] = prompt._alias;
      if (prompt.hash)
        attributes[ConfidentAttr.SPAN_PROMPT_COMMIT_HASH] = prompt.hash;
      if (prompt.label)
        attributes[ConfidentAttr.SPAN_PROMPT_LABEL] = prompt.label;
      if (prompt.version)
        attributes[ConfidentAttr.SPAN_PROMPT_VERSION] = prompt.version;
    }
    for (const key of Object.keys(attributes))
      if (span.attributes[key] !== undefined) delete attributes[key];
    if (Object.keys(attributes).length + Object.keys(trace).length === 0)
      return span;
    return Object.create(span, {
      attributes: { value: { ...attributes, ...span.attributes, ...trace } },
    }) as ReadableSpan;
  }
}
