import { traceManager } from "../../tracing";
import { SpanType, TraceManagerConfig, Trace } from "../../tracing/tracing";
import { Environment } from "../../tracing/utils";
import { getConfidentApiKey, isConfident } from "../../utils";
import { withCaptureTracingIntegration } from "../../telemetry";
import { Prompt } from "../../prompt";

import { TracingEventType } from "@mastra/core/observability";
import type {
  AnyExportedSpan,
  InitExporterOptions,
  ObservabilityExporter,
  TracingEvent,
} from "@mastra/core/observability";

import {
  buildDeepEvalSpan,
  buildToolCall,
  extractTraceContext,
  finalizeDeepEvalSpan,
  getToolName,
  mapSpanType,
  shouldDropSpan,
  updateDeepEvalSpan,
} from "./converter";

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
  debug?: boolean;
  traceCaptureSink?: (trace: Trace) => void;
}

export class DeepEvalExporter implements ObservabilityExporter {
  name = "deepeval";

  private config: DeepEvalExporterConfig;
  private disabled = false;

  private traceIds = new Map<string, string>();

  constructor(config: DeepEvalExporterConfig = {}) {
    this.config = config;

    const apiKey = config.apiKey ?? getConfidentApiKey() ?? undefined;
    if (!apiKey && !isConfident()) {
      this.disabled = true;
      console.warn(
        "DeepEval: No API Key found. Mastra tracing will be disabled.",
      );
      return;
    }

    const tmConfig: TraceManagerConfig = {};
    if (apiKey) tmConfig.confidentApiKey = apiKey;
    if (config.environment)
      tmConfig.environment = config.environment as Environment;
    if (Object.keys(tmConfig).length > 0) {
      traceManager.configure(tmConfig);
    }

    if (config.traceCaptureSink) {
      traceManager.setTraceCaptureSink(config.traceCaptureSink);
    }

    withCaptureTracingIntegration("mastra", () => {}).catch((err) => {
      if (config.debug) console.error("DeepEval telemetry failed:", err);
    });

    if (config.debug) {
      console.log("DeepEval Mastra exporter configured", {
        environment: config.environment,
        name: config.name,
      });
    }
  }

  init(options: InitExporterOptions): void {
    if (!this.config.name && options.config?.serviceName) {
      this.config.name = options.config.serviceName;
    }
  }

  async exportTracingEvent(event: TracingEvent): Promise<void> {
    if (this.disabled) return;

    const span = event.exportedSpan;
    if (!span) return;

    if (shouldDropSpan(span)) return;

    try {
      switch (event.type) {
        case TracingEventType.SPAN_STARTED:
          this.handleStart(span);
          break;
        case TracingEventType.SPAN_UPDATED:
          this.handleUpdate(span);
          break;
        case TracingEventType.SPAN_ENDED:
          this.handleEnd(span);
          break;
      }
    } catch (err) {
      if (this.config.debug) {
        console.error(`DeepEval: failed to handle ${event.type}`, err);
      }
    }
  }

  private handleStart(span: AnyExportedSpan): void {
    const traceUuid = this.ensureTrace(span.traceId);

    if (span.isRootSpan) {
      const trace = traceManager.getTraceByUuid(traceUuid);
      if (trace) this.applyPerRequestContext(trace, span);
    }

    const deepEvalSpan = buildDeepEvalSpan(span, traceUuid, {
      metricCollection: this.resolveSpanMetricCollection(span),
      prompt:
        mapSpanType(span.type) === SpanType.LLM
          ? this.config.prompt
          : undefined,
    });

    traceManager.addSpan(deepEvalSpan);
    try {
      traceManager.addSpanToTrace(deepEvalSpan);
    } catch {
      deepEvalSpan.parentUuid = undefined;
      try {
        traceManager.addSpanToTrace(deepEvalSpan);
      } catch {
        traceManager.removeSpan(deepEvalSpan.uuid);
      }
    }
  }

  private handleUpdate(span: AnyExportedSpan): void {
    const existing = traceManager.getSpanByUuid(span.id);
    if (existing) updateDeepEvalSpan(existing, span);
  }

  private handleEnd(span: AnyExportedSpan): void {
    const existing = traceManager.getSpanByUuid(span.id);
    if (!existing) return;

    finalizeDeepEvalSpan(existing, span);
    const traceUuid = existing.traceUuid;

    if (mapSpanType(span.type) === SpanType.TOOL) {
      const trace = traceManager.getTraceByUuid(traceUuid);
      if (trace) {
        if (!trace.toolsCalled) trace.toolsCalled = [];
        trace.toolsCalled.push(buildToolCall(span));
      }
    }

    if (span.isRootSpan) {
      const trace = traceManager.getTraceByUuid(traceUuid);
      if (trace) {
        if (trace.input === undefined && existing.input !== undefined)
          trace.input = existing.input;
        if (existing.output !== undefined) trace.output = existing.output;
        if (!trace.name && existing.name) trace.name = existing.name;
      }
    }

    traceManager.removeSpan(span.id);

    const stillActive = Array.from(traceManager.getActiveSpans().values()).some(
      (s) => s.traceUuid === traceUuid,
    );
    if (!stillActive) {
      this.traceIds.delete(span.traceId);
      traceManager.endTrace(traceUuid);
    }
  }

  private ensureTrace(mastraTraceId: string): string {
    let traceUuid = this.traceIds.get(mastraTraceId);
    if (traceUuid) return traceUuid;

    const trace = traceManager.startNewTrace();
    traceUuid = trace.uuid;
    this.traceIds.set(mastraTraceId, traceUuid);
    this.stampTrace(trace);
    return traceUuid;
  }

  private stampTrace(trace: Trace): void {
    const c = this.config;
    if (c.name) trace.name = c.name;
    if (c.tags) trace.tags = c.tags;
    if (c.metadata) trace.metadata = c.metadata;
    if (c.threadId) trace.threadId = c.threadId;
    if (c.userId) trace.userId = c.userId;
    if (c.testCaseId) trace.testCaseId = c.testCaseId;
    if (c.turnId) trace.turnId = c.turnId;
    const traceMetricCollection = c.traceMetricCollection ?? c.metricCollection;
    if (traceMetricCollection) trace.metricCollection = traceMetricCollection;
  }

  private applyPerRequestContext(trace: Trace, span: AnyExportedSpan): void {
    const ctx = extractTraceContext(span);
    if (ctx.threadId) trace.threadId = ctx.threadId;
    if (ctx.userId) trace.userId = ctx.userId;
    if (ctx.tags) trace.tags = ctx.tags;
    if (ctx.name) trace.name = ctx.name;
    if (ctx.testCaseId) trace.testCaseId = ctx.testCaseId;
    if (ctx.turnId) trace.turnId = ctx.turnId;
    if (ctx.metadata) {
      trace.metadata = { ...(trace.metadata ?? {}), ...ctx.metadata };
    }
  }

  private resolveSpanMetricCollection(
    span: AnyExportedSpan,
  ): string | undefined {
    switch (mapSpanType(span.type)) {
      case SpanType.LLM:
        return this.config.llmMetricCollection;
      case SpanType.AGENT:
        return this.config.agentMetricCollection;
      case SpanType.TOOL:
        return this.config.toolMetricCollectionMap?.[getToolName(span)];
      default:
        return undefined;
    }
  }

  async flush(): Promise<void> {
    if (this.disabled) return;
    await traceManager.flush();
  }

  async shutdown(): Promise<void> {
    if (this.disabled) return;

    for (const traceUuid of this.traceIds.values()) {
      if (traceManager.getTraceByUuid(traceUuid)) {
        try {
          traceManager.endTrace(traceUuid);
        } catch {
          // best effort
        }
      }
    }
    this.traceIds.clear();
    await traceManager.flush();
  }
}
