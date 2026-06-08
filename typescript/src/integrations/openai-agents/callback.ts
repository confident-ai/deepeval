import {
  traceManager,
  BaseSpan,
  getCurrentTrace,
  TraceSpanStatus,
  SpanType,
  AgentSpan,
  LlmSpan,
  ToolSpan,
} from "../../tracing/tracing";
import { getAgentContext, getLlmContext } from "../../tracing/trace-context";
import {
  updateSpanProperties,
  updateTracePropertiesFromSpanData,
} from "./extractors";

export class DeepEvalTracingProcessor {
  private activeSpans: Map<string, BaseSpan> = new Map();
  private traceIdMapping: Map<string, string> = new Map();

  public async onTraceStart(trace: any): Promise<void> {
    const traceDict = trace.export ? trace.export() : trace;
    const sdkTraceId = traceDict.trace_id || traceDict.traceId || traceDict.id;
    const traceName =
      traceDict.workflow_name || traceDict.workflowName || traceDict.name;

    const currentTrace = getCurrentTrace();

    if (currentTrace) {
      this.traceIdMapping.set(sdkTraceId, currentTrace.uuid);
      if (
        traceName &&
        (!currentTrace.name || currentTrace.name === "Agent workflow")
      ) {
        currentTrace.name = String(traceName);
      }
    } else {
      const newTrace = traceManager.startNewTrace();
      newTrace.name = traceName ? String(traceName) : "Agent workflow";

      this.traceIdMapping.set(sdkTraceId, newTrace.uuid);
    }
  }

  public async onTraceEnd(trace: any): Promise<void> {
    const traceDict = trace.export ? trace.export() : trace;
    const sdkTraceId = traceDict.trace_id || traceDict.traceId || traceDict.id;

    // Seal and push the trace
    const newTraceUuid = this.traceIdMapping.get(sdkTraceId);
    if (newTraceUuid) {
      traceManager.endTrace(newTraceUuid);
      this.traceIdMapping.delete(sdkTraceId);
    }
  }

  public async onSpanStart(span: any): Promise<void> {
    const spanData = span.span_data || span.spanData;
    const spanType = this.getSpanKind(spanData);

    // Try all common OpenTelemetry parent ID keys
    const parentSpanId =
      span.parent_span_id ||
      span.parentSpanId ||
      span.parent_id ||
      span.parentId;

    // Prevent nested LLM spans (double counting tokens)
    const parentSpan = parentSpanId
      ? traceManager.getSpanByUuid(parentSpanId)
      : undefined;
    if (parentSpan && parentSpan.type === SpanType.LLM) {
      return;
    }

    // Extract Context variables
    let metricCollection: string | undefined = undefined;
    if (spanType === SpanType.AGENT) {
      metricCollection = getAgentContext()?.metricCollection;
    } else if (spanType === SpanType.LLM) {
      metricCollection = getLlmContext()?.metricCollection;
    } else if (spanType === SpanType.TOOL) {
      metricCollection = getLlmContext()?.toolsMetricCollection;
    }

    const sdkTraceId = span.trace_id || span.traceId;
    const newTraceUuid = this.traceIdMapping.get(sdkTraceId);
    const spanId = span.span_id || span.spanId;

    if (!newTraceUuid) return;

    const baseParams = {
      uuid: spanId,
      traceUuid: newTraceUuid,
      parentUuid: parentSpanId || undefined,
      startTime: new Date(),
      name: spanData?.name || "NA",
      status: TraceSpanStatus.IN_PROGRESS,
      type: spanType,
      metricCollection,
    };

    // Natively instantiate the specific Span classes
    let newSpan: BaseSpan;
    if (spanType === SpanType.AGENT) {
      newSpan = new AgentSpan({
        ...baseParams,
        availableTools: [],
        agentHandoffs: [],
      });
    } else if (spanType === SpanType.LLM) {
      newSpan = new LlmSpan({ ...baseParams, model: "temporary model" });
    } else if (spanType === SpanType.TOOL) {
      newSpan = new ToolSpan({ ...baseParams } as any);
    } else {
      newSpan = new BaseSpan(baseParams);
    }

    traceManager.addSpan(newSpan);
    traceManager.addSpanToTrace(newSpan);
    this.activeSpans.set(spanId, newSpan);
  }

  public async onSpanEnd(span: any): Promise<void> {
    const spanId = span.span_id || span.spanId;
    const deSpan = this.activeSpans.get(spanId);

    if (!deSpan) return;

    const spanData = span.span_data || span.spanData;

    updateSpanProperties(deSpan, spanData);

    if (deSpan.type === SpanType.LLM) {
      const llmCtx = getLlmContext();
      if (llmCtx?.prompt) {
        (deSpan as any).prompt = llmCtx.prompt;
        (deSpan as any).promptAlias = (llmCtx.prompt as any)._alias;
        (deSpan as any).promptCommitHash = llmCtx.prompt.hash;
        (deSpan as any).promptVersion = llmCtx.prompt.version;
        (deSpan as any).promptLabel = llmCtx.prompt.label;
      }
    }

    const sdkTraceId = span.trace_id || span.traceId;
    const newTraceUuid = this.traceIdMapping.get(sdkTraceId);
    if (newTraceUuid) {
      const trace = traceManager.getTraceByUuid(newTraceUuid);
      if (trace) {
        updateTracePropertiesFromSpanData(trace, spanData);
      }
    }

    // Backfill agent span input/output from trace after trace has been updated
    if (deSpan.type === SpanType.AGENT) {
      const agentTraceUuid = this.traceIdMapping.get(sdkTraceId);
      if (agentTraceUuid) {
        const trace = traceManager.getTraceByUuid(agentTraceUuid);
        if (trace) {
          if (!deSpan.input && trace.input) deSpan.input = trace.input;
          deSpan.output = trace.output;
        }
      }
    }

    deSpan.endTime = new Date();
    deSpan.status = TraceSpanStatus.SUCCESS;

    traceManager.updateSpanInTrace(deSpan);
    traceManager.removeSpan(deSpan.uuid);
    this.activeSpans.delete(spanId);
  }

  private getSpanKind(spanData: any): SpanType | string {
    if (!spanData || !spanData.type) return "base";
    switch (spanData.type) {
      case "agent":
        return SpanType.AGENT;
      case "function":
      case "mcp_tools":
        return SpanType.TOOL;
      case "generation":
      case "response":
        return SpanType.LLM;
      default:
        return "base";
    }
  }

  public async forceFlush(): Promise<void> {}
  public async shutdown(): Promise<void> {}
}
