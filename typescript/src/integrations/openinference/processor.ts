import { SpanProcessor, ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Context, Span } from "@opentelemetry/api";
import {
  SpanType,
  getCurrentSpan,
  setCurrentSpan,
  traceManager,
  BaseSpan,
  LlmSpan,
  ToolSpan,
  TraceSpanStatus,
} from "@/tracing/tracing";
import {
  applyPendingToSpan,
  pendingToOtelAttributes,
  popPendingFor,
  setDefaultSpanAttribute,
} from "@/tracing/pending-context";
import {
  ROUTE_TO_REST_ATTRIBUTE,
  endOtelImplicitTrace,
  resolveSpanRoute,
  resolveTraceForOtelSpan,
} from "@/tracing/otel-routing";
import { OpenInferenceInstrumentationOptions } from "@/integrations/openinference/index";
import { ToolCall } from "@/test-case";
import { ConfidentAttr } from "@/tracing/attributes";

// ---------------------------------------------------------------------------
// OI span kind -> internal SpanType mapping
// ---------------------------------------------------------------------------

const OI_KIND_TO_SPAN_TYPE: Record<string, SpanType> = {
  AGENT: SpanType.AGENT,
  CHAIN: SpanType.AGENT,
  LLM: SpanType.LLM,
  TOOL: SpanType.TOOL,
  RETRIEVER: SpanType.RETRIEVER,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getSpanKind(attrs: Record<string, any>): SpanType | null {
  const raw = attrs["openinference.span.kind"];
  if (!raw) return null;
  const kind = String(raw).toUpperCase();
  return OI_KIND_TO_SPAN_TYPE[kind] ?? SpanType.CUSTOM;
}

function extractMessages(
  attrs: Record<string, any>,
): [string | null, string | null] {
  let inputText: string | null = null;
  let outputText: string | null = null;

  // INPUT
  // Strategy 1: walk flattened indexed keys llm.input_messages.{i}.message.content
  let idx = 0;
  let lastContent: string | null = null;
  while (true) {
    const roleKey = `llm.input_messages.${idx}.message.role`;
    const contentKey = `llm.input_messages.${idx}.message.content`;
    if (roleKey in attrs || contentKey in attrs) {
      const content = attrs[contentKey];
      if (content != null) lastContent = String(content);
      idx++;
    } else {
      break;
    }
  }
  if (lastContent != null) {
    inputText = lastContent;
  } else if ("llm.input_messages" in attrs) {
    // Strategy 2: raw JSON blob fallback
    try {
      const raw = attrs["llm.input_messages"];
      const data = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (Array.isArray(data) && data.length > 0) {
        const lastMsg = data[data.length - 1];
        inputText =
          lastMsg?.content ?? lastMsg?.message?.content ?? String(lastMsg);
      }
    } catch {
      inputText = String(attrs["llm.input_messages"]);
    }
  }

  // Strategy 3: generic fallback for agent/tool spans
  if (!inputText && attrs["input.value"] != null) {
    inputText = String(attrs["input.value"]);
  }

  // OUTPUT
  // Strategy 1: walk flattened indexed keys llm.output_messages.{i}.message.content
  idx = 0;
  lastContent = null;
  while (true) {
    const roleKey = `llm.output_messages.${idx}.message.role`;
    const contentKey = `llm.output_messages.${idx}.message.content`;
    if (roleKey in attrs || contentKey in attrs) {
      const content = attrs[contentKey];
      if (content != null) lastContent = String(content);
      idx++;
    } else {
      break;
    }
  }
  if (lastContent != null) {
    outputText = lastContent;
  } else if ("llm.output_messages" in attrs) {
    // Strategy 2: raw JSON blob fallback
    try {
      const raw = attrs["llm.output_messages"];
      const data = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (Array.isArray(data) && data.length > 0) {
        const lastMsg = data[data.length - 1];
        outputText =
          lastMsg?.content ?? lastMsg?.message?.content ?? String(lastMsg);
      }
    } catch {
      outputText = String(attrs["llm.output_messages"]);
    }
  }

  // Strategy 3: generic fallback for agent/tool spans
  if (!outputText && attrs["output.value"] != null) {
    outputText = String(attrs["output.value"]);
  }

  return [inputText, outputText];
}

function extractToolCalls(attrs: Record<string, any>): ToolCall[] {
  const tools: ToolCall[] = [];

  // Scenario A: the span itself IS a tool — tool.name is present directly
  if ("tool.name" in attrs) {
    const toolName = String(attrs["tool.name"]);
    const rawArgs = attrs["tool.parameters"] ?? "{}";
    let params: Record<string, any> = {};
    try {
      params = typeof rawArgs === "string" ? JSON.parse(rawArgs) : rawArgs;
    } catch {
      params = {};
    }
    tools.push({ name: toolName, inputParameters: params });
    return tools;
  }

  // Scenario B: LLM span with tool calls nested inside output_messages
  // Walk flattened: llm.output_messages.{msgIdx}.message.tool_calls.{tcIdx}.tool_call.function.name
  let msgIdx = 0;
  while (true) {
    const hasMsg =
      `llm.output_messages.${msgIdx}.message.role` in attrs ||
      `llm.output_messages.${msgIdx}.message.content` in attrs;
    if (!hasMsg) break;

    let tcIdx = 0;
    while (true) {
      const baseKey = `llm.output_messages.${msgIdx}.message.tool_calls.${tcIdx}.tool_call.function`;
      const nameKey = `${baseKey}.name`;
      if (!(nameKey in attrs)) break;

      const tName = String(attrs[nameKey]);
      const rawTArgs = attrs[`${baseKey}.arguments`] ?? "{}";
      let tParams: Record<string, any> = {};
      try {
        tParams =
          typeof rawTArgs === "string" ? JSON.parse(rawTArgs) : rawTArgs;
      } catch {
        tParams = {};
      }
      tools.push({ name: tName, inputParameters: tParams });
      tcIdx++;
    }

    msgIdx++;
  }

  // Fallback: llm.output_messages is a raw JSON blob
  if (tools.length === 0 && "llm.output_messages" in attrs) {
    try {
      const raw = attrs["llm.output_messages"];
      const data = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (Array.isArray(data)) {
        for (const msg of data) {
          for (const tc of msg?.tool_calls ?? []) {
            const func = tc?.function ?? {};
            if (!func.name) continue;
            let tParams: Record<string, any> = {};
            try {
              tParams =
                typeof func.arguments === "string"
                  ? JSON.parse(func.arguments)
                  : (func.arguments ?? {});
            } catch {
              tParams = {};
            }
            tools.push({ name: String(func.name), inputParameters: tParams });
          }
        }
      }
    } catch {
      // ignore
    }
  }

  return tools;
}

function safeJsonParse(val: any): any {
  if (typeof val === "string") {
    try {
      return JSON.parse(val);
    } catch {
      return val;
    }
  }
  return val;
}

// ---------------------------------------------------------------------------
// OpenInferenceSpanProcessor
// ---------------------------------------------------------------------------

export class OpenInferenceSpanProcessor implements SpanProcessor {
  private options: OpenInferenceInstrumentationOptions;
  private otlpEnabled: boolean;
  private oiSpanIds = new Set<string>();
  /** Span to restore as "current" when a span ends, keyed by OTel span id. */
  private previousSpans = new Map<string, BaseSpan | undefined>();

  constructor(
    options?: OpenInferenceInstrumentationOptions,
    routing: { otlpEnabled?: boolean } = {},
  ) {
    this.options = options || {};
    this.otlpEnabled = routing.otlpEnabled ?? true;
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  onStart(span: Span, _context: Context): void {
    const attrs = (span as any).attributes || {};

    // Ignore spans that are not OpenInference spans
    const spanType = getSpanKind(attrs);

    // Track this span id so the filter processor and onEnd can recognise it
    const spanId = span.spanContext().spanId;
    this.oiSpanIds.add(spanId);
    span.setAttribute(ConfidentAttr.INTERNAL_IS_OI_SPAN, true);

    // Trace-level attributes (stamped on every span)
    if (this.options.name) {
      span.setAttribute(ConfidentAttr.TRACE_NAME, this.options.name);
    }
    if (this.options.environment) {
      span.setAttribute(
        ConfidentAttr.TRACE_ENVIRONMENT,
        this.options.environment,
      );
    }
    if (this.options.threadId) {
      span.setAttribute(ConfidentAttr.TRACE_THREAD_ID, this.options.threadId);
    }
    if (this.options.userId) {
      span.setAttribute(ConfidentAttr.TRACE_USER_ID, this.options.userId);
    }
    if (this.options.testCaseId) {
      span.setAttribute(
        ConfidentAttr.TRACE_TEST_CASE_ID,
        this.options.testCaseId,
      );
    }
    if (this.options.turnId) {
      span.setAttribute(ConfidentAttr.TRACE_TURN_ID, this.options.turnId);
    }
    if (this.options.metadata) {
      span.setAttribute(
        ConfidentAttr.TRACE_METADATA,
        JSON.stringify(this.options.metadata),
      );
    }
    if (this.options.tags) {
      span.setAttribute(
        ConfidentAttr.TRACE_TAGS,
        JSON.stringify(this.options.tags),
      );
    }

    // traceMetricCollection takes precedence over metricCollection (same as Python)
    const traceMetricCollection =
      this.options.traceMetricCollection || this.options.metricCollection;
    if (traceMetricCollection) {
      span.setAttribute(
        ConfidentAttr.TRACE_METRIC_COLLECTION,
        traceMetricCollection,
      );
    }

    // Prompt attributes
    if (this.options.prompt) {
      const prompt = this.options.prompt;
      span.setAttribute(ConfidentAttr.SPAN_PROMPT_ALIAS, prompt._alias || "");
      if (prompt.hash) {
        span.setAttribute(
          ConfidentAttr.SPAN_PROMPT_COMMIT_HASH,
          prompt.hash || "",
        );
      }
      if (prompt.label) {
        span.setAttribute(ConfidentAttr.SPAN_PROMPT_LABEL, prompt.label || "");
      }
      if (prompt.version) {
        span.setAttribute(
          ConfidentAttr.SPAN_PROMPT_VERSION,
          prompt.version || "",
        );
      }
    }

    // Span-type attribute
    span.setAttribute(ConfidentAttr.SPAN_TYPE, spanType!);

    // Per-type enrichment
    if (spanType === SpanType.AGENT) {
      const agentName = attrs["agent.name"] || (span as any).name;
      if (agentName) {
        span.setAttribute(ConfidentAttr.SPAN_NAME, String(agentName));
      }
      if (this.options.agentMetricCollection) {
        span.setAttribute(
          ConfidentAttr.SPAN_METRIC_COLLECTION,
          this.options.agentMetricCollection,
        );
      }
    } else if (spanType === SpanType.LLM) {
      if (this.options.llmMetricCollection) {
        span.setAttribute(
          ConfidentAttr.SPAN_METRIC_COLLECTION,
          this.options.llmMetricCollection,
        );
      }
    } else if (spanType === SpanType.TOOL) {
      const toolName = attrs["tool.name"] || (span as any).name;
      if (toolName) {
        span.setAttribute(ConfidentAttr.SPAN_NAME, String(toolName));
        const toolMc = this.options.toolMetricCollectionMap?.[toolName];
        if (toolMc) {
          span.setAttribute(ConfidentAttr.SPAN_METRIC_COLLECTION, toolMc);
        }
      }
    }

    // Routing decision, stamped so `onEnd` and the export filter act on the same
    // answer even if the async context has moved on by then.
    const route = resolveSpanRoute({
      isTestMode: this.options.isTestMode,
      otlpEnabled: this.otlpEnabled,
    });
    if (route === "rest") {
      span.setAttribute(ROUTE_TO_REST_ATTRIBUTE, true);

      const parentId =
        (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
      const isOiRoot = !parentId || !this.oiSpanIds.has(parentId);

      // A bare caller has no trace of their own; open one implicitly for the
      // root so the spans have somewhere to live.
      const currentTrace = resolveTraceForOtelSpan(isOiRoot);
      if (currentTrace) {
        const traceId = currentTrace.uuid;
        span.setAttribute(ConfidentAttr.INTERNAL_TRACE_UUID, traceId);

        const commonParams = {
          uuid: spanId,
          traceUuid: traceId,
          parentUuid: parentId,
          startTime: new Date(), // Accurate time set in onEnd
          type: spanType!,
          name: (span as any).name,
          status: TraceSpanStatus.SUCCESS,
        };

        let deepEvalSpan: BaseSpan;
        if (spanType === SpanType.LLM) {
          deepEvalSpan = new LlmSpan({ ...commonParams, model: "unknown" });
        } else if (spanType === SpanType.TOOL) {
          deepEvalSpan = new ToolSpan(commonParams);
        } else {
          deepEvalSpan = new BaseSpan(commonParams);
        }
        applyPendingToSpan(deepEvalSpan, popPendingFor(spanType ?? undefined));

        traceManager.addSpan(deepEvalSpan);
        try {
          traceManager.addSpanToTrace(deepEvalSpan);
        } catch {
          deepEvalSpan.parentUuid = undefined;
          traceManager.addSpanToTrace(deepEvalSpan);
        }

        this.previousSpans.set(spanId, getCurrentSpan());
        setCurrentSpan(deepEvalSpan);
      }
    } else {
      const staged = pendingToOtelAttributes(
        popPendingFor(spanType ?? undefined),
        spanType ?? undefined,
      );
      for (const [key, value] of Object.entries(staged)) {
        span.setAttribute(key, value);
      }
    }
  }

  /** Restore the span that was current before `spanId` started. */
  private popSpanContext(spanId: string): void {
    if (!this.previousSpans.has(spanId)) return;
    const previous = this.previousSpans.get(spanId);
    this.previousSpans.delete(spanId);
    if (getCurrentSpan()?.uuid === spanId) {
      setCurrentSpan(previous ?? null);
    }
  }

  onEnd(span: ReadableSpan): void {
    const spanId = span.spanContext().spanId;
    if (!this.oiSpanIds.has(spanId)) return;

    const attributes = (span as any).attributes || {};

    // Fall back to re-deriving span type if onStart was somehow skipped
    const spanType = getSpanKind(attributes);
    if (!spanType) {
      this.oiSpanIds.delete(spanId);
      return;
    }
    attributes[ConfidentAttr.SPAN_TYPE] = spanType;

    // Extract input / output from OI semantic convention attributes
    const [inputText, outputText] = extractMessages(attributes);

    // What the framework reports is a default: on the OTLP route anything
    // already on the span was staged by user code at onStart and outranks it.
    if (inputText) {
      setDefaultSpanAttribute(attributes, ConfidentAttr.SPAN_INPUT, inputText);
      setDefaultSpanAttribute(attributes, ConfidentAttr.TRACE_INPUT, inputText);
    }
    if (outputText) {
      setDefaultSpanAttribute(
        attributes,
        ConfidentAttr.SPAN_OUTPUT,
        outputText,
      );
      setDefaultSpanAttribute(
        attributes,
        ConfidentAttr.TRACE_OUTPUT,
        outputText,
      );
    }

    // Token counts (OI keys → confident keys)
    const inputTokens = attributes["llm.token_count.prompt"];
    const outputTokens = attributes["llm.token_count.completion"];
    if (inputTokens != null) {
      setDefaultSpanAttribute(
        attributes,
        ConfidentAttr.LLM_INPUT_TOKEN_COUNT,
        Number(inputTokens),
      );
    }
    if (outputTokens != null) {
      setDefaultSpanAttribute(
        attributes,
        ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT,
        Number(outputTokens),
      );
    }

    // Model name
    const model = attributes["llm.model_name"];
    if (model) {
      setDefaultSpanAttribute(
        attributes,
        ConfidentAttr.LLM_MODEL,
        String(model),
      );
    }

    // Tool calls (agent, llm, and tool spans can all carry tool call info)
    if (
      spanType === SpanType.AGENT ||
      spanType === SpanType.LLM ||
      spanType === SpanType.TOOL
    ) {
      const toolsCalled = extractToolCalls(attributes);
      if (toolsCalled.length > 0) {
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_TOOLS_CALLED,
          JSON.stringify(toolsCalled),
        );
      }
    }

    // Update the span registered in onStart and finalise the trace, for spans the
    // interceptor routed to REST.
    if (attributes[ROUTE_TO_REST_ATTRIBUTE]) {
      this.updateAndEndSpan(span, attributes);
    }

    this.popSpanContext(span.spanContext().spanId);
    this.oiSpanIds.delete(span.spanContext().spanId);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  private updateAndEndSpan(span: ReadableSpan, attributes: any): void {
    const traceId = attributes[ConfidentAttr.INTERNAL_TRACE_UUID] as string;
    if (!traceId) return;

    const spanId = span.spanContext().spanId;
    const deepEvalSpan = (traceManager as any).activeSpans.get(spanId);
    if (!deepEvalSpan) return;

    // Accurate timestamps now that the span has ended
    deepEvalSpan.startTime = new Date(
      span.startTime[0] * 1000 + span.startTime[1] / 1_000_000,
    );
    deepEvalSpan.endTime = new Date(
      span.endTime[0] * 1000 + span.endTime[1] / 1_000_000,
    );

    if (attributes["error"]) {
      deepEvalSpan.status = TraceSpanStatus.ERRORED;
    }

    deepEvalSpan.error = attributes["error"]
      ? String(attributes["error"])
      : undefined;
    // These fields can also have been set from user code (`next*Span`,
    // `updateCurrentSpan`), which outranks what the framework reported: assign
    // only where the attribute carries something and the span does not.
    if (
      attributes[ConfidentAttr.SPAN_INPUT] !== undefined &&
      deepEvalSpan.input === undefined
    ) {
      deepEvalSpan.input = safeJsonParse(attributes[ConfidentAttr.SPAN_INPUT]);
    }
    if (
      attributes[ConfidentAttr.SPAN_OUTPUT] !== undefined &&
      deepEvalSpan.output === undefined
    ) {
      deepEvalSpan.output = safeJsonParse(
        attributes[ConfidentAttr.SPAN_OUTPUT],
      );
    }
    if (
      attributes[ConfidentAttr.SPAN_METRIC_COLLECTION] !== undefined &&
      deepEvalSpan.metricCollection === undefined
    ) {
      deepEvalSpan.metricCollection =
        attributes[ConfidentAttr.SPAN_METRIC_COLLECTION];
    }

    if (
      attributes[ConfidentAttr.SPAN_METADATA] &&
      deepEvalSpan.metadata === undefined
    ) {
      try {
        deepEvalSpan.metadata = JSON.parse(
          attributes[ConfidentAttr.SPAN_METADATA],
        );
      } catch {
        // ignore
      }
    }

    const spanType = deepEvalSpan.type as SpanType;

    if (spanType === SpanType.LLM) {
      const llmSpan = deepEvalSpan as LlmSpan;
      // The span is constructed with a placeholder model, so anything other
      // than "unknown" here came from user code and stands.
      if (!llmSpan.model || llmSpan.model === "unknown") {
        llmSpan.model = attributes[ConfidentAttr.LLM_MODEL] || "unknown";
      }
      if (llmSpan.inputTokenCount === undefined) {
        llmSpan.inputTokenCount =
          attributes[ConfidentAttr.LLM_INPUT_TOKEN_COUNT];
      }
      if (llmSpan.outputTokenCount === undefined) {
        llmSpan.outputTokenCount =
          attributes[ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT];
      }
      if (attributes[ConfidentAttr.SPAN_PROMPT_ALIAS])
        llmSpan.promptAlias = String(
          attributes[ConfidentAttr.SPAN_PROMPT_ALIAS],
        );
      if (attributes[ConfidentAttr.SPAN_PROMPT_COMMIT_HASH])
        llmSpan.promptCommitHash = String(
          attributes[ConfidentAttr.SPAN_PROMPT_COMMIT_HASH],
        );
      if (attributes[ConfidentAttr.SPAN_PROMPT_LABEL])
        llmSpan.promptLabel = String(
          attributes[ConfidentAttr.SPAN_PROMPT_LABEL],
        );
      if (attributes[ConfidentAttr.SPAN_PROMPT_VERSION])
        llmSpan.promptVersion = String(
          attributes[ConfidentAttr.SPAN_PROMPT_VERSION],
        );
    } else if (spanType === SpanType.TOOL) {
      const toolName = attributes[ConfidentAttr.SPAN_NAME] || deepEvalSpan.name;
      deepEvalSpan.name = toolName;

      const currentTrace = traceManager.getTraceByUuid(traceId);
      if (currentTrace) {
        const toolCall: ToolCall = {
          name: toolName,
          inputParameters: safeJsonParse(attributes[ConfidentAttr.SPAN_INPUT]),
          output: safeJsonParse(attributes[ConfidentAttr.SPAN_OUTPUT]),
        };
        if (!currentTrace.toolsCalled) {
          currentTrace.toolsCalled = [];
        }
        currentTrace.toolsCalled.push(toolCall);
      }
    }

    if (spanType === SpanType.AGENT) {
      const currentTrace = traceManager.getTraceByUuid(traceId);
      if (currentTrace) {
        if (!currentTrace.input && deepEvalSpan.input) {
          currentTrace.input = deepEvalSpan.input;
        }
        if (deepEvalSpan.output) {
          currentTrace.output = deepEvalSpan.output;
        }
      }

      const parentId =
        (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
      if (!parentId) {
        traceManager.endTrace(traceId);
      }
    }

    const rootParentId =
      (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
    if (!rootParentId) {
      endOtelImplicitTrace(traceId);
    }
  }
}
