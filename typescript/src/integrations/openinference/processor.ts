import { SpanProcessor, ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Context, Span } from "@opentelemetry/api";
import {
  SpanType,
  getCurrentTrace,
  traceManager,
  BaseSpan,
  LlmSpan,
  ToolSpan,
  TraceSpanStatus,
} from "../../tracing/tracing";
import { OpenInferenceInstrumentationOptions } from "./index";
import { ToolCall } from "../../test-case";

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
  private oiSpanIds = new Set<string>();

  constructor(options?: OpenInferenceInstrumentationOptions) {
    this.options = options || {};
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
    span.setAttribute("confident.internal.is_oi_span", true);

    // Trace-level attributes (stamped on every span)
    if (this.options.name) {
      span.setAttribute("confident.trace.name", this.options.name);
    }
    if (this.options.environment) {
      span.setAttribute(
        "confident.trace.environment",
        this.options.environment,
      );
    }
    if (this.options.threadId) {
      span.setAttribute("confident.trace.thread_id", this.options.threadId);
    }
    if (this.options.userId) {
      span.setAttribute("confident.trace.user_id", this.options.userId);
    }
    if (this.options.testCaseId) {
      span.setAttribute(
        "confident.trace.test_case_id",
        this.options.testCaseId,
      );
    }
    if (this.options.turnId) {
      span.setAttribute("confident.trace.turn_id", this.options.turnId);
    }
    if (this.options.metadata) {
      span.setAttribute(
        "confident.trace.metadata",
        JSON.stringify(this.options.metadata),
      );
    }
    if (this.options.tags) {
      span.setAttribute(
        "confident.trace.tags",
        JSON.stringify(this.options.tags),
      );
    }

    // traceMetricCollection takes precedence over metricCollection (same as Python)
    const traceMetricCollection =
      this.options.traceMetricCollection || this.options.metricCollection;
    if (traceMetricCollection) {
      span.setAttribute(
        "confident.trace.metric_collection",
        traceMetricCollection,
      );
    }

    // Prompt attributes
    if (this.options.prompt) {
      const prompt = this.options.prompt;
      span.setAttribute("confident.span.prompt_alias", prompt._alias || "");
      if (prompt.hash) {
        span.setAttribute(
          "confident.span.prompt_commit_hash",
          prompt.hash || "",
        );
      }
      if (prompt.label) {
        span.setAttribute("confident.span.prompt_label", prompt.label || "");
      }
      if (prompt.version) {
        span.setAttribute(
          "confident.span.prompt_version",
          prompt.version || "",
        );
      }
    }

    // Span-type attribute
    span.setAttribute("confident.span.type", spanType!);

    // Per-type enrichment
    if (spanType === SpanType.AGENT) {
      const agentName = attrs["agent.name"] || (span as any).name;
      if (agentName) {
        span.setAttribute("confident.span.name", String(agentName));
      }
      if (this.options.agentMetricCollection) {
        span.setAttribute(
          "confident.span.metric_collection",
          this.options.agentMetricCollection,
        );
      }
    } else if (spanType === SpanType.LLM) {
      if (this.options.llmMetricCollection) {
        span.setAttribute(
          "confident.span.metric_collection",
          this.options.llmMetricCollection,
        );
      }
    } else if (spanType === SpanType.TOOL) {
      const toolName = attrs["tool.name"] || (span as any).name;
      if (toolName) {
        span.setAttribute("confident.span.name", String(toolName));
        const toolMc = this.options.toolMetricCollectionMap?.[toolName];
        if (toolMc) {
          span.setAttribute("confident.span.metric_collection", toolMc);
        }
      }
    }

    // Test mode: register span with traceManager
    if (this.options.isTestMode) {
      const currentTrace = getCurrentTrace();
      if (currentTrace) {
        const traceId = currentTrace.uuid;
        span.setAttribute("confident.internal.trace_uuid", traceId);

        const parentId =
          (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;

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

        traceManager.addSpan(deepEvalSpan);
        try {
          traceManager.addSpanToTrace(deepEvalSpan);
        } catch {
          deepEvalSpan.parentUuid = undefined;
          traceManager.addSpanToTrace(deepEvalSpan);
        }
      }
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
    attributes["confident.span.type"] = spanType;

    // Extract input / output from OI semantic convention attributes
    const [inputText, outputText] = extractMessages(attributes);

    if (inputText) {
      attributes["confident.span.input"] = inputText;
      attributes["confident.trace.input"] = inputText;
    }
    if (outputText) {
      attributes["confident.span.output"] = outputText;
      attributes["confident.trace.output"] = outputText;
    }

    // Token counts (OI keys → confident keys)
    const inputTokens = attributes["llm.token_count.prompt"];
    const outputTokens = attributes["llm.token_count.completion"];
    if (inputTokens != null) {
      attributes["confident.llm.input_token_count"] = Number(inputTokens);
    }
    if (outputTokens != null) {
      attributes["confident.llm.output_token_count"] = Number(outputTokens);
    }

    // Model name
    const model = attributes["llm.model_name"];
    if (model) {
      attributes["confident.llm.model"] = String(model);
    }

    // Tool calls (agent, llm, and tool spans can all carry tool call info)
    if (
      spanType === SpanType.AGENT ||
      spanType === SpanType.LLM ||
      spanType === SpanType.TOOL
    ) {
      const toolsCalled = extractToolCalls(attributes);
      if (toolsCalled.length > 0) {
        attributes["confident.span.tools_called"] = JSON.stringify(toolsCalled);
      }
    }

    // Test mode: update the span registered in onStart and finalise trace
    if (this.options.isTestMode) {
      this.updateAndEndSpan(span, attributes);
    }

    this.oiSpanIds.delete(span.spanContext().spanId);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  private updateAndEndSpan(span: ReadableSpan, attributes: any): void {
    const traceId = attributes["confident.internal.trace_uuid"] as string;
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

    deepEvalSpan.input = safeJsonParse(attributes["confident.span.input"]);
    deepEvalSpan.output = safeJsonParse(attributes["confident.span.output"]);
    deepEvalSpan.error = attributes["error"]
      ? String(attributes["error"])
      : undefined;
    deepEvalSpan.metricCollection =
      attributes["confident.span.metric_collection"];

    if (attributes["confident.span.metadata"]) {
      try {
        deepEvalSpan.metadata = JSON.parse(
          attributes["confident.span.metadata"],
        );
      } catch {
        // ignore
      }
    }

    const spanType = deepEvalSpan.type as SpanType;

    if (spanType === SpanType.LLM) {
      const llmSpan = deepEvalSpan as LlmSpan;
      llmSpan.model = attributes["confident.llm.model"] || "unknown";
      llmSpan.inputTokenCount = attributes["confident.llm.input_token_count"];
      llmSpan.outputTokenCount = attributes["confident.llm.output_token_count"];
      if (attributes["confident.span.prompt_alias"])
        llmSpan.promptAlias = String(attributes["confident.span.prompt_alias"]);
      if (attributes["confident.span.prompt_commit_hash"])
        llmSpan.promptCommitHash = String(
          attributes["confident.span.prompt_commit_hash"],
        );
      if (attributes["confident.span.prompt_label"])
        llmSpan.promptLabel = String(attributes["confident.span.prompt_label"]);
      if (attributes["confident.span.prompt_version"])
        llmSpan.promptVersion = String(
          attributes["confident.span.prompt_version"],
        );
    } else if (spanType === SpanType.TOOL) {
      const toolName = attributes["confident.span.name"] || deepEvalSpan.name;
      deepEvalSpan.name = toolName;

      const currentTrace = traceManager.getTraceByUuid(traceId);
      if (currentTrace) {
        const toolCall: ToolCall = {
          name: toolName,
          inputParameters: safeJsonParse(attributes["confident.span.input"]),
          output: safeJsonParse(attributes["confident.span.output"]),
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
  }
}
