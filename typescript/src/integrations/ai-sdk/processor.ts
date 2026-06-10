import { SpanProcessor, ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Context, Span } from "@opentelemetry/api";
import { getLlmContext } from "../../tracing/trace-context";
import {
  SpanType,
  getCurrentTrace,
  traceManager,
  BaseSpan,
  LlmSpan,
  ToolSpan,
  RetrieverSpan,
  TraceSpanStatus,
} from "../../tracing/tracing";
import { AiSdkInstrumentationOptions } from "./index";
import { ToolCall } from "../../test-case";

export const ROOT_VERCEL_SPANS = new Set([
  "ai.generateText",
  "ai.streamText",
  "ai.generateObject",
  "ai.streamObject",
  "ai.embed",
  "ai.embedMany",
]);

const SPAN_TYPE_MAPPING: Record<string, SpanType> = {
  "ai.generateText": SpanType.LLM,
  "ai.streamText": SpanType.LLM,
  "ai.generateText.doGenerate": SpanType.LLM,
  "ai.streamText.doStream": SpanType.LLM,
  "ai.generateObject": SpanType.LLM,
  "ai.streamObject": SpanType.LLM,
  "ai.generateObject.doGenerate": SpanType.LLM,
  "ai.streamObject.doStream": SpanType.LLM,
  "ai.embed": SpanType.RETRIEVER,
  "ai.embedMany": SpanType.RETRIEVER,
  "ai.embed.doEmbed": SpanType.RETRIEVER,
  "ai.embedMany.doEmbed": SpanType.RETRIEVER,
  "ai.toolCall": SpanType.TOOL,
};

export class DeepEvalSpanProcessor implements SpanProcessor {
  private options: AiSdkInstrumentationOptions;
  private aiSpanIds = new Set<string>();

  constructor(options?: AiSdkInstrumentationOptions) {
    this.options = options || {};
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  onStart(span: Span, _context: Context): void {
    const spanName = (span as any).name;
    if (!spanName || !spanName.startsWith("ai.")) return;

    const spanId = span.spanContext().spanId;
    this.aiSpanIds.add(spanId);

    const parentId =
      (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
    const isAiRoot = !parentId || !this.aiSpanIds.has(parentId);
    if (isAiRoot) {
      span.setAttribute("confident.internal.is_ai_root", true);
    }

    this.setTraceAttributes(span);
    this.setSpanAttributes(span, spanName);

    if (this.options.isTestMode) {
      const currentTrace = getCurrentTrace();
      if (currentTrace) {
        const traceId = currentTrace.uuid;
        span.setAttribute("confident.internal.trace_uuid", traceId);

        const spanId = span.spanContext().spanId;
        const parentId =
          (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
        const type = this.determineSpanType(spanName);

        const commonParams = {
          uuid: spanId,
          traceUuid: traceId,
          parentUuid: parentId,
          startTime: new Date(), // Placeholder: Will be accurate in onEnd
          type,
          name: spanName,
          status: TraceSpanStatus.SUCCESS,
        };

        let deepEvalSpan: BaseSpan;
        if (type === SpanType.LLM) {
          deepEvalSpan = new LlmSpan({ ...commonParams, model: "unknown" });
        } else if (type === SpanType.TOOL) {
          deepEvalSpan = new ToolSpan(commonParams);
        } else if (type === SpanType.RETRIEVER) {
          deepEvalSpan = new RetrieverSpan({
            ...commonParams,
            embedder: "unknown",
          });
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
    const name = span.name;
    if (!name.startsWith("ai.")) return;

    const attributes = (span as any).attributes || {};
    const type = this.determineSpanType(name);
    const isAiRoot = attributes["confident.internal.is_ai_root"] === true;

    this.setSpanLevelAttributes(attributes, name);

    if (type === SpanType.TOOL) {
      const traceId =
        (attributes["confident.internal.trace_uuid"] as string) ||
        getCurrentTrace()?.uuid;

      if (traceId) {
        const currentTrace = traceManager.getTraceByUuid(traceId);
        if (currentTrace) {
          if (!currentTrace.toolsCalled) {
            currentTrace.toolsCalled = [];
          }
          const toolCall: ToolCall = {
            name: attributes["confident.tool.name"]
              ? String(attributes["confident.tool.name"])
              : name,
            inputParameters: this.safeJsonParse(
              attributes["confident.span.input"],
            ),
            output: this.safeJsonParse(attributes["confident.span.output"]),
            description: attributes["confident.span.metadata"]
              ? JSON.parse(attributes["confident.span.metadata"]).description
              : undefined,
          };

          currentTrace.toolsCalled.push(toolCall);

          attributes["confident.trace.tools_called"] = JSON.stringify(
            currentTrace.toolsCalled,
          );
        }
      }
    }

    if (ROOT_VERCEL_SPANS.has(name)) {
      const currentTrace = getCurrentTrace();
      if (attributes["confident.span.input"]) {
        if (currentTrace) {
          if (isAiRoot && !currentTrace.input) {
            currentTrace.input = attributes["confident.span.input"];
          }
          if (isAiRoot) {
            attributes["confident.trace.input"] =
              currentTrace.input || attributes["confident.span.input"];
          }
        } else {
          if (isAiRoot) {
            attributes["confident.trace.input"] =
              attributes["confident.span.input"];
          }
        }
      }
      if (attributes["confident.span.output"]) {
        if (currentTrace) {
          if (isAiRoot) {
            currentTrace.output = attributes["confident.span.output"];
            attributes["confident.trace.output"] =
              currentTrace.output || attributes["confident.span.output"];
          }
        } else {
          if (isAiRoot) {
            attributes["confident.trace.output"] =
              attributes["confident.span.output"];
          }
        }
      }
      if (attributes["ai.telemetry.functionId"]) {
        attributes["confident.trace.name"] =
          attributes["ai.telemetry.functionId"];
      }
    }

    if (this.options.isTestMode) {
      this.updateAndEndSpan(span, attributes, name);
    }

    this.aiSpanIds.delete(span.spanContext().spanId);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  private setTraceAttributes(span: Span): void {
    if (this.options.name) {
      span.setAttribute("confident.trace.name", this.options.name);
    }
    if (this.options.environment) {
      span.setAttribute(
        "confident.trace.environment",
        this.options.environment,
      );
    }
    if (this.options.traceMetricCollection) {
      span.setAttribute(
        "confident.trace.metric_collection",
        this.options.traceMetricCollection,
      );
    }

    const currentTrace = getCurrentTrace();

    if (currentTrace) {
      if (currentTrace.threadId) {
        span.setAttribute("confident.trace.thread_id", currentTrace.threadId);
      }
      if (currentTrace.userId) {
        span.setAttribute("confident.trace.user_id", currentTrace.userId);
      }
      if (currentTrace.testCaseId) {
        span.setAttribute(
          "confident.trace.test_case_id",
          currentTrace.testCaseId,
        );
      }
      if (currentTrace.turnId) {
        span.setAttribute("confident.trace.turn_id", currentTrace.turnId);
      }
      if (currentTrace.metadata) {
        span.setAttribute(
          "confident.trace.metadata",
          JSON.stringify(currentTrace.metadata),
        );
      }
      if (currentTrace.tags) {
        span.setAttribute(
          "confident.trace.tags",
          JSON.stringify(currentTrace.tags),
        );
      }
      if (currentTrace.metricCollection) {
        span.setAttribute(
          "confident.trace.metric_collection",
          currentTrace.metricCollection,
        );
      }
      if (currentTrace.context) {
        span.setAttribute(
          "confident.trace.context",
          JSON.stringify(currentTrace.context),
        );
      }
      if (currentTrace.retrievalContext) {
        span.setAttribute(
          "confident.trace.retrieval_context",
          JSON.stringify(currentTrace.retrievalContext),
        );
      }
      if (currentTrace.expectedOutput) {
        span.setAttribute(
          "confident.trace.expected_output",
          currentTrace.expectedOutput,
        );
      }
      if (currentTrace.expectedTools) {
        span.setAttribute(
          "confident.trace.expected_tools",
          JSON.stringify(currentTrace.expectedTools),
        );
      }
    }
  }

  private setSpanAttributes(span: Span, spanName: string): void {
    const type = this.determineSpanType(spanName);

    span.setAttribute("confident.span.type", type);

    const llmContext = getLlmContext();

    if (type === SpanType.LLM) {
      if (llmContext) {
        if (llmContext.metricCollection) {
          span.setAttribute(
            "confident.span.metric_collection",
            llmContext.metricCollection,
          );
        }
        if (llmContext.context) {
          span.setAttribute(
            "confident.span.context",
            JSON.stringify(llmContext.context),
          );
        }
        if (llmContext.retrievalContext) {
          span.setAttribute(
            "confident.span.retrieval_context",
            JSON.stringify(llmContext.retrievalContext),
          );
        }
        if (llmContext.expectedOutput) {
          span.setAttribute(
            "confident.span.expected_output",
            llmContext.expectedOutput,
          );
        }
        if (llmContext.expectedTools) {
          span.setAttribute(
            "confident.span.expected_tools",
            JSON.stringify(llmContext.expectedTools),
          );
        }
        if (llmContext.prompt) {
          span.setAttribute(
            "confident.span.prompt_alias",
            llmContext.prompt._alias || "",
          );
          span.setAttribute(
            "confident.span.prompt_commit_hash",
            llmContext.prompt.hash || "",
          );
          span.setAttribute(
            "confident.span.prompt_label",
            llmContext.prompt.label || "",
          );
          span.setAttribute(
            "confident.span.prompt_version",
            llmContext.prompt.version || "",
          );
        }
      }
    } else if (type === SpanType.TOOL) {
      const metricCollection = llmContext?.toolsMetricCollection;
      if (metricCollection) {
        span.setAttribute("confident.span.metric_collection", metricCollection);
      }
      span.setAttribute("confident.trace.tools_called", "true");
    }
  }

  private setSpanLevelAttributes(attributes: any, spanName: string): void {
    const type = this.determineSpanType(spanName);
    attributes["confident.span.type"] = type;

    const getMeta = (key: string) => {
      const val = attributes[`ai.telemetry.metadata.${key}`];
      return val !== undefined ? this.safeJsonParse(val) : undefined;
    };

    const userId = getMeta("userId");
    if (userId) attributes["confident.trace.user_id"] = String(userId);

    const testCaseId = getMeta("testCaseId");
    if (testCaseId)
      attributes["confident.trace.test_case_id"] = String(testCaseId);

    const turnId = getMeta("turnId");
    if (turnId) attributes["confident.trace.turn_id"] = String(turnId);

    const threadId = getMeta("threadId");
    if (threadId) attributes["confident.trace.thread_id"] = String(threadId);

    const metricCollection = getMeta("metricCollection");
    if (metricCollection)
      attributes["confident.span.metric_collection"] = String(metricCollection);

    const tags = getMeta("tags");
    if (tags) {
      attributes["confident.trace.tags"] =
        typeof tags === "string" ? tags : JSON.stringify(tags);
    }

    const contextAttr = getMeta("context");
    if (contextAttr) {
      attributes["confident.trace.context"] =
        typeof contextAttr === "string"
          ? contextAttr
          : JSON.stringify(contextAttr);
    }

    const traceName = getMeta("traceName");
    if (traceName) attributes["confident.trace.name"] = String(traceName);

    const traceMetricCollection = getMeta("traceMetricCollection");
    if (traceMetricCollection)
      attributes["confident.trace.metric_collection"] = String(
        traceMetricCollection,
      );

    const expectedOutput = getMeta("expectedOutput");
    if (expectedOutput)
      attributes["confident.trace.expected_output"] = String(expectedOutput);

    const sessionId = getMeta("sessionId");
    if (sessionId) attributes["confident.trace.session_id"] = String(sessionId);

    const promptAlias = getMeta("promptAlias");
    if (promptAlias)
      attributes["confident.span.prompt_alias"] = String(promptAlias);

    const promptCommitHash = getMeta("promptCommitHash");
    if (promptCommitHash)
      attributes["confident.span.prompt_commit_hash"] =
        String(promptCommitHash);

    const metadata: Record<string, any> = {};

    for (const [key, value] of Object.entries(attributes)) {
      if (key.startsWith("ai.telemetry.metadata.")) {
        const shortKey = key.replace("ai.telemetry.metadata.", "");
        metadata[shortKey] = value;
      }
    }

    if (type === SpanType.LLM) {
      const model =
        attributes["ai.model.id"] ||
        attributes["gen_ai.request.model"] ||
        attributes["gen_ai.response.model"];
      if (model) attributes["confident.llm.model"] = String(model);

      let input = attributes["ai.prompt"];
      if (!input && attributes["ai.prompt.messages"]) {
        input = this.ensureString(attributes["ai.prompt.messages"]);
      }
      if (input) attributes["confident.span.input"] = this.ensureString(input);

      let output = attributes["ai.response.text"];
      if (!output && attributes["ai.response.object"]) {
        output = this.ensureString(attributes["ai.response.object"]);
      }
      if (!output && attributes["ai.response.toolCalls"]) {
        output = this.ensureString(attributes["ai.response.toolCalls"]);
      }
      if (output)
        attributes["confident.span.output"] = this.ensureString(output);

      if (!ROOT_VERCEL_SPANS.has(spanName)) {
        const inputTokens =
          attributes["ai.usage.inputTokens.total"] ||
          attributes["gen_ai.usage.input_tokens"] ||
          attributes["ai.usage.promptTokens"];

        if (inputTokens !== undefined) {
          attributes["confident.llm.input_token_count"] = Number(inputTokens);
        }
        const outputTokens =
          attributes["ai.usage.outputTokens.total"] ||
          attributes["gen_ai.usage.output_tokens"] ||
          attributes["ai.usage.completionTokens"];

        if (outputTokens !== undefined) {
          attributes["confident.llm.output_token_count"] = Number(outputTokens);
        }
      }

      this.collectMetadata(
        attributes,
        metadata,
        "ai.response.id",
        "response_id",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.response.model",
        "response_model",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.model.provider",
        "provider",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.telemetry.functionId",
        "function_id",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "resource.name",
        "resource_name",
      );

      this.collectMetadata(
        attributes,
        metadata,
        "ai.response.msToFirstChunk",
        "ms_to_first_chunk",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.response.msToFinish",
        "ms_to_finish",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.response.avgCompletionTokensPerSecond",
        "avg_tokens_per_second",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.response.finishReason",
        "finish_reason",
      );

      this.collectMetadata(
        attributes,
        metadata,
        "ai.settings.maxOutputTokens",
        "max_tokens",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "gen_ai.request.max_tokens",
        "max_tokens",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "gen_ai.request.temperature",
        "temperature",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "gen_ai.request.top_p",
        "top_p",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "gen_ai.request.top_k",
        "top_k",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "gen_ai.request.frequency_penalty",
        "frequency_penalty",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "gen_ai.request.presence_penalty",
        "presence_penalty",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.settings.maxRetries",
        "max_retries",
      );

      this.collectMetadata(
        attributes,
        metadata,
        "ai.schema.name",
        "schema_name",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.schema.description",
        "schema_description",
      );
      this.collectMetadata(
        attributes,
        metadata,
        "ai.settings.output",
        "output_mode",
      );
      if (attributes["ai.schema"]) {
        metadata["schema"] = this.ensureString(attributes["ai.schema"]);
      }
    } else if (type === SpanType.TOOL) {
      const toolName = attributes["ai.toolCall.name"];
      if (toolName) attributes["confident.tool.name"] = String(toolName);

      const args = attributes["ai.toolCall.args"];
      if (args) attributes["confident.span.input"] = this.ensureString(args);

      const result = attributes["ai.toolCall.result"];
      if (result)
        attributes["confident.span.output"] = this.ensureString(result);

      const toolId = attributes["ai.toolCall.id"];
      if (toolId)
        attributes["confident.span.metadata.tool_id"] = String(toolId);
    } else if (type === SpanType.RETRIEVER) {
      const embedder = attributes["ai.model.id"];
      if (embedder)
        attributes["confident.retriever.embedder"] = String(embedder);

      const val = attributes["ai.value"] || attributes["ai.values"];
      if (val) attributes["confident.span.input"] = this.ensureString(val);

      const embedding =
        attributes["ai.embedding"] || attributes["ai.embeddings"];
      if (embedding)
        attributes["confident.span.output"] = this.ensureString(embedding);

      if (!ROOT_VERCEL_SPANS.has(spanName)) {
        this.collectMetadata(attributes, metadata, "ai.usage.tokens", "tokens");
      }
    }

    if (Object.keys(metadata).length > 0) {
      attributes["confident.span.metadata"] = JSON.stringify(metadata);
    }
  }

  private determineSpanType(name: string): SpanType {
    if (SPAN_TYPE_MAPPING[name]) return SPAN_TYPE_MAPPING[name];
    if (name.includes("generate") || name.includes("stream"))
      return SpanType.LLM;
    if (name.includes("embed")) return SpanType.RETRIEVER;
    if (name.includes("tool")) return SpanType.TOOL;
    return SpanType.CUSTOM;
  }

  private collectMetadata(
    attributes: any,
    metadata: Record<string, any>,
    sourceKey: string,
    destKey: string,
  ) {
    if (attributes[sourceKey] !== undefined) {
      metadata[destKey] = attributes[sourceKey];
    }
  }

  private ensureString(val: any): string {
    if (val === undefined || val === null) return "";
    if (typeof val === "string") return val;
    try {
      return JSON.stringify(val);
    } catch {
      return String(val);
    }
  }

  private updateAndEndSpan(span: ReadableSpan, attributes: any, name: string) {
    const traceId = attributes["confident.internal.trace_uuid"] as string;
    if (!traceId) return;

    const spanId = span.spanContext().spanId;

    const deepEvalSpan = (traceManager as any).activeSpans.get(spanId);
    if (!deepEvalSpan) return;

    deepEvalSpan.startTime = new Date(
      span.startTime[0] * 1000 + span.startTime[1] / 1000000,
    );
    deepEvalSpan.endTime = new Date(
      span.endTime[0] * 1000 + span.endTime[1] / 1000000,
    );

    if (attributes["error"]) deepEvalSpan.status = TraceSpanStatus.ERRORED;

    let inputObj = attributes["confident.span.input"];
    let outputObj = attributes["confident.span.output"];
    try {
      if (typeof inputObj === "string") inputObj = JSON.parse(inputObj);
    } catch {
      // Ignore parsing normal strings
    }
    try {
      if (typeof outputObj === "string") outputObj = JSON.parse(outputObj);
    } catch {
      // Ignore parsing normal strings
    }

    let metadataObj = undefined;
    if (attributes["confident.span.metadata"]) {
      try {
        metadataObj = JSON.parse(attributes["confident.span.metadata"]);
      } catch {
        // Ignore parsing normal strings
      }
    }

    deepEvalSpan.input = inputObj;
    deepEvalSpan.output = outputObj;
    deepEvalSpan.error = attributes["error"]
      ? String(attributes["error"])
      : undefined;
    deepEvalSpan.metricCollection =
      attributes["confident.span.metric_collection"];
    deepEvalSpan.metadata = metadataObj;

    if (deepEvalSpan.type === SpanType.LLM) {
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
    } else if (deepEvalSpan.type === SpanType.TOOL) {
      deepEvalSpan.name = attributes["confident.tool.name"]
        ? String(attributes["confident.tool.name"])
        : name;

      const currentTrace = traceManager.getTraceByUuid(traceId);
      if (currentTrace) {
        if (!currentTrace.toolsCalled) {
          currentTrace.toolsCalled = [];
        }

        const toolCall: ToolCall = {
          name: deepEvalSpan.name,
          inputParameters: inputObj,
          output: outputObj,
        };

        if (currentTrace.toolsCalled) {
          currentTrace.toolsCalled.push(toolCall);
        } else {
          currentTrace.toolsCalled = [toolCall];
        }
      }
    } else if (deepEvalSpan.type === SpanType.RETRIEVER) {
      (deepEvalSpan as RetrieverSpan).embedder =
        attributes["confident.retriever.embedder"] || "unknown";
    }

    if (ROOT_VERCEL_SPANS.has(name)) {
      const currentTrace = traceManager.getTraceByUuid(traceId);
      if (currentTrace) {
        if (!currentTrace.input && inputObj) currentTrace.input = inputObj;
        if (outputObj) currentTrace.output = outputObj;
        if (attributes["ai.telemetry.functionId"])
          currentTrace.name = String(attributes["ai.telemetry.functionId"]);
      }
      const parentId =
        (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
      if (!parentId) {
        traceManager.endTrace(traceId);
      }
    }
  }

  private safeJsonParse(val: any): any {
    if (typeof val === "string") {
      try {
        return JSON.parse(val);
      } catch {
        return val;
      }
    }
    return val;
  }
}
