import { SpanProcessor, ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Context, Span } from "@opentelemetry/api";
import { getLlmContext } from "@/tracing/trace-context";
import {
  SpanType,
  getCurrentTrace,
  getCurrentSpan,
  setCurrentSpan,
  traceManager,
  BaseSpan,
  LlmSpan,
  ToolSpan,
  RetrieverSpan,
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
import { AiSdkInstrumentationOptions } from "@/integrations/ai-sdk/index";
import { ToolCall } from "@/test-case";
import { ConfidentAttr } from "@/tracing/attributes";

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
  private otlpEnabled: boolean;
  private aiSpanIds = new Set<string>();
  private previousSpans = new Map<string, BaseSpan | undefined>();

  constructor(
    options?: AiSdkInstrumentationOptions,
    routing: { otlpEnabled?: boolean } = {},
  ) {
    this.options = options || {};
    this.otlpEnabled = routing.otlpEnabled ?? true;
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
      span.setAttribute(ConfidentAttr.INTERNAL_IS_AI_ROOT, true);
    }

    this.setTraceAttributes(span);
    this.setSpanAttributes(span, spanName);

    // Routing decision, stamped so `onEnd` and the export filter act on the same
    // answer even if the async context has moved on by then.
    const route = resolveSpanRoute({
      isTestMode: this.options.isTestMode,
      otlpEnabled: this.otlpEnabled,
    });
    if (route === "rest") {
      span.setAttribute(ROUTE_TO_REST_ATTRIBUTE, true);

      // A bare caller has no trace of their own; open one implicitly for the
      // root so the spans have somewhere to live.
      const currentTrace = resolveTraceForOtelSpan(isAiRoot);
      if (currentTrace) {
        const traceId = currentTrace.uuid;
        span.setAttribute(ConfidentAttr.INTERNAL_TRACE_UUID, traceId);

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

        // Drain `next*Span(...)` defaults and the scope-wide `setTracingContext`
        // values, before the attribute-derived fields land in `onEnd`.
        applyPendingToSpan(deepEvalSpan, popPendingFor(type));

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
      const type = this.determineSpanType(spanName);
      const staged = pendingToOtelAttributes(popPendingFor(type), type);
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
    // Siblings ending out of order may already have replaced the current span.
    if (getCurrentSpan()?.uuid === spanId) {
      setCurrentSpan(previous ?? null);
    }
  }

  onEnd(span: ReadableSpan): void {
    const name = span.name;
    if (!name.startsWith("ai.")) return;

    const attributes = (span as any).attributes || {};
    const type = this.determineSpanType(name);
    const isAiRoot = attributes[ConfidentAttr.INTERNAL_IS_AI_ROOT] === true;

    this.setSpanLevelAttributes(attributes, name);

    if (type === SpanType.TOOL) {
      const traceId =
        (attributes[ConfidentAttr.INTERNAL_TRACE_UUID] as string) ||
        getCurrentTrace()?.uuid;

      if (traceId) {
        const currentTrace = traceManager.getTraceByUuid(traceId);
        if (currentTrace) {
          if (!currentTrace.toolsCalled) {
            currentTrace.toolsCalled = [];
          }
          const toolCall: ToolCall = {
            name: attributes[ConfidentAttr.TOOL_NAME]
              ? String(attributes[ConfidentAttr.TOOL_NAME])
              : name,
            inputParameters: this.safeJsonParse(
              attributes[ConfidentAttr.SPAN_INPUT],
            ),
            output: this.safeJsonParse(attributes[ConfidentAttr.SPAN_OUTPUT]),
            description: attributes[ConfidentAttr.SPAN_METADATA]
              ? JSON.parse(attributes[ConfidentAttr.SPAN_METADATA]).description
              : undefined,
          };

          currentTrace.toolsCalled.push(toolCall);

          attributes[ConfidentAttr.TRACE_TOOLS_CALLED] = JSON.stringify(
            currentTrace.toolsCalled,
          );
        }
      }
    }

    if (ROOT_VERCEL_SPANS.has(name)) {
      const currentTrace = getCurrentTrace();
      if (attributes[ConfidentAttr.SPAN_INPUT]) {
        if (currentTrace) {
          if (isAiRoot && !currentTrace.input) {
            currentTrace.input = attributes[ConfidentAttr.SPAN_INPUT];
          }
          if (isAiRoot) {
            attributes[ConfidentAttr.TRACE_INPUT] =
              currentTrace.input || attributes[ConfidentAttr.SPAN_INPUT];
          }
        } else {
          if (isAiRoot) {
            attributes[ConfidentAttr.TRACE_INPUT] =
              attributes[ConfidentAttr.SPAN_INPUT];
          }
        }
      }
      if (attributes[ConfidentAttr.SPAN_OUTPUT]) {
        if (currentTrace) {
          if (isAiRoot) {
            currentTrace.output = attributes[ConfidentAttr.SPAN_OUTPUT];
            attributes[ConfidentAttr.TRACE_OUTPUT] =
              currentTrace.output || attributes[ConfidentAttr.SPAN_OUTPUT];
          }
        } else {
          if (isAiRoot) {
            attributes[ConfidentAttr.TRACE_OUTPUT] =
              attributes[ConfidentAttr.SPAN_OUTPUT];
          }
        }
      }
      if (attributes["ai.telemetry.functionId"]) {
        attributes[ConfidentAttr.TRACE_NAME] =
          attributes["ai.telemetry.functionId"];
      }
    }

    if (attributes[ROUTE_TO_REST_ATTRIBUTE]) {
      this.updateAndEndSpan(span, attributes, name);
    }

    this.popSpanContext(span.spanContext().spanId);
    this.aiSpanIds.delete(span.spanContext().spanId);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  private setTraceAttributes(span: Span): void {
    if (this.options.name) {
      span.setAttribute(ConfidentAttr.TRACE_NAME, this.options.name);
    }
    if (this.options.environment) {
      span.setAttribute(
        ConfidentAttr.TRACE_ENVIRONMENT,
        this.options.environment,
      );
    }
    if (this.options.traceMetricCollection) {
      span.setAttribute(
        ConfidentAttr.TRACE_METRIC_COLLECTION,
        this.options.traceMetricCollection,
      );
    }

    const currentTrace = getCurrentTrace();

    if (currentTrace) {
      if (currentTrace.threadId) {
        span.setAttribute(ConfidentAttr.TRACE_THREAD_ID, currentTrace.threadId);
      }
      if (currentTrace.userId) {
        span.setAttribute(ConfidentAttr.TRACE_USER_ID, currentTrace.userId);
      }
      if (currentTrace.testCaseId) {
        span.setAttribute(
          ConfidentAttr.TRACE_TEST_CASE_ID,
          currentTrace.testCaseId,
        );
      }
      if (currentTrace.turnId) {
        span.setAttribute(ConfidentAttr.TRACE_TURN_ID, currentTrace.turnId);
      }
      if (currentTrace.metadata) {
        span.setAttribute(
          ConfidentAttr.TRACE_METADATA,
          JSON.stringify(currentTrace.metadata),
        );
      }
      if (currentTrace.tags) {
        span.setAttribute(
          ConfidentAttr.TRACE_TAGS,
          JSON.stringify(currentTrace.tags),
        );
      }
      if (currentTrace.metricCollection) {
        span.setAttribute(
          ConfidentAttr.TRACE_METRIC_COLLECTION,
          currentTrace.metricCollection,
        );
      }
      if (currentTrace.context) {
        span.setAttribute(
          ConfidentAttr.TRACE_CONTEXT,
          JSON.stringify(currentTrace.context),
        );
      }
      if (currentTrace.retrievalContext) {
        span.setAttribute(
          ConfidentAttr.TRACE_RETRIEVAL_CONTEXT,
          JSON.stringify(currentTrace.retrievalContext),
        );
      }
      if (currentTrace.expectedOutput) {
        span.setAttribute(
          ConfidentAttr.TRACE_EXPECTED_OUTPUT,
          currentTrace.expectedOutput,
        );
      }
      if (currentTrace.expectedTools) {
        span.setAttribute(
          ConfidentAttr.TRACE_EXPECTED_TOOLS,
          JSON.stringify(currentTrace.expectedTools),
        );
      }
    }
  }

  private setSpanAttributes(span: Span, spanName: string): void {
    const type = this.determineSpanType(spanName);

    span.setAttribute(ConfidentAttr.SPAN_TYPE, type);

    const llmContext = getLlmContext();

    if (type === SpanType.LLM) {
      if (llmContext) {
        if (llmContext.metricCollection) {
          span.setAttribute(
            ConfidentAttr.SPAN_METRIC_COLLECTION,
            llmContext.metricCollection,
          );
        }
        if (llmContext.context) {
          span.setAttribute(
            ConfidentAttr.SPAN_CONTEXT,
            JSON.stringify(llmContext.context),
          );
        }
        if (llmContext.retrievalContext) {
          span.setAttribute(
            ConfidentAttr.SPAN_RETRIEVAL_CONTEXT,
            JSON.stringify(llmContext.retrievalContext),
          );
        }
        if (llmContext.expectedOutput) {
          span.setAttribute(
            ConfidentAttr.SPAN_EXPECTED_OUTPUT,
            llmContext.expectedOutput,
          );
        }
        if (llmContext.expectedTools) {
          span.setAttribute(
            ConfidentAttr.SPAN_EXPECTED_TOOLS,
            JSON.stringify(llmContext.expectedTools),
          );
        }
        if (llmContext.prompt) {
          span.setAttribute(
            ConfidentAttr.SPAN_PROMPT_ALIAS,
            llmContext.prompt._alias || "",
          );
          span.setAttribute(
            ConfidentAttr.SPAN_PROMPT_COMMIT_HASH,
            llmContext.prompt.hash || "",
          );
          span.setAttribute(
            ConfidentAttr.SPAN_PROMPT_LABEL,
            llmContext.prompt.label || "",
          );
          span.setAttribute(
            ConfidentAttr.SPAN_PROMPT_VERSION,
            llmContext.prompt.version || "",
          );
        }
      }
    } else if (type === SpanType.TOOL) {
      const metricCollection = llmContext?.toolsMetricCollection;
      if (metricCollection) {
        span.setAttribute(
          ConfidentAttr.SPAN_METRIC_COLLECTION,
          metricCollection,
        );
      }
      span.setAttribute(ConfidentAttr.TRACE_TOOLS_CALLED, "true");
    }
  }

  private setSpanLevelAttributes(attributes: any, spanName: string): void {
    const type = this.determineSpanType(spanName);
    attributes[ConfidentAttr.SPAN_TYPE] = type;

    const getMeta = (key: string) => {
      const val = attributes[`ai.telemetry.metadata.${key}`];
      return val !== undefined ? this.safeJsonParse(val) : undefined;
    };

    const userId = getMeta("userId");
    if (userId) attributes[ConfidentAttr.TRACE_USER_ID] = String(userId);

    const testCaseId = getMeta("testCaseId");
    if (testCaseId)
      attributes[ConfidentAttr.TRACE_TEST_CASE_ID] = String(testCaseId);

    const turnId = getMeta("turnId");
    if (turnId) attributes[ConfidentAttr.TRACE_TURN_ID] = String(turnId);

    const threadId = getMeta("threadId");
    if (threadId) attributes[ConfidentAttr.TRACE_THREAD_ID] = String(threadId);

    const metricCollection = getMeta("metricCollection");
    if (metricCollection)
      attributes[ConfidentAttr.SPAN_METRIC_COLLECTION] =
        String(metricCollection);

    const tags = getMeta("tags");
    if (tags) {
      attributes[ConfidentAttr.TRACE_TAGS] =
        typeof tags === "string" ? tags : JSON.stringify(tags);
    }

    const contextAttr = getMeta("context");
    if (contextAttr) {
      attributes[ConfidentAttr.TRACE_CONTEXT] =
        typeof contextAttr === "string"
          ? contextAttr
          : JSON.stringify(contextAttr);
    }

    const traceName = getMeta("traceName");
    if (traceName) attributes[ConfidentAttr.TRACE_NAME] = String(traceName);

    const traceMetricCollection = getMeta("traceMetricCollection");
    if (traceMetricCollection)
      attributes[ConfidentAttr.TRACE_METRIC_COLLECTION] = String(
        traceMetricCollection,
      );

    const expectedOutput = getMeta("expectedOutput");
    if (expectedOutput)
      attributes[ConfidentAttr.TRACE_EXPECTED_OUTPUT] = String(expectedOutput);

    const sessionId = getMeta("sessionId");
    if (sessionId)
      attributes[ConfidentAttr.TRACE_SESSION_ID] = String(sessionId);

    const promptAlias = getMeta("promptAlias");
    if (promptAlias)
      attributes[ConfidentAttr.SPAN_PROMPT_ALIAS] = String(promptAlias);

    const promptCommitHash = getMeta("promptCommitHash");
    if (promptCommitHash)
      attributes[ConfidentAttr.SPAN_PROMPT_COMMIT_HASH] =
        String(promptCommitHash);

    const metadata: Record<string, any> = {};

    for (const [key, value] of Object.entries(attributes)) {
      if (key.startsWith("ai.telemetry.metadata.")) {
        const shortKey = key.replace("ai.telemetry.metadata.", "");
        metadata[shortKey] = value;
      }
    }

    // What the SDK reports is a default: on the OTLP route anything already on
    // the span was staged by user code at onStart and outranks it.
    if (type === SpanType.LLM) {
      const model =
        attributes["ai.model.id"] ||
        attributes["gen_ai.request.model"] ||
        attributes["gen_ai.response.model"];
      if (model)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.LLM_MODEL,
          String(model),
        );

      let input = attributes["ai.prompt"];
      if (!input && attributes["ai.prompt.messages"]) {
        input = this.ensureString(attributes["ai.prompt.messages"]);
      }
      if (input)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_INPUT,
          this.ensureString(input),
        );

      let output = attributes["ai.response.text"];
      if (!output && attributes["ai.response.object"]) {
        output = this.ensureString(attributes["ai.response.object"]);
      }
      if (!output && attributes["ai.response.toolCalls"]) {
        output = this.ensureString(attributes["ai.response.toolCalls"]);
      }
      if (output)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_OUTPUT,
          this.ensureString(output),
        );

      if (!ROOT_VERCEL_SPANS.has(spanName)) {
        const inputTokens =
          attributes["ai.usage.inputTokens.total"] ||
          attributes["gen_ai.usage.input_tokens"] ||
          attributes["ai.usage.promptTokens"];

        if (inputTokens !== undefined) {
          setDefaultSpanAttribute(
            attributes,
            ConfidentAttr.LLM_INPUT_TOKEN_COUNT,
            Number(inputTokens),
          );
        }
        const outputTokens =
          attributes["ai.usage.outputTokens.total"] ||
          attributes["gen_ai.usage.output_tokens"] ||
          attributes["ai.usage.completionTokens"];

        if (outputTokens !== undefined) {
          setDefaultSpanAttribute(
            attributes,
            ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT,
            Number(outputTokens),
          );
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
      if (toolName) attributes[ConfidentAttr.TOOL_NAME] = String(toolName);

      const args = attributes["ai.toolCall.args"];
      if (args)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_INPUT,
          this.ensureString(args),
        );

      const result = attributes["ai.toolCall.result"];
      if (result)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_OUTPUT,
          this.ensureString(result),
        );

      const toolId = attributes["ai.toolCall.id"];
      if (toolId)
        attributes[ConfidentAttr.SPAN_METADATA_TOOL_ID] = String(toolId);
    } else if (type === SpanType.RETRIEVER) {
      const embedder = attributes["ai.model.id"];
      if (embedder)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.RETRIEVER_EMBEDDER,
          String(embedder),
        );

      const val = attributes["ai.value"] || attributes["ai.values"];
      if (val)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_INPUT,
          this.ensureString(val),
        );

      const embedding =
        attributes["ai.embedding"] || attributes["ai.embeddings"];
      if (embedding)
        setDefaultSpanAttribute(
          attributes,
          ConfidentAttr.SPAN_OUTPUT,
          this.ensureString(embedding),
        );

      if (!ROOT_VERCEL_SPANS.has(spanName)) {
        this.collectMetadata(attributes, metadata, "ai.usage.tokens", "tokens");
      }
    }

    if (Object.keys(metadata).length > 0) {
      attributes[ConfidentAttr.SPAN_METADATA] = JSON.stringify(metadata);
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
    const traceId = attributes[ConfidentAttr.INTERNAL_TRACE_UUID] as string;
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

    let inputObj = attributes[ConfidentAttr.SPAN_INPUT];
    let outputObj = attributes[ConfidentAttr.SPAN_OUTPUT];
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
    if (attributes[ConfidentAttr.SPAN_METADATA]) {
      try {
        metadataObj = JSON.parse(attributes[ConfidentAttr.SPAN_METADATA]);
      } catch {
        // Ignore parsing normal strings
      }
    }

    deepEvalSpan.error = attributes["error"]
      ? String(attributes["error"])
      : undefined;
    // These fields can also have been set from user code (`next*Span`,
    // `updateCurrentSpan`), which outranks what the SDK reported: assign only
    // where the attribute carries something and the span does not.
    if (inputObj !== undefined && deepEvalSpan.input === undefined) {
      deepEvalSpan.input = inputObj;
    }
    if (outputObj !== undefined && deepEvalSpan.output === undefined) {
      deepEvalSpan.output = outputObj;
    }
    if (
      attributes[ConfidentAttr.SPAN_METRIC_COLLECTION] !== undefined &&
      deepEvalSpan.metricCollection === undefined
    ) {
      deepEvalSpan.metricCollection =
        attributes[ConfidentAttr.SPAN_METRIC_COLLECTION];
    }
    if (metadataObj !== undefined && deepEvalSpan.metadata === undefined) {
      deepEvalSpan.metadata = metadataObj;
    }

    if (deepEvalSpan.type === SpanType.LLM) {
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
    } else if (deepEvalSpan.type === SpanType.TOOL) {
      deepEvalSpan.name = attributes[ConfidentAttr.TOOL_NAME]
        ? String(attributes[ConfidentAttr.TOOL_NAME])
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
        attributes[ConfidentAttr.RETRIEVER_EMBEDDER] || "unknown";
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

    const rootParentId =
      (span as any).parentSpanId || (span as any).parentSpanContext?.spanId;
    if (!rootParentId) {
      endOtelImplicitTrace(traceId);
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
