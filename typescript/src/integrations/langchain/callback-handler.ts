import { performance } from "perf_hooks";

import {
  BaseCallbackHandler,
  BaseCallbackHandlerInput,
  HandleLLMNewTokenCallbackFields,
  NewTokenIndices,
} from "@langchain/core/callbacks/base";
import { ChainValues } from "@langchain/core/utils/types";
import { DocumentInterface } from "@langchain/core/documents";
import { LLMResult } from "@langchain/core/outputs";
import { Serialized } from "@langchain/core/load/serializable";

import {
  enterCurrentContext,
  exitCurrentContext,
  isAIMessage,
  isChatGeneration,
  parsePromptsToMessages,
  prepareToolCallInputParameters,
  safeExtractTokenUsage,
} from "./utils";
import { RunHierarchyTracker } from "./langgraph-utils";
import { SpanType, TraceSpanStatus } from "../../tracing/tracing";
import { traceManager } from "../../tracing";
import { withCaptureTracingIntegration } from "../../telemetry";
import { BaseMetric } from "../../metrics/base-metrics";

let langchainInstalled: boolean | null = null;

function checkLangchainInstalled(): boolean {
  if (langchainInstalled !== null) {
    return langchainInstalled;
  }

  try {
    import("@langchain/core/callbacks/base");
    import("@langchain/core/outputs");
    import("@langchain/core/messages");
    langchainInstalled = true;
    return true;
  } catch (_error) {
    langchainInstalled = false;
    return false;
  }
}

function isLangchainInstalled() {
  if (!checkLangchainInstalled()) {
    throw new Error(
      "LangChain is not installed. Please install it with `npm install langchain @langchain/core`.",
    );
  }
}

export class DeepEvalCallbackHandler
  extends BaseCallbackHandler
  implements BaseCallbackHandlerInput
{
  name = "DeepEvalCallbackHandler";
  private metrics?: BaseMetric[];
  private metricCollection?: string;

  // Resolves trace/parent from run_id/parent_run_id and owns lazy per-run trace
  // creation, so this handler works under the LangGraph server (where the ALS
  // "current span/trace" is lost across callbacks). See RunHierarchyTracker.
  private hierarchy: RunHierarchyTracker;

  constructor({
    name,
    tags,
    metadata,
    threadId,
    userId,
    testCaseId,
    turnId,
    metrics,
    metricCollection,
  }: {
    name?: string;
    tags?: string[];
    metadata?: Record<string, any>;
    threadId?: string;
    userId?: string;
    testCaseId?: string;
    turnId?: string;
    metrics?: BaseMetric[];
    metricCollection?: string;
  }) {
    super();

    isLangchainInstalled();

    // NOTE: the trace is created lazily on the first callback (not here). Creating
    // it in the constructor binds a single trace at construction time, which is
    // wrong when one handler is baked into a long-lived graph (e.g. exported via
    // langgraph.json) and reused across many server requests.
    this.hierarchy = new RunHierarchyTracker({
      name,
      tags,
      metadata,
      threadId,
      userId,
      testCaseId,
      turnId,
    });
    this.metrics = metrics;
    this.metricCollection = metricCollection;

    this.captureTelemetry();
  }

  private captureTelemetry() {
    withCaptureTracingIntegration("langchain.CallbackHandler", () => {
      // NOTE(tanay): Just capture the event, don't do trace management here
    }).catch((err) => console.error("Telemetry failed:", err));
  }

  async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runType?: string,
    runName?: string,
  ) {
    const uuidStr = String(runId);
    const { traceUuid, parentUuid } = this.hierarchy.resolveContext(
      uuidStr,
      parentRunId,
    );
    // Record linkage for EVERY chain (incl. intermediate LangGraph nodes) so that
    // descendant LLM/tool callbacks can resolve their ancestry without ALS.
    this.hierarchy.recordRun(uuidStr, parentRunId, traceUuid);

    // Only the root chain gets a span (preserves the established trace shape:
    // the agent/graph as root, with LLM/tool spans beneath it).
    if (parentUuid === undefined) {
      const baseSpan = enterCurrentContext({
        uuidStr,
        spanType: SpanType.CUSTOM,
        funcName: runName ?? "Langchain Chain Run",
        traceUuidOverride: traceUuid,
        parentUuidOverride: undefined,
      });

      if (baseSpan) {
        this.hierarchy.recordSpan(uuidStr);
        baseSpan.input = inputs;

        const trace = traceManager.getTraceByUuid(traceUuid);
        if (trace) {
          trace.input = inputs;
        }

        // TODO(tanay): Add metrics to BaseSpan
        // baseSpan.metrics = this.metrics;
        baseSpan.metricCollection = this.metricCollection;
      }
    }
  }

  async handleChainEnd(
    outputs: ChainValues,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _kwargs?: { inputs?: Record<string, unknown> },
  ) {
    const uuidStr = String(runId);

    const baseSpan = traceManager.getSpanByUuid(uuidStr);
    if (baseSpan) {
      baseSpan.output = outputs;

      const trace = traceManager.getTraceByUuid(baseSpan.traceUuid);
      if (trace) {
        trace.output = outputs;
      }

      exitCurrentContext({ uuidStr: uuidStr });
    }
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleChainError(
    err: any,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _kwargs?: { inputs?: Record<string, unknown> },
  ) {
    const uuidStr = String(runId);
    const baseSpan = traceManager.getSpanByUuid(uuidStr);
    if (baseSpan) {
      baseSpan.status = TraceSpanStatus.ERRORED;
      baseSpan.error = String(err);
      exitCurrentContext({ uuidStr, excType: "error", excVal: err });
    } else {
      // Root chain may error without ever having created a span; finalize the
      // trace by ancestry so it is not left dangling.
      const traceUuid = this.hierarchy.getTraceUuid(uuidStr);
      if (traceUuid && traceManager.getTraceByUuid(traceUuid)) {
        const others = Array.from(traceManager.getActiveSpans().values()).filter(
          (s) => s.traceUuid === traceUuid,
        );
        if (others.length === 0) {
          traceManager.setTraceStatus(traceUuid, TraceSpanStatus.ERRORED);
          traceManager.endTrace(traceUuid);
        }
      }
    }
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    _tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ) {
    const uuidStr = String(runId);
    const inputMessages = parsePromptsToMessages(prompts, extraParams);
    const modelName = llm.name;

    const { traceUuid, parentUuid } = this.hierarchy.resolveContext(
      uuidStr,
      parentRunId,
    );
    this.hierarchy.recordRun(uuidStr, parentRunId, traceUuid);

    // TODO(tanay): Fix `any`
    const llmSpan: any = enterCurrentContext({
      uuidStr: uuidStr,
      spanType: SpanType.LLM,
      funcName: runName ?? "Langchain LLM Run",
      traceUuidOverride: traceUuid,
      parentUuidOverride: parentUuid,
    });
    this.hierarchy.recordSpan(uuidStr);

    llmSpan.input = inputMessages;
    llmSpan.model = modelName;

    const metrics = metadata?.["metrics"];
    const metricCollection = metadata?.["metricCollection"];
    const prompt = metadata?.["prompt"];
    llmSpan.metrics = metrics;
    llmSpan.metricCollection = metricCollection;
    llmSpan.prompt = prompt;
  }

  async handleLLMEnd(
    output: LLMResult,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _extraParams?: Record<string, unknown>,
  ) {
    const uuidStr = String(runId);
    const llmSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (!llmSpan) {
      this.hierarchy.cleanupRun(uuidStr);
      return;
    }

    let llmOutput;
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let modelName;

    for (const generations of output.generations) {
      for (const generation of generations) {
        if (isChatGeneration(generation)) {
          if (generation.message.response_metadata) {
            const responseMetadata = generation.message.response_metadata as
              | Record<string, unknown>
              | undefined;
            if (
              responseMetadata &&
              typeof responseMetadata["model_name"] === "string"
            ) {
              modelName = responseMetadata["model_name"];

              const extractedTokens = safeExtractTokenUsage(responseMetadata);
              totalInputTokens += extractedTokens.inputTokens;
              totalOutputTokens += extractedTokens.outputTokens;
            }
          }

          if (isAIMessage(generation.message)) {
            const aiMessage = generation.message;
            const toolCallsArray = [];

            if (aiMessage.tool_calls) {
              for (const toolCall of aiMessage.tool_calls) {
                toolCallsArray.push({
                  id: toolCall.id,
                  name: toolCall.name,
                  args: toolCall.args,
                });
              }
            }

            llmOutput = {
              role: "AI",
              content: aiMessage.content,
              toolCalls: toolCallsArray,
            };
          }
        }
      }
    }

    llmSpan.model = modelName;
    llmSpan.output = llmOutput;
    llmSpan.inputTokenCount = totalInputTokens > 0 ? totalInputTokens : 0;
    llmSpan.outputTokenCount = totalOutputTokens > 0 ? totalOutputTokens : 0;

    exitCurrentContext({ uuidStr: uuidStr });
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleLLMError(
    err: any,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _extraParams?: Record<string, unknown>,
  ) {
    const uuidStr = String(runId);
    const llmSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (llmSpan) {
      llmSpan.status = TraceSpanStatus.ERRORED;
      llmSpan.error = String(err);
      exitCurrentContext({ uuidStr, excType: "error", excVal: err });
    }
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleLLMNewToken(
    token: string,
    idx: NewTokenIndices,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _fields?: HandleLLMNewTokenCallbackFields,
  ) {
    const uuidStr = String(runId);
    const llmSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (!llmSpan) return;

    if (!llmSpan.tokenIntervals) {
      llmSpan.tokenIntervals = {};
    }

    const now = String(performance.now());
    llmSpan.tokenIntervals[now] = token;
  }

  async handleToolStart(
    tool: Serialized,
    input: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ) {
    const uuidStr = String(runId);

    const { traceUuid, parentUuid } = this.hierarchy.resolveContext(
      uuidStr,
      parentRunId,
    );
    this.hierarchy.recordRun(uuidStr, parentRunId, traceUuid);

    const toolSpan: any = enterCurrentContext({
      uuidStr: uuidStr,
      spanType: SpanType.TOOL,
      funcName: runName ?? "Langchain Tool Run",
      traceUuidOverride: traceUuid,
      parentUuidOverride: parentUuid,
    });
    this.hierarchy.recordSpan(uuidStr);

    toolSpan.input = input;
    if (
      tool &&
      typeof tool === "object" &&
      "kwargs" in tool &&
      tool.kwargs &&
      typeof tool.kwargs === "object" &&
      "description" in tool.kwargs
    ) {
      toolSpan.description = tool.kwargs.description;
    } else if (tool && typeof tool === "object" && "description" in tool) {
      toolSpan.description = (tool as any).description;
    }
  }

  async handleToolEnd(output: any, runId: string, _parentRunId?: string, _tags?: string[]) {
    const uuidStr = String(runId);
    const toolSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (toolSpan) {
      toolSpan.output = output;
      const trace = traceManager.getTraceByUuid(toolSpan.traceUuid);
      if (trace) {
        if (!trace.toolsCalled) {
          trace.toolsCalled = [];
        }
        const toolCall = {
          name: toolSpan.name,
          inputParameters: prepareToolCallInputParameters(toolSpan.input),
          output: output,
          description: toolSpan.description || undefined,
        };
        trace.toolsCalled.push(toolCall);
      }
      exitCurrentContext({ uuidStr: uuidStr });
    }
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleToolError(err: any, runId: string, _parentRunId?: string, _tags?: string[]) {
    const uuidStr = String(runId);
    const toolSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (toolSpan) {
      toolSpan.status = TraceSpanStatus.ERRORED;
      toolSpan.error = String(err);
      exitCurrentContext({ uuidStr, excType: "error", excVal: err });
    }
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ) {
    const uuidStr = String(runId);

    const { traceUuid, parentUuid } = this.hierarchy.resolveContext(
      uuidStr,
      parentRunId,
    );
    this.hierarchy.recordRun(uuidStr, parentRunId, traceUuid);

    const retrieverSpan = enterCurrentContext({
      uuidStr: uuidStr,
      spanType: SpanType.RETRIEVER,
      funcName: name ?? "Langchain Retriever Run",
      traceUuidOverride: traceUuid,
      parentUuidOverride: parentUuid,
      observeKwargs: {
        embedder: metadata?.["ls_embedding_provider"] ?? "unknown",
      },
    });
    this.hierarchy.recordSpan(uuidStr);
    retrieverSpan.input = query;
  }

  async handleRetrieverEnd(
    documents: DocumentInterface[],
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
  ) {
    const uuidStr = String(runId);
    const retrieverSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (retrieverSpan) {
      const outputArray = [];
      for (const document of documents) {
        outputArray.push(String(document));
      }

      retrieverSpan.output = outputArray;
      exitCurrentContext({ uuidStr: uuidStr });
    }
    this.hierarchy.cleanupRun(uuidStr);
  }

  async handleRetrieverError(
    err: any,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
  ) {
    const uuidStr = String(runId);
    const retrieverSpan: any = traceManager.getSpanByUuid(uuidStr);

    if (retrieverSpan) {
      retrieverSpan.status = TraceSpanStatus.ERRORED;
      retrieverSpan.error = String(err);
      exitCurrentContext({ uuidStr, excType: "error", excVal: err });
    }
    this.hierarchy.cleanupRun(uuidStr);
  }
}
