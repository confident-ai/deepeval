import { ChatGeneration } from "@langchain/core/outputs";
import { AIMessage } from "@langchain/core/messages";
import cliProgress from "cli-progress";

import { traceManager } from "../../tracing";
import { BaseMetric } from "../../metrics/base-metrics";
import {
  SpanType,
  TraceSpanStatus,
  AgentSpan,
  LlmSpan,
  RetrieverSpan,
  ToolSpan,
  BaseSpan,
  setCurrentTrace,
  setCurrentSpan,
  getCurrentTrace,
  getCurrentSpan,
} from "../../tracing/tracing";

enum MessageRole {
  System = "system",
  Assistant = "assistant",
  AI = "ai",
  User = "user",
  Human = "human",
  Tool = "tool",
  Function = "function",
}
const VALID_ROLES = Object.values(MessageRole);

interface Message {
  role: string;
  content: string;
}

type ParsePromptsToMessagesKwargs = {
  invocationParams?: {
    tools?: any[];
    [key: string]: any;
  };
  [key: string]: any;
};

export function parsePromptsToMessages(
  prompts: string[],
  kwargs?: ParsePromptsToMessagesKwargs,
): Message[] {
  const messages: Message[] = [];

  let currentRole: string | null = null;
  let currentContent: string[] = [];

  for (const prompt of prompts) {
    for (const rawLine of prompt.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line) {
        continue;
      }

      const idx = line.indexOf(":");
      let firstWord = "";
      let rest = "";
      let role: MessageRole | undefined;

      if (idx !== -1) {
        firstWord = line.slice(0, idx).trim();
        rest = line.slice(idx + 1).trim();
        if (VALID_ROLES.includes(firstWord.toLowerCase() as MessageRole)) {
          role = firstWord.toLowerCase() as MessageRole;
        }
      }

      if (role) {
        if (currentRole && currentContent.length > 0) {
          messages.push({
            role: currentRole,
            content: currentContent.join("\n").trim(),
          });
        }
        currentRole = role;
        currentContent = [rest];
      } else {
        if (!currentRole) {
          currentRole = "human";
        }
        currentContent.push(line);
      }
    }

    if (currentRole && currentContent.length > 0) {
      messages.push({
        role: currentRole,
        content: currentContent.join("\n").trim(),
      });
      currentRole = null;
      currentContent = [];
    }
  }

  const tools = kwargs?.invocationParams?.tools;
  if (tools && Array.isArray(tools)) {
    for (const tool of tools) {
      messages.push({ role: "Tool Input", content: String(tool) });
    }
  }

  return messages;
}

// Confident AI expects `toolsCalled[].inputParameters` to be an object. LangChain
// can hand us the tool input as a JSON string, so normalize it to an object —
// mirrors Python's prepare_tool_call_input_parameters.
export function prepareToolCallInputParameters(
  input: any,
): Record<string, any> {
  let res = input;
  if (typeof res === "string") {
    try {
      res = JSON.parse(res);
    } catch {
      // leave as string; wrapped below
    }
  }
  if (res === null || res === undefined || typeof res !== "object") {
    return { output: res };
  }
  if (Array.isArray(res)) {
    return { output: res };
  }
  return res;
}

interface EnterCurrentContextParams {
  spanType?: SpanType;
  funcName: string;
  metrics?: BaseMetric[];
  metricCollection?: string;
  observeKwargs?: Record<string, any>;
  functionKwargs?: Record<string, any>;
  progress?: typeof cliProgress;
  pbarCallbackId?: number;
  uuidStr?: string;
  // When set, the span is placed using these explicit ids instead of the
  // AsyncLocalStorage "current span/trace". The LangGraph server invokes graphs
  // across async boundaries where ALS context is not preserved, so the caller
  // (the callback handler) resolves trace/parent from run_id/parent_run_id and
  // passes them in here. `traceUuidOverride` must reference an active trace.
  traceUuidOverride?: string;
  parentUuidOverride?: string;
}

export const enterCurrentContext = ({
  spanType,
  funcName,
  metrics,
  metricCollection,
  observeKwargs,
  functionKwargs,
  uuidStr,
  traceUuidOverride,
  parentUuidOverride,
}: EnterCurrentContextParams) => {
  const startTime = new Date();
  observeKwargs = observeKwargs ?? {};
  functionKwargs = functionKwargs ?? {};

  const name = funcName ?? observeKwargs["name"];
  const prompt = observeKwargs["prompt"] ?? "";
  const uuidString = uuidStr ?? crypto.randomUUID();

  let traceUuid: string | undefined;
  let parentUuid: string | undefined;

  if (traceUuidOverride) {
    // Caller-driven placement (keyed off run_id/parent_run_id). Do not consult
    // AsyncLocalStorage: it is unreliable under the LangGraph server.
    traceUuid = traceUuidOverride;
    parentUuid = parentUuidOverride;
  } else {
    const parentSpan = getCurrentSpan();
    if (parentSpan) {
      parentUuid = parentSpan.uuid;
      traceUuid = parentSpan.traceUuid;
    } else {
      const currentTrace = getCurrentTrace();
      if (currentTrace) {
        traceUuid = currentTrace.uuid;
      } else {
        const trace = traceManager.startNewTrace();
        traceUuid = trace.uuid;
        setCurrentTrace(trace);
      }
    }
  }

  const spanKwargs = {
    uuid: uuidString,
    traceUuid: traceUuid,
    parentUuid: parentUuid,
    startTime: startTime,
    endTime: undefined,
    status: TraceSpanStatus.SUCCESS,
    children: [],
    name: name,
    input: undefined,
    output: undefined,
    metrics: metrics,
    metricCollection: metricCollection,
  };

  let spanInstance;
  const spanTypeValue = spanType ?? observeKwargs["type"];

  if (spanTypeValue === SpanType.AGENT) {
    const availableTools = observeKwargs["availableTools"] ?? [];
    const agentHandoffs = observeKwargs["agentHandoffs"] ?? [];

    spanInstance = new AgentSpan({
      ...spanKwargs,
      availableTools: availableTools,
      agentHandoffs: agentHandoffs,
      type: SpanType.AGENT,
    });
  } else if (spanTypeValue === SpanType.LLM) {
    const model = observeKwargs["model"] ?? "";
    const costPerInputToken = observeKwargs["costPerInputToken"] ?? 0;
    const costPerOutputToken = observeKwargs["costPerOutputToken"] ?? 0;

    spanInstance = new LlmSpan({
      ...spanKwargs,
      model,
      costPerInputToken: costPerInputToken,
      costPerOutputToken: costPerOutputToken,
      type: SpanType.LLM,
    });
  } else if (spanTypeValue === SpanType.RETRIEVER) {
    const embedder = observeKwargs["embedder"] ?? "";

    spanInstance = new RetrieverSpan({
      ...spanKwargs,
      embedder,
      type: SpanType.RETRIEVER,
    });
  } else if (spanTypeValue === SpanType.TOOL) {
    spanInstance = new ToolSpan({
      ...spanKwargs,
      ...observeKwargs,
      type: SpanType.TOOL,
    });
  } else {
    spanInstance = new BaseSpan({
      ...spanKwargs,
      type: spanTypeValue,
    });
  }

  spanInstance.input = traceManager.mask(functionKwargs);
  if (spanInstance instanceof LlmSpan && prompt) {
    spanInstance["prompt"] = prompt;
  }

  traceManager.addSpan(spanInstance);
  traceManager.addSpanToTrace(spanInstance);

  setCurrentSpan(spanInstance);

  return spanInstance;
};

type UpdateSpanPropertiesFn = ((span: BaseSpan) => void) | undefined;
interface ExitCurrentContextParams {
  uuidStr: string;
  result?: any;
  updateSpanProperties?: UpdateSpanPropertiesFn;
  progress?: typeof cliProgress | any;
  pbarCallbackId?: number;
  excType?: any;
  excVal?: any;
  excTb?: any;
}

export const exitCurrentContext = ({
  uuidStr,
  result,
  updateSpanProperties,
  excType,
  excVal,
}: ExitCurrentContextParams): void => {
  const endTime = new Date();

  let currentSpan = getCurrentSpan();
  if (!currentSpan) {
    const recoveredSpan = traceManager.getSpanByUuid(uuidStr);
    if (recoveredSpan) {
      console.warn(
        "[DeepEval] Async context lost, restoring span from traceManager for",
        uuidStr,
      );
      setCurrentSpan(recoveredSpan);
      currentSpan = recoveredSpan;
    }
  }

  if (!currentSpan || currentSpan.uuid !== uuidStr) {
    console.error(
      `Error: Current span in context does not match the span being exited. Expected UUID: ${uuidStr}, Got: ${
        currentSpan ? currentSpan.uuid : "None"
      }`,
    );
    return;
  }

  currentSpan.endTime = endTime;
  if (excType != null && excType != null) {
    currentSpan.status = TraceSpanStatus.ERRORED;
    currentSpan.error = String(excVal);
  } else {
    currentSpan.status = TraceSpanStatus.SUCCESS;
  }

  if (updateSpanProperties) {
    updateSpanProperties(currentSpan);
  }

  if (!currentSpan.output) {
    if (typeof traceManager.mask === "function") {
      currentSpan.output = traceManager.mask(result);
    } else {
      currentSpan.output = result;
    }
  }

  traceManager.removeSpan(uuidStr);
  if (currentSpan.parentUuid) {
    const parentSpan = traceManager.getSpanByUuid(currentSpan.parentUuid);
    if (parentSpan) {
      setCurrentSpan(parentSpan);
    } else {
      setCurrentSpan(null);
    }
  } else {
    // Root span finished. Resolve the owning trace by the span's OWN traceUuid
    // rather than the AsyncLocalStorage "current trace": under the LangGraph
    // server the ALS context is routinely lost across callbacks, which used to
    // leave the trace un-finalized (and therefore never posted). traceManager is
    // a global registry, so currentSpan.traceUuid is always reliable.
    const owningTrace =
      traceManager.getTraceByUuid(currentSpan.traceUuid) ?? getCurrentTrace();
    if (currentSpan.status === TraceSpanStatus.ERRORED && owningTrace) {
      owningTrace.status = TraceSpanStatus.ERRORED;
    }

    const otherActiveSpans = Array.from(
      traceManager.getActiveSpans().values(),
    ).filter((span) => span.traceUuid === currentSpan.traceUuid);

    if (otherActiveSpans.length === 0) {
      traceManager.endTrace(currentSpan.traceUuid);
      setCurrentTrace(null);
    }

    setCurrentSpan(null);
  }
};

export function extractName(
  serialized: Record<string, any>,
  kwargs: Record<string, any> = {},
): string {
  if ("name" in kwargs && kwargs["name"]) {
    return kwargs["name"];
  }

  if ("name" in serialized && serialized["name"]) {
    return serialized["name"];
  }

  return "Langchain Agent";
}

export function safeExtractTokenUsage(responseMetadata: Record<string, any>) {
  let promptTokens = 0;
  let completionTokens = 0;

  const tokenUsage = responseMetadata?.["tokenUsage"];

  if (tokenUsage && typeof tokenUsage === "object") {
    promptTokens =
      typeof tokenUsage["promptTokens"] === "number"
        ? tokenUsage["promptTokens"]
        : 0;
    completionTokens =
      typeof tokenUsage["completionTokens"] === "number"
        ? tokenUsage["completionTokens"]
        : 0;
  }

  return { inputTokens: promptTokens, outputTokens: completionTokens };
}

export function isChatGeneration(gen: any): gen is ChatGeneration {
  const result =
    gen && typeof gen === "object" && "message" in gen && "text" in gen;
  return result;
}

export function isAIMessage(message: any): message is AIMessage {
  const result =
    message !== null &&
    typeof message === "object" &&
    "tool_calls" in message &&
    "invalid_tool_calls" in message;
  return result;
}
