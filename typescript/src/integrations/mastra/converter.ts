import {
  SpanType,
  TraceSpanStatus,
  BaseSpan,
  AgentSpan,
  LlmSpan,
  RetrieverSpan,
  ToolSpan,
} from "../../tracing/tracing";
import { ToolCall } from "../../test-case";
import {
  MastraSpanType,
  MastraExportedSpan,
  MastraUsageStats,
} from "./mastra-types";

const SPAN_TYPE_EXCEPTIONS: Record<string, SpanType> = {
  [MastraSpanType.AGENT_RUN]: SpanType.AGENT,
  [MastraSpanType.WORKFLOW_RUN]: SpanType.AGENT,
  [MastraSpanType.MODEL_GENERATION]: SpanType.LLM,
  [MastraSpanType.TOOL_CALL]: SpanType.TOOL,
  [MastraSpanType.MCP_TOOL_CALL]: SpanType.TOOL,
  [MastraSpanType.PROVIDER_TOOL_CALL]: SpanType.TOOL,
  [MastraSpanType.CLIENT_TOOL_CALL]: SpanType.TOOL,
  [MastraSpanType.RAG_EMBEDDING]: SpanType.RETRIEVER,
  [MastraSpanType.RAG_VECTOR_OPERATION]: SpanType.RETRIEVER,
};

export function mapSpanType(mastraType: string): SpanType {
  return SPAN_TYPE_EXCEPTIONS[mastraType] ?? SpanType.CUSTOM;
}

const DROPPED_SPAN_TYPES = new Set<string>([MastraSpanType.MODEL_CHUNK]);

export function shouldDropSpan(span: MastraExportedSpan): boolean {
  return span.isEvent === true || DROPPED_SPAN_TYPES.has(span.type);
}

function toDate(value: Date | string | undefined): Date | undefined {
  if (value === undefined || value === null) return undefined;
  return value instanceof Date ? value : new Date(value);
}

export function extractUsage(usage?: MastraUsageStats): {
  inputTokenCount?: number;
  outputTokenCount?: number;
} {
  if (!usage) return {};
  return {
    inputTokenCount:
      typeof usage.inputTokens === "number" ? usage.inputTokens : undefined,
    outputTokenCount:
      typeof usage.outputTokens === "number" ? usage.outputTokens : undefined,
  };
}

export function getToolName(span: MastraExportedSpan): string {
  if (span.entityName) return span.entityName;
  const quoted = span.name?.match(/'([^']+)'/)?.[1];
  return quoted ?? span.name;
}

function buildMetadata(
  span: MastraExportedSpan,
  deepevalType: SpanType,
): Record<string, any> | undefined {
  const attrs = span.attributes ?? {};
  const metadata: Record<string, any> = { ...(span.metadata ?? {}) };

  metadata.mastraSpanType = span.type;

  if (deepevalType === SpanType.LLM) {
    if (attrs.provider !== undefined) metadata.provider = attrs.provider;
    if (attrs.finishReason !== undefined)
      metadata.finishReason = attrs.finishReason;
    if (attrs.responseId !== undefined) metadata.responseId = attrs.responseId;
    if (attrs.streaming !== undefined) metadata.streaming = attrs.streaming;
    if (attrs.completionStartTime !== undefined)
      metadata.completionStartTime =
        attrs.completionStartTime instanceof Date
          ? attrs.completionStartTime.toISOString()
          : attrs.completionStartTime;
    if (attrs.parameters !== undefined)
      metadata.modelParameters = attrs.parameters;
    if (attrs.costContext?.estimatedCost !== undefined)
      metadata.estimatedCost = attrs.costContext.estimatedCost;

    const inDetails = attrs.usage?.inputDetails;
    const outDetails = attrs.usage?.outputDetails;
    if (inDetails?.cacheRead !== undefined)
      metadata.cacheReadInputTokens = inDetails.cacheRead;
    if (inDetails?.cacheWrite !== undefined)
      metadata.cacheWriteInputTokens = inDetails.cacheWrite;
    if (outDetails?.reasoning !== undefined)
      metadata.reasoningTokens = outDetails.reasoning;
  } else if (deepevalType === SpanType.RETRIEVER) {
    if (attrs.dimensions !== undefined) metadata.dimensions = attrs.dimensions;
    if (attrs.inputCount !== undefined) metadata.inputCount = attrs.inputCount;
  } else if (deepevalType === SpanType.TOOL) {
    if (attrs.toolCallId !== undefined) metadata.toolCallId = attrs.toolCallId;
    if (attrs.mcpServer !== undefined) metadata.mcpServer = attrs.mcpServer;
    if (attrs.success !== undefined) metadata.success = attrs.success;
  }

  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

export interface BuildSpanOptions {
  metricCollection?: string;
  prompt?: {
    _alias?: string | null;
    hash?: string;
    version?: string | null;
    label?: string | null;
  };
}

export function buildDeepEvalSpan(
  span: MastraExportedSpan,
  traceUuid: string,
  options: BuildSpanOptions = {},
): BaseSpan {
  const deepevalType = mapSpanType(span.type);
  const attrs = span.attributes ?? {};
  const name = span.name ?? span.entityName;

  const common = {
    uuid: span.id,
    status: TraceSpanStatus.IN_PROGRESS,
    traceUuid,
    parentUuid: span.parentSpanId,
    startTime: toDate(span.startTime) ?? new Date(),
    input: span.input,
    metadata: buildMetadata(span, deepevalType),
    metricCollection: options.metricCollection,
  };

  switch (deepevalType) {
    case SpanType.LLM: {
      const llm = new LlmSpan({
        ...common,
        type: SpanType.LLM,
        name,
        model: attrs.responseModel ?? attrs.model ?? "unknown",
        ...extractUsage(attrs.usage),
      });
      if (options.prompt) {
        if (options.prompt._alias) llm.promptAlias = options.prompt._alias;
        if (options.prompt.hash) llm.promptCommitHash = options.prompt.hash;
        if (options.prompt.version) llm.promptVersion = options.prompt.version;
        if (options.prompt.label) llm.promptLabel = options.prompt.label;
      }
      return llm;
    }
    case SpanType.RETRIEVER:
      return new RetrieverSpan({
        ...common,
        type: SpanType.RETRIEVER,
        name,
        embedder: attrs.model ?? "unknown",
        topK: attrs.topK,
      });
    case SpanType.TOOL:
      return new ToolSpan({
        ...common,
        type: SpanType.TOOL,
        name: getToolName(span),
        description: attrs.toolDescription,
      });
    case SpanType.AGENT:
      return new AgentSpan({
        ...common,
        type: SpanType.AGENT,
        name,
        availableTools: attrs.availableTools ?? [],
        agentHandoffs: [],
      });
    default:
      return new BaseSpan({ ...common, type: SpanType.CUSTOM, name });
  }
}

export function updateDeepEvalSpan(
  target: BaseSpan,
  span: MastraExportedSpan,
): void {
  const attrs = span.attributes ?? {};
  const deepevalType = target.type as SpanType;

  if (span.output !== undefined) target.output = span.output;
  if (span.input !== undefined) target.input = span.input;

  const metadata = buildMetadata(span, deepevalType);
  if (metadata) target.metadata = metadata;

  if (deepevalType === SpanType.LLM) {
    const llm = target as LlmSpan;
    const model = attrs.responseModel ?? attrs.model;
    if (model) llm.model = model;
    const usage = extractUsage(attrs.usage);
    if (usage.inputTokenCount !== undefined)
      llm.inputTokenCount = usage.inputTokenCount;
    if (usage.outputTokenCount !== undefined)
      llm.outputTokenCount = usage.outputTokenCount;
  } else if (deepevalType === SpanType.RETRIEVER) {
    const retriever = target as RetrieverSpan;
    if (attrs.model) retriever.embedder = attrs.model;
  } else if (deepevalType === SpanType.TOOL) {
    const tool = target as ToolSpan;
    if (attrs.toolDescription) tool.description = attrs.toolDescription;
  }
}

export function finalizeDeepEvalSpan(
  target: BaseSpan,
  span: MastraExportedSpan,
): void {
  updateDeepEvalSpan(target, span);

  const endTime = toDate(span.endTime);
  if (endTime) target.endTime = endTime;

  if (span.errorInfo) {
    target.status = TraceSpanStatus.ERRORED;
    target.error = span.errorInfo.message;
  } else {
    target.status = TraceSpanStatus.SUCCESS;
  }
}

export function buildToolCall(span: MastraExportedSpan): ToolCall {
  const input = span.input;
  const inputParameters =
    input && typeof input === "object" && !Array.isArray(input)
      ? (input as Record<string, any>)
      : input !== undefined
        ? { input }
        : undefined;

  return new ToolCall({
    name: getToolName(span),
    description: span.attributes?.toolDescription,
    inputParameters,
    output: span.output,
  });
}

const RESERVED_TRACE_META_KEYS = new Set<string>([
  "userId",
  "threadId",
  "sessionId",
  "resourceId",
  "traceName",
  "testCaseId",
  "turnId",
]);

export interface PerRequestTraceContext {
  threadId?: string;
  userId?: string;
  tags?: string[];
  name?: string;
  testCaseId?: string;
  turnId?: string;
  metadata?: Record<string, any>;
}

export function extractTraceContext(
  span: MastraExportedSpan,
): PerRequestTraceContext {
  const meta = span.metadata ?? {};
  const attrs = span.attributes ?? {};
  const ctx: PerRequestTraceContext = {};

  const threadId = meta.threadId ?? meta.sessionId ?? attrs.conversationId;
  if (threadId) ctx.threadId = String(threadId);

  const userId = meta.userId ?? meta.resourceId;
  if (userId) ctx.userId = String(userId);

  if (span.tags && span.tags.length > 0) ctx.tags = span.tags;
  if (meta.traceName) ctx.name = String(meta.traceName);
  if (meta.testCaseId) ctx.testCaseId = String(meta.testCaseId);
  if (meta.turnId) ctx.turnId = String(meta.turnId);

  const custom: Record<string, any> = {};
  for (const [key, value] of Object.entries(meta)) {
    if (!RESERVED_TRACE_META_KEYS.has(key) && value !== undefined) {
      custom[key] = value;
    }
  }
  if (Object.keys(custom).length > 0) ctx.metadata = custom;

  return ctx;
}
