/** Convert confident-trace's OTel spans into deepeval spans. */

import type { Attributes, HrTime } from "@opentelemetry/api";
import { SpanStatusCode } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import type { Trace } from "@/tracing/tracing";
import {
  type AgentSpan,
  type BaseSpan,
  type LlmSpan,
  type RetrieverSpan,
  type ToolSpan,
  SpanType,
  TraceSpanStatus,
} from "@/tracing/tracing";
import { ConfidentAttr } from "@/tracing/attributes";
import { ADAPTERS, type Adapter } from "@/tracing/otel/adapters";
import {
  attrCoercedStringList,
  attrDecoded,
  attrJsonContent,
  attrJsonObject,
  attrJsonStringList,
  attrNumber,
  attrString,
  attrStringList,
  attrToolCalls,
  firstDefined,
  MISSING,
  structuredValue,
  unsupported,
} from "@/tracing/otel/attribute-values";
import { GenAIAttr } from "@/tracing/otel/genai";
import {
  latestUserText,
  llmIo,
  messageToolCalls,
  toolArguments,
  toolResponse,
  traceIo,
} from "@/tracing/otel/messages";

const CONFIDENT_TRACE_SCOPES = new Set(["confident-trace", "confident_trace"]);

const DECLARED_TYPES: Record<string, SpanType> = {
  agent: SpanType.AGENT,
  llm: SpanType.LLM,
  retriever: SpanType.RETRIEVER,
  tool: SpanType.TOOL,
  custom: SpanType.CUSTOM,
  base: SpanType.CUSTOM,
};

const OPERATION_TYPES: Record<string, SpanType> = {
  chat: SpanType.LLM,
  generate_content: SpanType.LLM,
  text_completion: SpanType.LLM,
  create_agent: SpanType.AGENT,
  invoke_agent: SpanType.AGENT,
  invoke_workflow: SpanType.AGENT,
  execute_tool: SpanType.TOOL,
  retrieval: SpanType.RETRIEVER,
};

export interface ConvertedSpan {
  span: BaseSpan;
  traceFields: Partial<Trace>;
  conversationId?: string;
}

export function spanUuid(span: ReadableSpan): string {
  return span.spanContext().spanId;
}

export function traceUuidOf(span: ReadableSpan): string {
  return span.spanContext().traceId;
}

export function hrTimeToDate(time: HrTime): Date {
  return new Date(time[0] * 1000 + time[1] / 1e6);
}

function detectAdapters(span: ReadableSpan): Adapter[] {
  return ADAPTERS.filter((adapter) => {
    try {
      return adapter.detect(span);
    } catch {
      return false;
    }
  });
}

/**
 * Classify a span.
 *
 * A declared `confident.span.type` wins, then a framework adapter, which knows
 * its framework's spans, then the GenAI conventions. `atStart` classifies a
 * span still opening: frameworks may only set its attributes later, so its
 * GenAI-convention name counts too.
 */
export function resolveSpanType(
  span: ReadableSpan,
  atStart = false,
  adapters: Adapter[] = detectAdapters(span),
): SpanType {
  const attributes = span.attributes;
  const declared = attributes[ConfidentAttr.SPAN_TYPE];
  if (typeof declared === "string" && declared in DECLARED_TYPES)
    return DECLARED_TYPES[declared];
  for (const adapter of adapters) {
    const type = adapter.spanType(span, atStart);
    if (type) return type;
  }
  const operation = attrString(attributes, GenAIAttr.OPERATION_NAME);
  if (operation) return OPERATION_TYPES[operation] ?? SpanType.CUSTOM;
  if (attrString(attributes, GenAIAttr.TOOL_NAME)) return SpanType.TOOL;
  if (atStart && declared === undefined)
    return OPERATION_TYPES[span.name.split(" ")[0]] ?? SpanType.CUSTOM;
  return SpanType.CUSTOM;
}

/**
 * confident-trace's export rule, plus any span a framework adapter reads.
 *
 * Other spans only travel as ancestors.
 */
export function isAiSpan(span: ReadableSpan): boolean {
  if (CONFIDENT_TRACE_SCOPES.has(span.instrumentationScope?.name ?? ""))
    return true;
  const prefixed = (key: string) =>
    key.startsWith("confident.") || key.startsWith("gen_ai.");
  return (
    Object.keys(span.attributes).some(prefixed) ||
    span.events.some((event) => event.name.startsWith("gen_ai.")) ||
    detectAdapters(span).length > 0
  );
}

/** Convert `span`, as `spanType` when given instead of its own. */
export function convertSpan(
  span: ReadableSpan,
  spanType?: SpanType,
): ConvertedSpan {
  const adapters = detectAdapters(span);
  const type = spanType ?? resolveSpanType(span, false, adapters);
  const converted = BUILDERS[type]?.(span) ?? baseSpan(span, type);
  for (const adapter of adapters) {
    try {
      fillUnset(converted, adapter.spanFields(span, type));
    } catch {
      // An adapter that can't read a span leaves it as recorded.
    }
  }
  if (type === SpanType.RETRIEVER && !converted.retrievalContext)
    converted.retrievalContext = documentTexts(converted.output);
  if (type === SpanType.TOOL) {
    converted.input = toolArguments(converted.input);
    converted.output = toolResponse(converted.output);
  }
  const traceFields = recordedTraceFields(span.attributes);
  for (const adapter of adapters)
    for (const [key, value] of Object.entries(adapter.traceFields(span)))
      if (value !== undefined && !(key in traceFields))
        (traceFields as Record<string, unknown>)[key] = value;
  return {
    span: converted,
    traceFields,
    conversationId: attrString(span.attributes, GenAIAttr.CONVERSATION_ID),
  };
}

/** Set each of `fields` that `target` leaves empty. */
export function fillUnset(target: object, fields: Record<string, unknown>) {
  const record = target as Record<string, unknown>;
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined) continue;
    const current = record[key];
    if (
      current === undefined ||
      current === null ||
      current === "" ||
      (Array.isArray(current) && current.length === 0)
    )
      record[key] = value;
  }
}

/** A content attribute, `MISSING` when confident-trace serialized none of it. */
function knownContent(attributes: Attributes, key: string): unknown {
  const value = attrJsonContent(attributes, key);
  return value !== MISSING && unsupported(value) ? MISSING : value;
}

function recordedTraceFields(attributes: Attributes): Partial<Trace> {
  const fields: Partial<Trace> = {
    name: attrString(attributes, ConfidentAttr.TRACE_NAME),
    tags: firstDefined(
      attrStringList(attributes, ConfidentAttr.TRACE_TAGS),
      attrJsonStringList(attributes, ConfidentAttr.TRACE_TAGS),
    ),
    metadata: attrJsonObject(attributes, ConfidentAttr.TRACE_METADATA),
    threadId: attrString(attributes, ConfidentAttr.TRACE_THREAD_ID),
    userId: attrString(attributes, ConfidentAttr.TRACE_USER_ID),
    testCaseId: attrString(attributes, ConfidentAttr.TRACE_TEST_CASE_ID),
    turnId: attrString(attributes, ConfidentAttr.TRACE_TURN_ID),
    metricCollection: attrString(
      attributes,
      ConfidentAttr.TRACE_METRIC_COLLECTION,
    ),
    retrievalContext: attrJsonStringList(
      attributes,
      ConfidentAttr.TRACE_RETRIEVAL_CONTEXT,
    ),
    context: attrJsonStringList(attributes, ConfidentAttr.TRACE_CONTEXT),
    toolsCalled: attrToolCalls(attributes, ConfidentAttr.TRACE_TOOLS_CALLED),
    expectedTools: attrToolCalls(
      attributes,
      ConfidentAttr.TRACE_EXPECTED_TOOLS,
    ),
    expectedOutput: textOf(
      attrDecoded(attributes, ConfidentAttr.TRACE_EXPECTED_OUTPUT),
    ),
  };
  for (const key of Object.keys(fields) as (keyof Trace)[])
    if (fields[key] === undefined) delete fields[key];
  for (const [key, attribute] of [
    ["input", ConfidentAttr.TRACE_INPUT],
    ["output", ConfidentAttr.TRACE_OUTPUT],
  ] as const) {
    const value = knownContent(attributes, attribute);
    if (value !== MISSING) fields[key] = traceIo(key, value);
  }
  return fields;
}

function textOf(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  return typeof value === "string" ? value : JSON.stringify(value);
}

function commonFields(
  span: ReadableSpan,
  type: SpanType,
  typedName?: string,
): BaseSpan {
  const attributes = span.attributes;
  const errored = span.status.code === SpanStatusCode.ERROR;
  return {
    uuid: spanUuid(span),
    traceUuid: traceUuidOf(span),
    parentUuid: span.parentSpanContext?.spanId,
    startTime: hrTimeToDate(span.startTime),
    endTime: hrTimeToDate(span.endTime),
    status: errored ? TraceSpanStatus.ERRORED : TraceSpanStatus.SUCCESS,
    ...(errored
      ? {
          error:
            span.status.message ||
            attrString(attributes, "error.type") ||
            "error",
        }
      : {}),
    children: [],
    type,
    name:
      attrString(attributes, ConfidentAttr.SPAN_NAME) ?? typedName ?? span.name,
    metadata: attrJsonObject(attributes, ConfidentAttr.SPAN_METADATA),
    metricCollection: attrString(
      attributes,
      ConfidentAttr.SPAN_METRIC_COLLECTION,
    ),
    retrievalContext: attrJsonStringList(
      attributes,
      ConfidentAttr.SPAN_RETRIEVAL_CONTEXT,
    ),
    context: attrJsonStringList(attributes, ConfidentAttr.SPAN_CONTEXT),
    expectedOutput: textOf(
      attrDecoded(attributes, ConfidentAttr.SPAN_EXPECTED_OUTPUT),
    ),
    toolsCalled: attrToolCalls(attributes, ConfidentAttr.SPAN_TOOLS_CALLED),
    expectedTools: attrToolCalls(attributes, ConfidentAttr.SPAN_EXPECTED_TOOLS),
  };
}

/** Declared `confident.span.*` I/O beats the type-specific I/O. */
function withDeclaredIo<T extends BaseSpan>(
  span: ReadableSpan,
  fields: T,
  input: unknown,
  output: unknown,
): T {
  const declaredInput = knownContent(span.attributes, ConfidentAttr.SPAN_INPUT);
  const declaredOutput = knownContent(
    span.attributes,
    ConfidentAttr.SPAN_OUTPUT,
  );
  fields.input = declaredInput === MISSING ? input : declaredInput;
  fields.output = declaredOutput === MISSING ? output : declaredOutput;
  return fields;
}

function baseSpan(span: ReadableSpan, type = SpanType.CUSTOM): BaseSpan {
  return withDeclaredIo(span, commonFields(span, type), undefined, undefined);
}

function llmSpan(span: ReadableSpan): LlmSpan {
  const attributes = span.attributes;
  const [input, output] = llmIo(attributes);
  const fields = commonFields(span, SpanType.LLM) as LlmSpan;
  fields.model =
    attrString(attributes, ConfidentAttr.LLM_MODEL) ??
    attrString(attributes, GenAIAttr.RESPONSE_MODEL) ??
    attrString(attributes, GenAIAttr.REQUEST_MODEL) ??
    "";
  fields.inputTokenCount = firstDefined(
    attrNumber(attributes, ConfidentAttr.LLM_INPUT_TOKEN_COUNT),
    attrNumber(attributes, GenAIAttr.USAGE_INPUT_TOKENS),
  );
  fields.outputTokenCount = firstDefined(
    attrNumber(attributes, ConfidentAttr.LLM_OUTPUT_TOKEN_COUNT),
    attrNumber(attributes, GenAIAttr.USAGE_OUTPUT_TOKENS),
  );
  fields.costPerInputToken = attrNumber(
    attributes,
    ConfidentAttr.LLM_COST_PER_INPUT_TOKEN,
  );
  fields.costPerOutputToken = attrNumber(
    attributes,
    ConfidentAttr.LLM_COST_PER_OUTPUT_TOKEN,
  );
  fields.promptAlias = attrString(attributes, ConfidentAttr.SPAN_PROMPT_ALIAS);
  fields.promptVersion = attrString(
    attributes,
    ConfidentAttr.SPAN_PROMPT_VERSION,
  );
  fields.promptLabel = attrString(attributes, ConfidentAttr.SPAN_PROMPT_LABEL);
  fields.promptCommitHash = attrString(
    attributes,
    ConfidentAttr.SPAN_PROMPT_COMMIT_HASH,
  );
  fields.toolsCalled ??= messageToolCalls(
    structuredValue(attributes[GenAIAttr.OUTPUT_MESSAGES]),
  );
  return withDeclaredIo(span, fields, input, output);
}

function agentSpan(span: ReadableSpan): AgentSpan {
  const attributes = span.attributes;
  const [input, output] = llmIo(attributes);
  const fields = commonFields(
    span,
    SpanType.AGENT,
    attrString(attributes, GenAIAttr.AGENT_NAME),
  ) as AgentSpan;
  fields.availableTools =
    attrCoercedStringList(attributes, ConfidentAttr.AGENT_AVAILABLE_TOOLS) ??
    [];
  fields.agentHandoffs =
    attrCoercedStringList(attributes, ConfidentAttr.AGENT_AGENT_HANDOFFS) ?? [];
  fields.toolsCalled ??= messageToolCalls(
    structuredValue(attributes[GenAIAttr.OUTPUT_MESSAGES]),
  );
  // An agent serves the latest request in its conversation.
  return withDeclaredIo(span, fields, latestUserText(input), output);
}

function toolSpan(span: ReadableSpan): ToolSpan {
  const attributes = span.attributes;
  const fields = commonFields(
    span,
    SpanType.TOOL,
    attrString(attributes, ConfidentAttr.TOOL_NAME) ??
      attrString(attributes, GenAIAttr.TOOL_NAME),
  ) as ToolSpan;
  fields.description =
    attrString(attributes, ConfidentAttr.TOOL_DESCRIPTION) ??
    attrString(attributes, GenAIAttr.TOOL_DESCRIPTION);
  return withDeclaredIo(
    span,
    fields,
    attrDecoded(attributes, GenAIAttr.TOOL_CALL_ARGUMENTS),
    attrDecoded(attributes, GenAIAttr.TOOL_CALL_RESULT),
  );
}

function retrieverSpan(span: ReadableSpan): RetrieverSpan {
  const attributes = span.attributes;
  const fields = commonFields(span, SpanType.RETRIEVER) as RetrieverSpan;
  fields.embedder =
    attrString(attributes, ConfidentAttr.RETRIEVER_EMBEDDER) ?? "";
  fields.topK = attrNumber(attributes, ConfidentAttr.RETRIEVER_TOP_K);
  fields.chunkSize = attrNumber(attributes, ConfidentAttr.RETRIEVER_CHUNK_SIZE);
  return withDeclaredIo(
    span,
    fields,
    attributes[GenAIAttr.RETRIEVAL_QUERY_TEXT],
    structuredValue(attributes[GenAIAttr.RETRIEVAL_DOCUMENTS]),
  );
}

const BUILDERS: Partial<Record<SpanType, (span: ReadableSpan) => BaseSpan>> = {
  [SpanType.LLM]: llmSpan,
  [SpanType.AGENT]: agentSpan,
  [SpanType.TOOL]: toolSpan,
  [SpanType.RETRIEVER]: retrieverSpan,
};

/** Retrieved documents' texts, which retrieval metrics read as context. */
function documentTexts(documents: unknown): string[] | undefined {
  if (!Array.isArray(documents) || documents.length === 0) return undefined;
  const texts: string[] = [];
  for (const document of documents) {
    if (typeof document === "string") texts.push(document);
    else if (document && typeof document === "object") {
      const record = document as Record<string, unknown>;
      const text = ["page_content", "pageContent", "text", "content"]
        .map((key) => record[key])
        .find((value) => typeof value === "string");
      if (typeof text === "string") texts.push(text);
    }
  }
  return texts.length > 0 ? texts : undefined;
}
