/**
 * OpenInference, the conventions of the `@arizeai/openinference-instrumentation-*`
 * packages, mapped as on the platform.
 *
 * A span's kind is `openinference.span.kind`; content lives in flattened
 * attributes such as `llm.input_messages.0.message.content`, with
 * `input.value` / `output.value` as the fallback.
 */

import type { Attributes } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { SpanType, type Trace } from "@/tracing/tracing";
import type { Adapter } from "@/tracing/otel/adapters/base";
import {
  attrJsonObject,
  attrNumber,
  attrString,
  attrStringList,
  firstDefined,
  structuredValue,
} from "@/tracing/otel/attribute-values";
import {
  answerContent,
  flattenMessages,
  messageToolCalls,
} from "@/tracing/otel/messages";

export const SCOPE = "@arizeai/openinference-instrumentation";
export const SPAN_KIND = "openinference.span.kind";
const KIND_TYPES: Record<string, SpanType> = {
  LLM: SpanType.LLM,
  EMBEDDING: SpanType.LLM,
  AGENT: SpanType.AGENT,
  TOOL: SpanType.TOOL,
  RETRIEVER: SpanType.RETRIEVER,
  RERANKER: SpanType.RETRIEVER,
};

export const openInferenceAdapter: Adapter = {
  detect(span) {
    return SPAN_KIND in span.attributes;
  },

  spanType(span, atStart) {
    const kind = span.attributes[SPAN_KIND];
    if (typeof kind !== "string") return undefined;
    const type = KIND_TYPES[kind] ?? SpanType.CUSTOM;
    // Nothing ran under a tool's name, as in a handoff between agents.
    return type === SpanType.TOOL &&
      !atStart &&
      !("tool.name" in span.attributes)
      ? SpanType.CUSTOM
      : type;
  },

  spanFields(span, type) {
    const attributes = span.attributes;
    const fields: Record<string, unknown> = {
      input: value(attributes, "input"),
      output: value(attributes, "output"),
      metadata: attrJsonObject(attributes, "metadata"),
    };
    if (type === SpanType.LLM) Object.assign(fields, llmFields(attributes));
    else if (type === SpanType.TOOL) {
      const args = attributes["tool_call.function.arguments"];
      if (args !== undefined)
        fields.input = firstDefined(structuredValue(args), args);
      fields.description = attrString(attributes, "tool.description");
    } else if (type === SpanType.RETRIEVER) {
      const documents = indexed(attributes, "retrieval.documents.")
        .map((item) => (item as Record<string, unknown>).document)
        .filter((document) => document && typeof document === "object");
      if (documents.length > 0) fields.output = documents;
    }
    return fields;
  },

  traceFields(span): Partial<Trace> {
    const attributes = span.attributes;
    return {
      threadId: attrString(attributes, "session.id"),
      userId: attrString(attributes, "user.id"),
      tags: attrStringList(attributes, "tag.tags"),
    };
  },
};

function llmFields(attributes: Attributes): Record<string, unknown> {
  const fields: Record<string, unknown> = {
    model: attrString(attributes, "llm.model_name"),
    inputTokenCount: attrNumber(attributes, "llm.token_count.prompt"),
    outputTokenCount: attrNumber(attributes, "llm.token_count.completion"),
  };
  const input = messages(attributes, "llm.input_messages.");
  if (input) fields.input = flattenMessages(input);
  const output = messages(attributes, "llm.output_messages.");
  if (output) {
    fields.output = answerContent(flattenMessages(output));
    fields.toolsCalled = messageToolCalls(output);
  }
  return fields;
}

/** `input.value` / `output.value`, decoded when its mime type is JSON. */
function value(attributes: Attributes, prefix: string): unknown {
  const raw = attributes[`${prefix}.value`];
  if (
    typeof raw === "string" &&
    attributes[`${prefix}.mime_type`] === "application/json"
  )
    return firstDefined(structuredValue(raw), raw);
  return raw;
}

/** Flattened OpenInference messages as GenAI `{role, parts}` messages. */
function messages(
  attributes: Attributes,
  prefix: string,
): Record<string, unknown>[] | undefined {
  const result: Record<string, unknown>[] = [];
  for (const flattened of indexed(attributes, prefix)) {
    const message = (flattened as Record<string, unknown>).message;
    if (!message || typeof message !== "object") continue;
    const fields = message as Record<string, unknown>;
    const role = typeof fields.role === "string" ? fields.role : "user";
    let parts: Record<string, unknown>[];
    if (role === "tool")
      parts = [
        {
          type: "tool_call_response",
          response: fields.content,
          id: fields.tool_call_id,
        },
      ];
    else {
      parts = [];
      const contents = listed(fields.contents);
      // `contents` is the richer form of `content`, when both come.
      if (contents.length === 0 && typeof fields.content === "string")
        parts.push({ type: "text", content: fields.content });
      for (const item of contents) {
        const part = (item as Record<string, unknown>)?.message_content;
        if (part && typeof part === "object") {
          const text = (part as Record<string, unknown>).text;
          parts.push(
            typeof text === "string"
              ? { type: "text", content: text }
              : (part as Record<string, unknown>),
          );
        }
      }
      for (const item of listed(fields.tool_calls)) {
        const call = (item as Record<string, unknown>)?.tool_call as
          | Record<string, unknown>
          | undefined;
        const fn = call?.function as Record<string, unknown> | undefined;
        if (fn)
          parts.push({
            type: "tool_call",
            id: call?.id,
            name: fn.name,
            arguments: fn.arguments,
          });
      }
    }
    result.push({ role, parts });
  }
  return result.length > 0 ? result : undefined;
}

/**
 * `{prefix}{i}.{path}` attributes as one nested value per index;
 * `llm.input_messages.0.message.role` becomes `[{message: {role}}]`.
 */
function indexed(attributes: Attributes, prefix: string): unknown[] {
  const root: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(attributes)) {
    if (!key.startsWith(prefix)) continue;
    const path = key.slice(prefix.length).split(".");
    const leaf = path.pop()!;
    let node: Record<string, unknown> = root;
    for (const segment of path) {
      const child = node[segment];
      if (child !== undefined && (typeof child !== "object" || child === null))
        break;
      node = (node[segment] ??= {}) as Record<string, unknown>;
    }
    node[leaf] = item;
  }
  return listed(root);
}

/** A digit-keyed object's values in index order. */
function listed(node: unknown): unknown[] {
  if (!node || typeof node !== "object") return [];
  return Object.keys(node)
    .filter((key) => /^\d+$/.test(key))
    .sort((a, b) => Number(a) - Number(b))
    .map((key) => (node as Record<string, unknown>)[key]);
}
