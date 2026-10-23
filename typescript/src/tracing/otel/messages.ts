/** Read GenAI message attributes as deepeval span content. */

import type { Attributes } from "@opentelemetry/api";

import type { ToolCall } from "@/test-case";
import { structuredValue, toolCallFrom } from "@/tracing/otel/attribute-values";
import { GenAIAttr } from "@/tracing/otel/genai";

export interface FlatMessage {
  role: string;
  content: unknown;
}

/** GenAI `{role, parts}` messages as one `{role, content}` per part. */
export function flattenMessages(messages: unknown): FlatMessage[] | undefined {
  if (!Array.isArray(messages)) return undefined;
  const flattened: FlatMessage[] = [];
  for (const message of messages) {
    if (!message || typeof message !== "object") {
      flattened.push({ role: "assistant", content: message });
      continue;
    }
    const { role, parts, content } = message as Record<string, unknown>;
    const speaker = typeof role === "string" && role ? role : "assistant";
    if (!Array.isArray(parts) || parts.length === 0) {
      flattened.push({ role: speaker, content });
      continue;
    }
    for (const part of parts) {
      const fields =
        part && typeof part === "object"
          ? (part as Record<string, unknown>)
          : undefined;
      flattened.push({
        role: speaker,
        content:
          fields?.type === "text" && typeof fields.content === "string"
            ? fields.content
            : part,
      });
    }
  }
  return flattened;
}

/** A model call's input messages and its answer, from GenAI attributes. */
export function llmIo(attributes: Attributes): [unknown, unknown] {
  const instructions = flattenMessages(
    structuredValue(attributes[GenAIAttr.SYSTEM_INSTRUCTIONS]),
  )?.map((message) => ({ ...message, role: "system" }));
  const input = flattenMessages(
    structuredValue(attributes[GenAIAttr.INPUT_MESSAGES]),
  );
  const output = flattenMessages(
    structuredValue(attributes[GenAIAttr.OUTPUT_MESSAGES]),
  );
  const messages =
    input || instructions
      ? [...(instructions ?? []), ...(input ?? [])]
      : undefined;
  return [messages, answerContent(output)];
}

/**
 * Flattened output messages as their text.
 *
 * Output without text, such as a turn that only calls tools, stays, less the
 * empty text a framework may send with it.
 */
export function answerContent(messages: unknown): unknown {
  if (!Array.isArray(messages)) return messages;
  const turn = finalAssistantTurn(messages);
  const texts = turn
    .map((message) => (message as FlatMessage)?.content)
    .filter((content): content is string => typeof content === "string");
  if (texts.some(Boolean)) return texts.filter(Boolean).join("\n");
  const rest = turn.filter(
    (message) => (message as FlatMessage)?.content !== "",
  );
  return rest.length > 0 ? rest : texts.length > 0 ? "" : turn;
}

/** A conversation's final assistant turn; a model's output is all one turn. */
function finalAssistantTurn(messages: unknown[]): unknown[] {
  let start = messages.length;
  while (
    start > 0 &&
    (messages[start - 1] as FlatMessage)?.role === "assistant"
  )
    start--;
  return start > 0 && start < messages.length
    ? messages.slice(start)
    : messages;
}

/** The last user message's text in flattened messages; anything else stays. */
export function latestUserText(messages: unknown): unknown {
  if (!Array.isArray(messages)) return messages;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i] as FlatMessage | undefined;
    if (message?.role === "user" && typeof message.content === "string")
      return message.content;
  }
  return messages;
}

/** Whether `value` is a list of GenAI `{role, parts}` messages. */
export function isMessageArray(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (message) =>
        message &&
        typeof message === "object" &&
        typeof (message as Record<string, unknown>).role === "string" &&
        Array.isArray((message as Record<string, unknown>).parts),
    )
  );
}

/**
 * Read a model's messages, which confident-trace copies to its trace.
 *
 * As on the platform, the trace takes the latest user message as input; its
 * output is the answer's text.
 */
export function traceIo(key: "input" | "output", value: unknown): unknown {
  if (!isMessageArray(value)) return value;
  const messages = flattenMessages(value);
  return key === "output" ? answerContent(messages) : latestUserText(messages);
}

/** The tool calls in GenAI output messages. */
export function messageToolCalls(messages: unknown): ToolCall[] | undefined {
  if (!Array.isArray(messages)) return undefined;
  const calls: ToolCall[] = [];
  for (const message of messages) {
    const parts = (message as Record<string, unknown>)?.parts;
    for (const part of Array.isArray(parts) ? parts : []) {
      const fields = part as Record<string, unknown>;
      if (fields?.type !== "tool_call" || typeof fields.name !== "string")
        continue;
      // Some frameworks send the arguments as JSON text.
      const call = toolCallFrom({
        name: fields.name,
        inputParameters: structuredValue(fields.arguments),
      });
      if (call) calls.push(call);
    }
  }
  return calls.length > 0 ? calls : undefined;
}

/** A tool's reply sent as one message, as its response; any other value stays. */
export function toolResponse(value: unknown): unknown {
  if (!isMessageArray(value) || (value as unknown[]).length !== 1) return value;
  const parts = (value as Record<string, unknown>[])[0].parts as unknown[];
  const part = parts.length === 1 ? (parts[0] as Record<string, unknown>) : {};
  return part?.type === "tool_call_response" && "response" in part
    ? part.response
    : value;
}

/**
 * A tool's input recorded as its call's `{args, kwargs}`, as its keyword
 * arguments when it took no positional ones; any other value stays.
 */
export function toolArguments(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const fields = value as Record<string, unknown>;
  const keys = Object.keys(fields);
  const positional = fields.args;
  return keys.every((key) => key === "args" || key === "kwargs") &&
    (!Array.isArray(positional) || positional.length === 0) &&
    fields.kwargs &&
    typeof fields.kwargs === "object"
    ? fields.kwargs
    : value;
}
