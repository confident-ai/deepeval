import { getCurrentSpan } from "../tracing";
import { SpanType, ToolSpan, TraceSpanStatus } from "../tracing/tracing";
import { OutputParameters } from "./types";

function fmtUrl(u?: string) {
  if (!u) return "";
  try {
    const url = new URL(u);
    const redacted =
      url.protocol === "data:"
        ? "[data-uri]"
        : `${url.protocol}//${url.host}${url.pathname}`;
    return redacted;
  } catch {
    return "[invalid-url]";
  }
}

export function compactDump(x: any, max = 2000): string {
  try {
    const s = JSON.stringify(x);
    return s.length > max ? s.slice(0, max) + "…" : s;
  } catch {
    return String(x);
  }
}

export function stringifyMultimodalContent(content: any): string {
  if (content == null) {
    return "";
  }

  if (typeof content === "string") {
    return content;
  }

  if (
    content instanceof Uint8Array ||
    (typeof Buffer !== "undefined" && content instanceof Buffer)
  ) {
    return `[bytes:${content.byteLength}]`;
  }

  if (Array.isArray(content)) {
    const parts: string[] = [];

    for (const part of content) {
      const s = stringifyMultimodalContent(part);
      if (s) parts.push(s);
    }

    return parts.join("\n");
  }

  if (typeof content === "object") {
    const t = content.type;

    if (t === "text") {
      return String(content.text ?? "");
    }

    if (t === "image_url" || t === "input_image") {
      const image_url =
        typeof content.image_url === "string"
          ? content.image_url
          : (content.image_url?.url ?? content.url);
      return `[image:${fmtUrl(image_url)}]`;
    }

    if (t === "input_text") {
      return String(content.text ?? "");
    }

    if (typeof t === "string" && t.startsWith("input_")) {
      return `[${t}]`;
    }
  }

  return compactDump(content);
}

export function renderMessages(
  messages: Array<any>,
): Array<Record<string, any>> {
  const out: Array<Record<string, any>> = [];

  for (const message of messages ?? []) {
    const role = message?.role;
    const content = message?.content;

    const toolCalls: any[] = Array.isArray(message?.toolCalls)
      ? message.toolCalls
      : Array.isArray(message?.tool_calls)
        ? message.tool_calls
        : [];

    if (role === "assistant" && toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        const toolType = toolCall?.type ?? "function";
        let name = "";
        let argumentsRaw: any = "";

        if (toolType === "function") {
          name = toolCall?.function?.name ?? "";
          argumentsRaw = toolCall?.function?.arguments ?? "";
        } else if (toolType === "custom") {
          name = toolCall?.custom?.name ?? "";
          argumentsRaw = toolCall?.custom?.input ?? "";
        }

        out.push({
          id: toolCall?.id ?? "",
          callId: toolCall?.id ?? "",
          name,
          type: toolType,
          arguments: safeJson(argumentsRaw),
        });
      }
    } else if (role === "tool") {
      out.push({
        callId: message?.toolCallId ?? message?.tool_call_id ?? "",
        type: role,
        output: message?.content ?? {},
      });
    } else {
      out.push({ role, content });
    }
  }

  return out;
}

export function renderResponseInput(
  input: Array<Record<string, any>>,
): Array<Record<string, any>> {
  const out: Array<Record<string, any>> = [];

  for (const item of input ?? []) {
    const type = item.type;
    const role = item.role;

    if (type === "message") {
      out.push({ role, content: item.content });
    } else {
      out.push(item);
    }
  }

  return out;
}

export function safeJson(x: any) {
  if (typeof x === "string") {
    try {
      return JSON.parse(x);
    } catch {
      return x;
    }
  }
  return x;
}

export function createChildToolSpans(outputParameters: OutputParameters) {
  if (!outputParameters.toolsCalled) {
    return;
  }

  const currentSpan = getCurrentSpan();
  if (!currentSpan) {
    console.log(`[createChildToolSpans]: getCurrentSpan() returned undefined`);

    return;
  }

  for (const toolCalled of outputParameters.toolsCalled) {
    const toolSpan: ToolSpan = {
      uuid: crypto.randomUUID(),
      traceUuid: currentSpan.traceUuid,
      parentUuid: currentSpan.uuid,
      startTime: currentSpan.startTime,
      endTime: currentSpan.endTime,
      status: TraceSpanStatus.SUCCESS,
      children: [],
      name: toolCalled.name,
      input: toolCalled.inputParameters,
      output: null,
      // metrics: null,
      description: toolCalled.description,
      type: SpanType.TOOL,
    };

    if (!currentSpan.children) {
      currentSpan.children = [];
    }

    currentSpan.children.push(toolSpan);
  }
}
