import { stringifyMultimodalContent, renderMessages } from "@/openai/utils";
import { InputParameters, OutputParameters } from "@/openai/types";
import { ToolCall, ToolCallType } from "@/test-case";

export function unwrapRequest(args: any[]): Record<string, any> {
  const first = args?.[0];
  if (first == null || typeof first !== "object") return {};
  return first.chatRequest ?? first.responsesRequest ?? first.request ?? first;
}

function toolDescriptionsOf(
  tools: any,
): Record<string, string> | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;

  const descriptions: Record<string, string> = {};
  for (const tool of tools) {
    // Tools come in either the nested `{ function: {...} }` form or a flat one.
    const spec = tool?.function ?? tool;
    const name = spec?.name;
    if (name) descriptions[name] = spec?.description;
  }
  return Object.keys(descriptions).length > 0 ? descriptions : undefined;
}

/** Tool-call arguments arrive as a JSON string; never throw on bad JSON. */
function parseArguments(args: any): Record<string, any> {
  if (args != null && typeof args === "object") return args;
  try {
    const parsed = JSON.parse(args ?? "{}");
    return parsed != null && typeof parsed === "object"
      ? parsed
      : { input: parsed };
  } catch {
    return { raw: String(args) };
  }
}

export function safeExtractInputParameters(
  args: any[],
): InputParameters {
  try {
    return extractInputParameters(unwrapRequest(args));
  } catch {
    return { model: "NA" };
  }
}

export function extractInputParameters(
  request: Record<string, any>,
): InputParameters {
  const tools = request?.tools;
  const messages = request?.messages ?? [];

  // `responses.send` takes `input`/`instructions` instead of `messages`.
  const inputPayload = request?.input;
  const instructions = request?.instructions;

  let rendered: Array<Record<string, any>>;
  let summary: string;

  if (Array.isArray(messages) && messages.length > 0) {
    rendered = renderMessages(messages);
    const firstUser = messages.find((m: any) => m?.role === "user");
    summary = stringifyMultimodalContent(firstUser?.content);
  } else {
    rendered = Array.isArray(inputPayload)
      ? renderMessages(inputPayload)
      : inputPayload != null
        ? [
            {
              role: "user",
              content: stringifyMultimodalContent(inputPayload),
            },
          ]
        : [];
    if (instructions) {
      rendered.unshift({ role: "system", content: instructions });
    }
    summary = stringifyMultimodalContent(inputPayload);
  }

  return {
    model: request?.model,
    input: summary,
    messages: rendered,
    instructions,
    tools,
    toolDescriptions: toolDescriptionsOf(tools),
  };
}

export function safeExtractOutputParameters(
  response: any,
  inputParameters: InputParameters,
): OutputParameters {
  try {
    return response?.choices
      ? extractOutputParametersFromChat(response, inputParameters)
      : extractOutputParametersFromResponses(response, inputParameters);
  } catch {
    return {};
  }
}

export function extractOutputParametersFromChat(
  response: any,
  inputParameters: InputParameters,
): OutputParameters {
  const message = response?.choices?.[0]?.message;
  const output = stringifyMultimodalContent(message?.content);

  let toolsCalled: ToolCall[] | undefined;
  const rawToolCalls = message?.toolCalls ?? message?.tool_calls;
  if (Array.isArray(rawToolCalls) && rawToolCalls.length > 0) {
    const descriptions = inputParameters.toolDescriptions ?? {};
    toolsCalled = [];
    for (const toolCall of rawToolCalls) {
      const name = toolCall?.function?.name;
      if (!name) continue;
      toolsCalled.push(
        new ToolCall({
          name,
          type: ToolCallType.FUNCTION,
          inputParameters: parseArguments(toolCall?.function?.arguments),
          description: descriptions[name],
        }),
      );
    }
  }

  const usage = response?.usage;
  return {
    output: output || toolsCalled,
    promptTokens: usage?.promptTokens ?? usage?.prompt_tokens,
    completionTokens: usage?.completionTokens ?? usage?.completion_tokens,
    toolsCalled,
  };
}

export function extractOutputParametersFromResponses(
  response: any,
  inputParameters: InputParameters,
): OutputParameters {
  const output = response?.outputText ?? response?.output_text ?? "";

  let toolsCalled: ToolCall[] | undefined;
  const items = response?.output;
  if (Array.isArray(items)) {
    const descriptions = inputParameters.toolDescriptions ?? {};
    for (const item of items) {
      if (item?.type !== "function_call") continue;
      const name = item?.name;
      if (!name) continue;
      toolsCalled = toolsCalled ?? [];
      toolsCalled.push(
        new ToolCall({
          name,
          type: ToolCallType.FUNCTION,
          inputParameters: parseArguments(item?.arguments),
          description: descriptions[name],
        }),
      );
    }
  }

  const usage = response?.usage;
  return {
    output: output || toolsCalled,
    promptTokens: usage?.inputTokens ?? usage?.input_tokens,
    completionTokens: usage?.outputTokens ?? usage?.output_tokens,
    toolsCalled,
  };
}
