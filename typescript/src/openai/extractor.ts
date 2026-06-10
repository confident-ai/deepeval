import type { ChatCompletion } from "openai/resources/chat/completions";
import type { Response } from "openai/resources/responses/responses";

import {
  stringifyMultimodalContent,
  renderResponseInput,
  renderMessages,
  safeJson,
} from "./utils";
import { InputParameters, OutputParameters } from "./types";
import { ToolCall } from "../test-case";

export function safeExtractInputParameters(
  isCompletion: boolean,
  kwargs: Record<string, any>,
): InputParameters {
  try {
    return isCompletion
      ? extractInputParametersFromCompletionAPI(kwargs)
      : extractInputParametersFromResponseAPI(kwargs);
  } catch {
    return { model: "NA" };
  }
}

export function extractInputParametersFromCompletionAPI(
  kwargs: Record<string, any>,
): InputParameters {
  const model = kwargs?.model;
  const messages = kwargs?.messages ?? [];
  const tools = kwargs?.tools;
  const toolDescriptions =
    tools != null
      ? Object.fromEntries(
          (tools as any[]).map((tool) => [
            tool?.function?.name,
            tool?.function?.description,
          ]),
        )
      : undefined;

  let inputArg: any = "";

  const userMsgs = messages
    .filter((message: any) => message?.role === "user")
    .map((message: any) => message?.content);

  if (userMsgs.length > 0) {
    inputArg = userMsgs[0];
  }

  const rendered = renderMessages(messages);

  return {
    model,
    input: stringifyMultimodalContent(inputArg),
    messages: rendered,
    tools,
    toolDescriptions,
  };
}

export function extractInputParametersFromResponseAPI(
  kwargs: Record<string, any>,
): InputParameters {
  const model = kwargs?.model;
  const inputPayload = kwargs?.input;
  const instructions = kwargs?.instructions;
  const tools = kwargs?.tools;
  const toolDescriptions =
    tools != null
      ? Object.fromEntries(
          (tools as any[]).map((tool) => [tool?.name, tool?.description]),
        )
      : undefined;

  let messages: any[] = [];

  if (Array.isArray(inputPayload)) {
    messages = renderResponseInput(inputPayload);
  } else if (typeof inputPayload === "string") {
    messages = [{ role: "user", content: inputPayload }];
  }

  if (instructions) {
    messages.unshift({ role: "system", content: instructions });
  }

  return {
    model,
    input: stringifyMultimodalContent(inputPayload),
    messages,
    instructions,
    tools,
    toolDescriptions,
  };
}

export function safeExtractOutputParameters(
  isCompletion: boolean,
  response: ChatCompletion | Response | any,
  inputParameters: InputParameters,
): OutputParameters {
  try {
    return isCompletion
      ? extractOutputParametersFromCompletionAPI(
          response as ChatCompletion,
          inputParameters,
        )
      : extractOutputParametersFromResponseAPI(
          response as Response,
          inputParameters,
        );
  } catch {
    return {};
  }
}

export function extractOutputParametersFromCompletionAPI(
  completion: ChatCompletion,
  inputParameters: InputParameters,
): OutputParameters {
  const choice = completion.choices?.[0];
  const output = String(choice.message.content ?? "");
  const promptTokens = completion.usage?.prompt_tokens;
  const completionTokens = completion.usage?.completion_tokens;

  let toolsCalled: ToolCall[] | undefined = undefined;

  const openaiToolCalls = choice.message.tool_calls;

  if (openaiToolCalls != null) {
    toolsCalled = [];
    for (const toolCall of openaiToolCalls) {
      const description = inputParameters.toolDescriptions ?? {};

      toolsCalled.push({
        name:
          toolCall.type == "function"
            ? toolCall.function.name
            : toolCall.custom.name,
        inputParameters: safeJson(
          toolCall.type == "function"
            ? toolCall.function.arguments
            : toolCall.custom.input,
        ),
        description:
          description[
            toolCall.type == "function"
              ? toolCall.function.name
              : toolCall.custom.name
          ],
      });
    }
  }

  return {
    output: output || toolsCalled,
    promptTokens,
    completionTokens,
    toolsCalled,
  };
}

export function extractOutputParametersFromResponseAPI(
  response: Response,
  inputParameters: InputParameters,
): OutputParameters {
  const output = response.output_text;
  const promptTokens = response.usage?.input_tokens;
  const completionTokens = response.usage?.output_tokens;

  let toolsCalled: ToolCall[] | undefined = undefined;

  const rawOutput = response.output;

  if (rawOutput != null && Array.isArray(rawOutput)) {
    toolsCalled = [];

    for (const part of rawOutput) {
      if (part.type !== "function_call") {
        continue;
      }

      const description = inputParameters.toolDescriptions ?? {};
      toolsCalled.push({
        name: part.name,
        inputParameters: safeJson(part.arguments),
        description: description[part.name],
      });
    }
  }

  return {
    output: output || toolsCalled || undefined,
    promptTokens,
    completionTokens,
    toolsCalled,
  };
}
