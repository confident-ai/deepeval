import { type OpenAI } from "openai";

import { createChildToolSpans } from "./utils";
import {
  safeExtractInputParameters,
  safeExtractOutputParameters,
} from "./extractor";
import { getLlmContext } from "../tracing/trace-context";
import {
  getCurrentTrace,
  observe,
  updateCurrentSpan,
  updateLlmSpan,
} from "../tracing";
import { InputParameters, OutputParameters } from "./types";
import { ToolCall } from "../test-case";

type AnyFunction = (...args: any[]) => any;

const _ORIGINAL_METHODS: Record<string, AnyFunction> = {};
let _OPENAI_PATCHED = false;

export function patchOpenAI(client: OpenAI) {
  if (_OPENAI_PATCHED) {
    return;
  }

  // Patch chat.completions.create
  const completions = client.chat.completions;
  if (completions.create) {
    const key = "chat.completions.create";

    if (!_ORIGINAL_METHODS[key]) {
      _ORIGINAL_METHODS[key] = completions.create;
    }

    completions.create = createAsyncOpenAIWrapper(completions.create, true);
  }

  // Patch chat.completions.parse
  if (completions.parse) {
    const key = "chat.completions.parse";

    if (!_ORIGINAL_METHODS[key]) {
      _ORIGINAL_METHODS[key] = completions.parse;
    }

    completions.parse = createAsyncOpenAIWrapper(completions.parse, true);
  }

  // Patch responses.create
  const responses = client.responses;
  if (responses.create) {
    const key = "responses.create";

    if (!_ORIGINAL_METHODS[key]) {
      _ORIGINAL_METHODS[key] = responses.create;
    }

    responses.create = createAsyncOpenAIWrapper(responses.create, false);
  }

  // Patch responses.parse
  if (responses.parse) {
    const key = "responses.parse";

    if (!_ORIGINAL_METHODS[key]) {
      _ORIGINAL_METHODS[key] = responses.parse;
    }

    responses.parse = createAsyncOpenAIWrapper(responses.parse, false);
  }

  _OPENAI_PATCHED = true;
}

function createAsyncOpenAIWrapper(
  originalMethod: AnyFunction,
  isCompletionMethod: boolean,
): AnyFunction {
  return async function (this: any, ...args: any[]): Promise<any> {
    const boundMethod = originalMethod.bind(this);

    const patched = patchAsyncOpenAIClientMethod(
      boundMethod,
      isCompletionMethod,
    );

    return await patched(...args);
  };
}

function patchAsyncOpenAIClientMethod(
  originalMethod: AnyFunction,
  isCompletionMethod: boolean = false,
): AnyFunction {
  return async function (...args: any[]): Promise<any> {
    let kwargs: Record<string, any> = {};

    if (
      args.length > 0 &&
      typeof args[0] === "object" &&
      !Array.isArray(args[0])
    ) {
      kwargs = args[0];
    }

    const inputParameters: InputParameters = safeExtractInputParameters(
      isCompletionMethod,
      kwargs,
    );

    const llmContext = getLlmContext();

    return await observe({
      type: "llm",
      name: originalMethod.name.replace(/^bound /, "") ?? "OpenAI LLM Call",
      model: inputParameters.model,
      //   metrics: llmContext?.metrics,
      metricCollection: llmContext?.metricCollection,
      fn: async (...obsArgs: any[]) => {
        const response = await originalMethod(...obsArgs);

        const outputParameters: OutputParameters = safeExtractOutputParameters(
          isCompletionMethod,
          response,
          inputParameters,
        );

        if (llmContext && typeof updateAllAttributes === "function") {
          updateAllAttributes(
            inputParameters,
            outputParameters,
            llmContext.expectedTools ?? [],
            llmContext.expectedOutput ?? "",
            llmContext.context ?? [],
            llmContext.retrievalContext ?? [],
          );
        }

        return response;
      },
    })(...args);
  };
}

function updateAllAttributes(
  inputParameters: InputParameters,
  outputParameters: OutputParameters,
  expectedTools: ToolCall[],
  expectedOutput: string,
  context: string[],
  retrievalContext: string[],
) {
  updateCurrentSpan({
    input: inputParameters.messages,
    output: outputParameters.output ?? outputParameters.toolsCalled,
    toolsCalled: outputParameters.toolsCalled,
    expectedOutput: expectedOutput,
    expectedTools: expectedTools,
    context: context,
    retrievalContext: retrievalContext,
  });

  const llmContext = getLlmContext();
  if (llmContext) {
    updateLlmSpan({
      inputTokenCount: outputParameters.promptTokens,
      outputTokenCount: outputParameters.completionTokens,
      prompt: llmContext.prompt,
    });

    if (outputParameters.toolsCalled) {
      createChildToolSpans(outputParameters);
    }

    updateInputAndOutputOfCurrentTrace(inputParameters, outputParameters);
  } else {
    console.log(`[updateAllAttributes]: getLlmContext() returned undefined`);
  }
}

function updateInputAndOutputOfCurrentTrace(
  inputParameters: InputParameters,
  outputParameters: OutputParameters,
) {
  const currentTrace = getCurrentTrace();
  if (currentTrace) {
    if (!currentTrace.input) {
      currentTrace.input = inputParameters.input ?? inputParameters.messages;
    }

    if (!currentTrace.output) {
      currentTrace.output = outputParameters.output;
    }
  } else {
    console.log(
      `[updateInputAndOutputOfCurrentTrace]: getCurrentTrace() returned undefined`,
    );
  }
}

export function unpatchOpenAI(client: OpenAI) {
  if (!_OPENAI_PATCHED) {
    return;
  }

  const completions = client.chat.completions;
  if (completions && _ORIGINAL_METHODS["chat.completions.create"]) {
    completions.create = _ORIGINAL_METHODS["chat.completions.create"];
  }

  if (completions && _ORIGINAL_METHODS["chat.completions.parse"]) {
    completions.parse = _ORIGINAL_METHODS["chat.completions.parse"];
  }

  const responses = client.responses;
  if (responses && _ORIGINAL_METHODS["responses.create"]) {
    responses.create = _ORIGINAL_METHODS["responses.create"];
  }

  if (responses && _ORIGINAL_METHODS["responses.parse"]) {
    responses.parse = _ORIGINAL_METHODS["responses.parse"];
  }

  for (const key in _ORIGINAL_METHODS) {
    delete _ORIGINAL_METHODS[key];
  }

  _OPENAI_PATCHED = false;
}
