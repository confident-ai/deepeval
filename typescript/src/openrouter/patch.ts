import { createChildToolSpans } from "@/openai/utils";
import { InputParameters, OutputParameters } from "@/openai/types";
import {
  safeExtractInputParameters,
  safeExtractOutputParameters,
  unwrapRequest,
} from "@/openrouter/extractor";
import {
  OPENROUTER_PROVIDER,
  extractOpenRouterMetadata,
  mergeOpenRouterMetadata,
} from "@/openrouter/metadata";
import { getLlmContext } from "@/tracing/trace-context";
import {
  getCurrentTrace,
  observe,
  updateCurrentSpan,
  updateLlmSpan,
} from "@/tracing";
import { ToolCall } from "@/test-case";

type AnyFunction = (...args: any[]) => any;

const _ORIGINAL_METHODS: Record<string, AnyFunction> = {};
let _OPENROUTER_PATCHED = false;

/** Any object exposing the resources we patch; typed loosely so the SDK stays an optional dependency. */
type OpenRouterLike = {
  chat?: { send?: AnyFunction };
  responses?: { send?: AnyFunction };
};

export function patchOpenRouter(client: OpenRouterLike) {
  if (_OPENROUTER_PATCHED) {
    return;
  }

  const chat = client.chat;
  if (chat?.send) {
    const key = "chat.send";
    if (!_ORIGINAL_METHODS[key]) {
      _ORIGINAL_METHODS[key] = chat.send;
    }
    chat.send = createAsyncOpenRouterWrapper(chat.send);
  }

  const responses = client.responses;
  if (responses?.send) {
    const key = "responses.send";
    if (!_ORIGINAL_METHODS[key]) {
      _ORIGINAL_METHODS[key] = responses.send;
    }
    responses.send = createAsyncOpenRouterWrapper(responses.send);
  }

  _OPENROUTER_PATCHED = true;
}

function isStreaming(args: any[]): boolean {
  return Boolean(unwrapRequest(args)?.stream);
}

function createAsyncOpenRouterWrapper(
  originalMethod: AnyFunction,
): AnyFunction {
  return async function (this: any, ...args: any[]): Promise<any> {
    const boundMethod = originalMethod.bind(this);

    if (isStreaming(args)) {
      return await boundMethod(...args);
    }

    return await patchAsyncOpenRouterClientMethod(boundMethod)(...args);
  };
}

/** OpenRouter unless the user explicitly labelled it otherwise. */
function resolveProvider(): string {
  return getLlmContext()?.provider ?? OPENROUTER_PROVIDER;
}

function patchAsyncOpenRouterClientMethod(
  originalMethod: AnyFunction,
): AnyFunction {
  return async function (...args: any[]): Promise<any> {
    const inputParameters: InputParameters = safeExtractInputParameters(args);
    const llmContext = getLlmContext();

    return await observe({
      type: "llm",
      name: "OpenRouter LLM Call",
      model: inputParameters.model,
      // `metrics` needs nothing here: `observe` drains both `next*Span(...)` and
      // the scope-wide `setTracingContext({ llmSpanContext })` values itself.
      metricCollection: llmContext?.metricCollection,
      fn: async (...obsArgs: any[]) => {
        const response = await originalMethod(...obsArgs);

        const outputParameters: OutputParameters = safeExtractOutputParameters(
          response,
          inputParameters,
        );

        updateAllAttributes(
          inputParameters,
          outputParameters,
          llmContext?.expectedTools ?? [],
          llmContext?.expectedOutput ?? "",
          llmContext?.context ?? [],
          llmContext?.retrievalContext ?? [],
        );

        mergeOpenRouterMetadata(
          extractOpenRouterMetadata(response, resolveProvider()),
        );

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
    expectedOutput,
    expectedTools,
    context,
    retrievalContext,
  });

  updateLlmSpan({
    inputTokenCount: outputParameters.promptTokens,
    outputTokenCount: outputParameters.completionTokens,
    prompt: getLlmContext()?.prompt,
  });

  if (outputParameters.toolsCalled) {
    createChildToolSpans(outputParameters);
  }

  updateInputAndOutputOfCurrentTrace(inputParameters, outputParameters);
}

function updateInputAndOutputOfCurrentTrace(
  inputParameters: InputParameters,
  outputParameters: OutputParameters,
) {
  const currentTrace = getCurrentTrace();
  if (!currentTrace) return;

  if (!currentTrace.input) {
    currentTrace.input = inputParameters.input ?? inputParameters.messages;
  }
  if (!currentTrace.output) {
    currentTrace.output = outputParameters.output;
  }
}

export function unpatchOpenRouter(client: OpenRouterLike) {
  if (!_OPENROUTER_PATCHED) {
    return;
  }

  const chat = client.chat;
  if (chat && _ORIGINAL_METHODS["chat.send"]) {
    chat.send = _ORIGINAL_METHODS["chat.send"];
  }

  const responses = client.responses;
  if (responses && _ORIGINAL_METHODS["responses.send"]) {
    responses.send = _ORIGINAL_METHODS["responses.send"];
  }

  for (const key in _ORIGINAL_METHODS) {
    delete _ORIGINAL_METHODS[key];
  }

  _OPENROUTER_PATCHED = false;
}
