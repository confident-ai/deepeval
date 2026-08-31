import { type OpenAI } from "openai";

import { createChildToolSpans } from "@/openai/utils";
import {
  safeExtractInputParameters,
  safeExtractOutputParameters,
} from "@/openai/extractor";
import { getLlmContext } from "@/tracing/trace-context";
import { normalizeSpanProviderForPlatform } from "@/tracing/utils";
import {
  getCurrentTrace,
  observe,
  updateCurrentSpan,
  updateLlmSpan,
} from "@/tracing";
import { Integration, Provider } from "@/tracing/integrations";
import {
  getCurrentSpan,
  LlmSpan,
  SpanType,
  traceManager,
} from "@/tracing/tracing";
import { InputParameters, OutputParameters } from "@/openai/types";
import {
  detectProviderFromBaseUrl,
  extractOpenRouterMetadata,
  mergeOpenRouterMetadata,
} from "@/openrouter/utils";
import { ToolCall } from "@/test-case";

type AnyFunction = (...args: any[]) => any;

const _ORIGINAL_METHODS: Record<string, AnyFunction> = {};
let _OPENAI_PATCHED = false;
let _DETECTED_PROVIDER: string | undefined;

export function patchOpenAI(client: OpenAI) {
  if (_OPENAI_PATCHED) {
    return;
  }

  _DETECTED_PROVIDER = detectProviderFromBaseUrl(client.baseURL);

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
      // `metrics` needs nothing here: `observe` drains both `next*Span(...)` and
      // the scope-wide `setTracingContext({ llmSpanContext })` values itself.
      metricCollection: llmContext?.metricCollection,
      fn: async (...obsArgs: any[]) => {
        const response = await originalMethod(...obsArgs);

        const outputParameters: OutputParameters = safeExtractOutputParameters(
          isCompletionMethod,
          response,
          inputParameters,
        );

        const activeSpan = getCurrentSpan();
        if (activeSpan) {
          activeSpan.integration = Integration.OPEN_AI;
          if (activeSpan.type === SpanType.LLM) {
            (activeSpan as LlmSpan).provider = resolveProvider(llmContext);
          }
          if (activeSpan.parentUuid) {
            const parentSpan = traceManager.getSpanByUuid(
              activeSpan.parentUuid,
            );
            if (parentSpan && !parentSpan.integration) {
              parentSpan.integration = Integration.OPEN_AI;
            }
          }
        }

        // Runs whether or not an `llmSpanContext` is in scope: input, output
        // and token counts are properties of the call itself, and a bare call
        // with no context is still worth recording.
        updateAllAttributes(
          inputParameters,
          outputParameters,
          llmContext?.expectedTools ?? [],
          llmContext?.expectedOutput ?? "",
          llmContext?.context ?? [],
          llmContext?.retrievalContext ?? [],
        );

        // Which response fields are worth reading is decided by the detected
        // host, not by the label: relabelling a span must not cost the user
        // OpenRouter's cost and routing data.
        if (_DETECTED_PROVIDER === Provider.OPEN_ROUTER) {
          mergeOpenRouterMetadata(extractOpenRouterMetadata(response));
        }

        return response;
      },
    })(...args);
  };
}

function resolveProvider(llmContext: ReturnType<typeof getLlmContext>): string {
  return (
    normalizeSpanProviderForPlatform(
      llmContext?.provider ?? _DETECTED_PROVIDER,
    ) ?? Provider.OPEN_AI
  );
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

  _DETECTED_PROVIDER = undefined;
  _OPENAI_PATCHED = false;
}
