import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type OpenRouterModelOptions = OpenAICompatibleModelOptions;

const DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1";
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

/**
 * OpenRouter gateway, via the OpenAI SDK pointed at OpenRouter's endpoint.
 * Pass OpenRouter's optional ranking headers (`HTTP-Referer`, `X-Title`) through
 * `defaultHeaders`.
 */
export class OpenRouterModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "OpenRouter";
  protected apiKeyEnvVar = "OPENROUTER_API_KEY";

  constructor(options: OpenRouterModelOptions = {}) {
    super({
      ...options,
      model:
        options.model ??
        process.env.OPENROUTER_MODEL_NAME ??
        DEFAULT_OPENROUTER_MODEL,
      apiKey: options.apiKey ?? process.env.OPENROUTER_API_KEY,
      baseURL:
        options.baseURL ?? process.env.OPENROUTER_BASE_URL ?? OPENROUTER_BASE_URL,
    });
  }
}
