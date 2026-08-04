import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "@/models/openai-compatible-model";
import { defaultModelName } from "@/models/registry";

export type OpenRouterModelOptions = OpenAICompatibleModelOptions;

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

/**
 * OpenRouter gateway, via the OpenAI SDK pointed at OpenRouter's endpoint.
 * Pass OpenRouter's optional ranking headers (`HTTP-Referer`, `X-Title`) through
 * `defaultHeaders`.
 */
export class OpenRouterModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "OpenRouter";
  protected apiKeyEnvVar = "OPENROUTER_API_KEY";
  protected costEnvPrefix = "OPENROUTER";

  constructor(options: OpenRouterModelOptions = {}) {
    super({
      ...options,
      model:
        options.model ??
        process.env.OPENROUTER_MODEL_NAME ??
        defaultModelName("openrouter"),
      apiKey: options.apiKey ?? process.env.OPENROUTER_API_KEY,
      baseURL:
        options.baseURL ??
        process.env.OPENROUTER_BASE_URL ??
        OPENROUTER_BASE_URL,
    });
  }
}
