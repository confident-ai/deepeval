import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type DeepSeekModelOptions = OpenAICompatibleModelOptions;

const DEFAULT_DEEPSEEK_MODEL = "deepseek-chat";
const DEEPSEEK_BASE_URL = "https://api.deepseek.com";

/**
 * DeepSeek evaluation model, via the OpenAI SDK pointed at DeepSeek's
 * OpenAI-compatible endpoint.
 */
export class DeepSeekModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "DeepSeek";
  protected apiKeyEnvVar = "DEEPSEEK_API_KEY";

  constructor(options: DeepSeekModelOptions = {}) {
    super({
      ...options,
      model:
        options.model ?? process.env.DEEPSEEK_MODEL_NAME ?? DEFAULT_DEEPSEEK_MODEL,
      apiKey: options.apiKey ?? process.env.DEEPSEEK_API_KEY,
      baseURL: options.baseURL ?? DEEPSEEK_BASE_URL,
    });
  }
}
