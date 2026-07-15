import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type KimiModelOptions = OpenAICompatibleModelOptions;

const DEFAULT_KIMI_MODEL = "moonshot-v1-8k";
const MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1";

/**
 * Moonshot AI (Kimi) evaluation model, via the OpenAI SDK pointed at Moonshot's
 * OpenAI-compatible endpoint. Use `baseURL: "https://api.moonshot.ai/v1"` for
 * the international endpoint.
 */
export class KimiModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "Moonshot";
  protected apiKeyEnvVar = "MOONSHOT_API_KEY";

  constructor(options: KimiModelOptions = {}) {
    super({
      ...options,
      model: options.model ?? process.env.MOONSHOT_MODEL_NAME ?? DEFAULT_KIMI_MODEL,
      apiKey: options.apiKey ?? process.env.MOONSHOT_API_KEY,
      baseURL: options.baseURL ?? process.env.MOONSHOT_BASE_URL ?? MOONSHOT_BASE_URL,
    });
  }
}
