import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type GrokModelOptions = OpenAICompatibleModelOptions;

const DEFAULT_GROK_MODEL = "grok-3";
const GROK_BASE_URL = "https://api.x.ai/v1";

/**
 * xAI Grok evaluation model. xAI exposes an OpenAI-compatible API, so this uses
 * the OpenAI SDK pointed at `api.x.ai/v1` (no xAI-specific SDK needed).
 */
export class GrokModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "Grok";
  protected apiKeyEnvVar = "GROK_API_KEY";

  constructor(options: GrokModelOptions = {}) {
    super({
      ...options,
      model: options.model ?? process.env.GROK_MODEL_NAME ?? DEFAULT_GROK_MODEL,
      apiKey:
        options.apiKey ?? process.env.GROK_API_KEY ?? process.env.XAI_API_KEY,
      baseURL: options.baseURL ?? GROK_BASE_URL,
    });
  }
}
