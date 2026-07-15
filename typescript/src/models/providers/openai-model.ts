import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type OpenAIModelOptions = OpenAICompatibleModelOptions;

const DEFAULT_OPENAI_MODEL = "gpt-4.1";

/**
 * OpenAI evaluation model, backed by the official `openai` SDK. The canonical
 * OpenAI-compatible model — all behavior comes from
 * `DeepEvalOpenAICompatibleModel`; this only resolves OpenAI defaults.
 */
export class OpenAIModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "OpenAI";
  protected apiKeyEnvVar = "OPENAI_API_KEY";

  constructor(options: OpenAIModelOptions = {}) {
    super({
      ...options,
      model: options.model ?? process.env.OPENAI_MODEL_NAME ?? DEFAULT_OPENAI_MODEL,
      apiKey: options.apiKey ?? process.env.OPENAI_API_KEY,
    });
  }
}
