import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "@/models/openai-compatible-model";
import { defaultModelName, type ModelNamespace } from "@/models/registry";

export type OpenAIModelOptions = OpenAICompatibleModelOptions;

/**
 * OpenAI evaluation model, backed by the official `openai` SDK. The canonical
 * OpenAI-compatible model — all behavior comes from
 * `DeepEvalOpenAICompatibleModel`; this only resolves OpenAI defaults.
 */
export class OpenAIModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "OpenAI";
  protected apiKeyEnvVar = "OPENAI_API_KEY";
  protected registryNamespace: ModelNamespace = "openai";

  constructor(options: OpenAIModelOptions = {}) {
    super({
      ...options,
      model:
        options.model ??
        process.env.OPENAI_MODEL_NAME ??
        defaultModelName("openai"),
      apiKey: options.apiKey ?? process.env.OPENAI_API_KEY,
    });
  }
}
