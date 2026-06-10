import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type LocalModelOptions = OpenAICompatibleModelOptions;

/**
 * Local OpenAI-compatible server model — covers **vLLM**, **LM Studio**, and any
 * other server exposing an OpenAI `/v1` API. Requires a `baseURL` (e.g.
 * `http://localhost:8000/v1` for vLLM, `http://localhost:1234/v1` for LM Studio).
 *
 * Local servers usually don't require auth; a placeholder API key is used when
 * none is provided, since the OpenAI SDK requires a non-empty key.
 */
export class LocalModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "Local Model";
  protected apiKeyEnvVar = "LOCAL_MODEL_API_KEY";

  constructor(options: LocalModelOptions = {}) {
    super({
      ...options,
      model: options.model ?? process.env.LOCAL_MODEL_NAME,
      apiKey:
        options.apiKey ?? process.env.LOCAL_MODEL_API_KEY ?? "local-no-key",
      baseURL: options.baseURL ?? process.env.LOCAL_MODEL_BASE_URL,
    });

    if (!this.baseURL) {
      throw new Error(
        "LocalModel requires a base URL. Pass `baseURL` (e.g. http://localhost:8000/v1) or set LOCAL_MODEL_BASE_URL.",
      );
    }
    if (!this.modelName) {
      throw new Error(
        "LocalModel requires a model name. Pass `model` or set LOCAL_MODEL_NAME.",
      );
    }
  }
}
