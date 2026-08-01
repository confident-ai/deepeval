import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export type OrcaRouterModelOptions = OpenAICompatibleModelOptions;

const DEFAULT_ORCAROUTER_MODEL = "openai/gpt-5.4";
const ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1";

/**
 * OrcaRouter gateway, via the OpenAI SDK pointed at OrcaRouter's endpoint.
 * Pass OrcaRouter's optional attribution headers (`HTTP-Referer`, `X-Title`)
 * through `defaultHeaders`.
 */
export class OrcaRouterModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "OrcaRouter";
  protected apiKeyEnvVar = "ORCAROUTER_API_KEY";

  constructor(options: OrcaRouterModelOptions = {}) {
    super({
      ...options,
      model:
        options.model ??
        process.env.ORCAROUTER_MODEL_NAME ??
        DEFAULT_ORCAROUTER_MODEL,
      apiKey: options.apiKey ?? process.env.ORCAROUTER_API_KEY,
      baseURL:
        options.baseURL ??
        process.env.ORCAROUTER_BASE_URL ??
        ORCAROUTER_BASE_URL,
    });
  }
}
