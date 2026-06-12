import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";

export interface PortkeyModelOptions extends OpenAICompatibleModelOptions {
  /** Upstream provider routed by Portkey (sent as the `x-portkey-provider` header). */
  provider?: string;
}

const PORTKEY_BASE_URL = "https://api.portkey.ai/v1";

/**
 * Portkey gateway, via the OpenAI SDK. Portkey authenticates with its own
 * headers (`x-portkey-api-key` / `x-portkey-provider`) rather than a bearer
 * token, injected here through `defaultHeaders`.
 */
export class PortkeyModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "Portkey";
  protected apiKeyEnvVar = "PORTKEY_API_KEY";

  constructor(options: PortkeyModelOptions = {}) {
    const apiKey = options.apiKey ?? process.env.PORTKEY_API_KEY;
    const provider = options.provider ?? process.env.PORTKEY_PROVIDER_NAME;

    super({
      ...options,
      model: options.model ?? process.env.PORTKEY_MODEL_NAME,
      apiKey,
      baseURL: options.baseURL ?? process.env.PORTKEY_BASE_URL ?? PORTKEY_BASE_URL,
      defaultHeaders: {
        ...(options.defaultHeaders ?? {}),
        ...(apiKey ? { "x-portkey-api-key": apiKey } : {}),
        ...(provider ? { "x-portkey-provider": provider } : {}),
      },
    });
  }
}
