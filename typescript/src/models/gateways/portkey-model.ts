import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "@/models/openai-compatible-model";

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
    // Peeled off so it becomes a header rather than a request body param.
    const { provider: providerOption, ...rest } = options;

    const apiKey = rest.apiKey ?? process.env.PORTKEY_API_KEY;
    const provider = providerOption ?? process.env.PORTKEY_PROVIDER_NAME;

    super({
      ...rest,
      model: rest.model ?? process.env.PORTKEY_MODEL_NAME,
      apiKey,
      baseURL: rest.baseURL ?? process.env.PORTKEY_BASE_URL ?? PORTKEY_BASE_URL,
      defaultHeaders: {
        ...(rest.defaultHeaders ?? {}),
        ...(apiKey ? { "x-portkey-api-key": apiKey } : {}),
        ...(provider ? { "x-portkey-provider": provider } : {}),
      },
    });
  }
}
