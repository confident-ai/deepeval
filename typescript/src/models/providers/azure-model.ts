import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "@/models/openai-compatible-model";
import { requireApiKey } from "@/models/utils";
import type { ModelNamespace } from "@/models/registry";

export interface AzureOpenAIModelOptions extends OpenAICompatibleModelOptions {
  endpoint?: string;
  apiVersion?: string;
  deployment?: string;
  /** Entra ID token, used instead of an API key. */
  adToken?: string;
}

export class AzureOpenAIModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "Azure OpenAI";
  protected apiKeyEnvVar = "AZURE_OPENAI_API_KEY";
  protected registryNamespace: ModelNamespace = "openai";
  private endpoint?: string;
  private apiVersion?: string;
  private deployment?: string;
  private adToken?: string;

  constructor(options: AzureOpenAIModelOptions = {}) {
    // Peeled off so they reach the client rather than the request body.
    const { endpoint, apiVersion, deployment, adToken, ...rest } = options;

    const resolvedDeployment =
      deployment ??
      rest.model ??
      process.env.AZURE_DEPLOYMENT_NAME ??
      process.env.AZURE_MODEL_NAME;

    super({
      ...rest,
      model: resolvedDeployment,
      apiKey: rest.apiKey ?? process.env.AZURE_OPENAI_API_KEY,
    });

    this.endpoint = endpoint ?? process.env.AZURE_OPENAI_ENDPOINT;
    this.apiVersion = apiVersion ?? process.env.OPENAI_API_VERSION;
    this.adToken = adToken ?? process.env.AZURE_OPENAI_AD_TOKEN;
    this.deployment = resolvedDeployment;
    // Requests route by deployment, but pricing belongs to the underlying model.
    this.registryModelName = options.model;

    if (!this.endpoint) {
      throw new Error(
        "Azure OpenAI requires an endpoint. Pass `endpoint` or set AZURE_OPENAI_ENDPOINT.",
      );
    }
    if (!this.deployment) {
      throw new Error(
        "Azure OpenAI requires a deployment. Pass `deployment` (or `model`) or set AZURE_DEPLOYMENT_NAME.",
      );
    }
  }

  protected async createClient(): Promise<any> {
    const { AzureOpenAI } = (await import("openai")) as any;
    const credential = this.adToken
      ? { azureADTokenProvider: async () => this.adToken as string }
      : {
          apiKey: requireApiKey(
            this.apiKey,
            this.providerLabel,
            this.apiKeyEnvVar,
          ),
        };
    return new AzureOpenAI({
      ...credential,
      endpoint: this.endpoint,
      apiVersion: this.apiVersion,
      deployment: this.deployment,
    });
  }
}
