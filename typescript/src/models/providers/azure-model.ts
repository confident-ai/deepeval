import {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "../openai-compatible-model";
import { requireApiKey } from "../utils";

export interface AzureOpenAIModelOptions extends OpenAICompatibleModelOptions {
  endpoint?: string;
  apiVersion?: string;
  deployment?: string;
}

export class AzureOpenAIModel extends DeepEvalOpenAICompatibleModel {
  protected providerLabel = "Azure OpenAI";
  protected apiKeyEnvVar = "AZURE_OPENAI_API_KEY";
  private endpoint?: string;
  private apiVersion?: string;
  private deployment?: string;

  constructor(options: AzureOpenAIModelOptions = {}) {
    const deployment =
      options.deployment ??
      options.model ??
      process.env.AZURE_DEPLOYMENT_NAME ??
      process.env.AZURE_MODEL_NAME;

    super({
      ...options,
      // Azure routes by deployment; the request `model` is the deployment name.
      model: deployment,
      apiKey: options.apiKey ?? process.env.AZURE_OPENAI_API_KEY,
    });

    this.endpoint = options.endpoint ?? process.env.AZURE_OPENAI_ENDPOINT;
    this.apiVersion = options.apiVersion ?? process.env.OPENAI_API_VERSION;
    this.deployment = deployment;

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
    return new AzureOpenAI({
      apiKey: requireApiKey(this.apiKey, this.providerLabel, this.apiKeyEnvVar),
      endpoint: this.endpoint,
      apiVersion: this.apiVersion,
      deployment: this.deployment,
    });
  }
}
