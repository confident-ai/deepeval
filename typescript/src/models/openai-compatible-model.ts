import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "./base-model";
import { computeCost, extractJson, requireApiKey, toJsonSchema } from "./utils";
import { openAIContent } from "./multimodal";

export interface OpenAICompatibleModelOptions {
  model?: string;
  apiKey?: string;
  baseURL?: string;
  temperature?: number;
  defaultHeaders?: Record<string, string>;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

/**
 * Shared base for every provider/gateway that speaks the OpenAI Chat
 * Completions API. Subclasses are thin: they resolve their own defaults
 * (model name, base URL, env-var-backed API key, headers) and hand them to
 * `super(...)`. Everything else — client construction, generation, structured
 * output, token→cost — lives here.
 */
export class DeepEvalOpenAICompatibleModel extends DeepEvalBaseLLM {
  protected apiKey: string;
  protected baseURL?: string;
  protected temperature?: number;
  protected defaultHeaders?: Record<string, string>;
  protected costPerInputToken?: number;
  protected costPerOutputToken?: number;
  protected client?: any;

  protected providerLabel = "OpenAI-compatible";
  protected apiKeyEnvVar = "OPENAI_API_KEY";

  constructor(options: OpenAICompatibleModelOptions = {}) {
    super(options.model);
    this.apiKey = options.apiKey ?? "";
    this.baseURL = options.baseURL;
    // Only sent when explicitly set — some models (e.g. reasoning models) reject `temperature`.
    this.temperature = options.temperature;
    this.defaultHeaders = options.defaultHeaders;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
  }

  /**
   * Builds the underlying client. Override to use a different OpenAI-SDK client
   * (e.g. `AzureOpenAI`). Called lazily on first generation.
   */
  protected async createClient(): Promise<any> {
    const { default: OpenAI } = (await import("openai")) as any;
    return new OpenAI({
      apiKey: requireApiKey(this.apiKey, this.providerLabel, this.apiKeyEnvVar),
      baseURL: this.baseURL,
      defaultHeaders: this.defaultHeaders,
    });
  }

  protected async getClient(): Promise<any> {
    if (!this.client) {
      this.client = await this.createClient();
    }
    return this.client;
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    const client = await this.getClient();

    const request: Record<string, unknown> = {
      model: this.modelName,
      messages: [{ role: "user", content: openAIContent(prompt) }],
      ...(this.temperature !== undefined && { temperature: this.temperature }),
    };
    if (schema) {
      request.response_format = {
        type: "json_schema",
        json_schema: {
          name: "response",
          schema: toJsonSchema(schema),
          strict: false,
        },
      };
    }

    const completion = await client.chat.completions.create(request);
    const content: string = completion.choices?.[0]?.message?.content ?? "";
    const cost = computeCost(
      completion.usage?.prompt_tokens,
      completion.usage?.completion_tokens,
      this.costPerInputToken,
      this.costPerOutputToken,
    );

    if (schema) {
      return { output: schema.parse(extractJson(content)), cost };
    }
    return { output: content as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? this.providerLabel;
  }

  supportsStructuredOutputs(): boolean {
    return true;
  }

  supportsMultimodal(): boolean {
    return true;
  }
}
