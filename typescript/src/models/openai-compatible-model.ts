import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type ContentTokenLogProbs,
  type ExtraGenerationParams,
  type GenerationResult,
  type RawGenerationOptions,
  type RawGenerationResult,
} from "@/models/base-model";
import { extractJson, requireApiKey, toJsonSchema } from "@/models/utils";
import { openAIContent } from "@/models/multimodal";

/** Any other key is forwarded to `chat.completions.create(...)`. */
export interface OpenAICompatibleModelOptions extends ExtraGenerationParams {
  model?: string;
  apiKey?: string;
  baseURL?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
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
  protected defaultHeaders?: Record<string, string>;
  protected extraParams: ExtraGenerationParams;
  protected client?: any;

  protected providerLabel = "OpenAI-compatible";
  protected apiKeyEnvVar = "OPENAI_API_KEY";

  constructor(options: OpenAICompatibleModelOptions = {}) {
    const {
      model,
      apiKey,
      baseURL,
      temperature,
      defaultHeaders,
      costPerInputToken,
      costPerOutputToken,
      ...extraParams
    } = options;

    super(model);
    this.apiKey = apiKey ?? "";
    this.baseURL = baseURL;
    this.temperature = temperature;
    this.defaultHeaders = defaultHeaders;
    this.costPerInputToken = costPerInputToken;
    this.costPerOutputToken = costPerOutputToken;
    this.extraParams = extraParams;
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

    const temperature = this.resolveTemperature();
    const request: Record<string, unknown> = {
      model: this.modelName,
      messages: [{ role: "user", content: openAIContent(prompt) }],
      ...(temperature !== undefined && { temperature }),
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
    Object.assign(request, this.extraParams);

    const completion = await client.chat.completions.create(request);
    const content: string = completion.choices?.[0]?.message?.content ?? "";
    const cost = this.resolveCost(
      completion.usage?.prompt_tokens,
      completion.usage?.completion_tokens,
    );

    if (schema) {
      return { output: schema.parse(extractJson(content)), cost };
    }
    return { output: content as T, cost };
  }

  /**
   * Deliberately sends no `response_format`: structured outputs and
   * `top_logprobs` don't compose well, and the caller needs the score token to
   * appear as a plain token. The prompt asks for JSON and the caller recovers
   * it with `extractJson`, matching Python's `generate_raw_response`.
   */
  async generateRaw(
    prompt: string,
    options: RawGenerationOptions = {},
  ): Promise<RawGenerationResult> {
    if (this.supportsLogProbs() === false) {
      throw new Error(
        `Model '${this.getModelName()}' does not support 'logprobs' / 'top_logprobs'.`,
      );
    }
    const client = await this.getClient();

    const temperature = this.resolveTemperature();
    const request: Record<string, unknown> = {
      model: this.modelName,
      messages: [{ role: "user", content: openAIContent(prompt) }],
      ...(temperature !== undefined && { temperature }),
      logprobs: true,
      top_logprobs: this.capTopLogprobs(options.topLogprobs ?? 20),
    };
    Object.assign(request, this.extraParams);

    const completion = await client.chat.completions.create(request);
    const choice = completion.choices?.[0];
    const cost = this.resolveCost(
      completion.usage?.prompt_tokens,
      completion.usage?.completion_tokens,
    );

    const logProbs: ContentTokenLogProbs[] | undefined = (
      choice?.logprobs?.content as any[] | undefined
    )?.map((entry) => ({
      token: entry.token,
      logprob: entry.logprob,
      topLogProbs: (entry.top_logprobs ?? []).map((alt: any) => ({
        token: alt.token,
        logprob: alt.logprob,
      })),
    }));

    return { output: choice?.message?.content ?? "", cost, logProbs };
  }

  getModelName(): string {
    return this.modelName ?? this.providerLabel;
  }

  // Fall back to `true` for models the registry omits, since the transport
  // itself supports both.
  supportsStructuredOutputs(): boolean {
    return this.modelData.supportsStructuredOutputs ?? true;
  }

  supportsMultimodal(): boolean {
    return this.modelData.supportsMultimodal ?? true;
  }
}
