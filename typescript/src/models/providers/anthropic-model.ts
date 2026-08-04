import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type GenerationKwargs,
  type GenerationResult,
} from "@/models/base-model";
import { extractJson, importOptional, requireApiKey } from "@/models/utils";
import { anthropicContent } from "@/models/multimodal";
import { defaultModelName, type ModelNamespace } from "@/models/registry";

const DEFAULT_MAX_TOKENS = 4096;

export interface AnthropicModelOptions {
  model?: string;
  apiKey?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  maxTokens?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  /** Extra params forwarded to `messages.create(...)`. */
  generationKwargs?: GenerationKwargs;
}

export class AnthropicModel extends DeepEvalBaseLLM {
  private readonly apiKey: string;
  private readonly maxTokens: number;
  private readonly generationKwargs: GenerationKwargs;
  private client?: any;
  protected registryNamespace: ModelNamespace = "anthropic";

  constructor(options: AnthropicModelOptions = {}) {
    super(
      options.model ??
        process.env.ANTHROPIC_MODEL_NAME ??
        defaultModelName("anthropic"),
    );
    this.apiKey = options.apiKey ?? process.env.ANTHROPIC_API_KEY ?? "";
    this.temperature = options.temperature;
    this.maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
    this.generationKwargs = { ...options.generationKwargs };
  }

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { default: Anthropic } = await importOptional(
        "@anthropic-ai/sdk",
        "Anthropic",
      );
      this.client = new Anthropic({
        apiKey: requireApiKey(this.apiKey, "Anthropic", "ANTHROPIC_API_KEY"),
      });
    }
    return this.client;
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    const client = await this.getClient();

    const temperature = this.resolveTemperature();
    const message = await client.messages.create({
      model: this.modelName,
      max_tokens: this.maxTokens,
      ...(temperature !== undefined && { temperature }),
      messages: [{ role: "user", content: anthropicContent(prompt) }],
      ...this.generationKwargs,
    });

    const text: string = (message.content ?? [])
      .filter((block: any) => block.type === "text")
      .map((block: any) => block.text)
      .join("");
    const cost = this.resolveCost(
      message.usage?.input_tokens,
      message.usage?.output_tokens,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? defaultModelName("anthropic");
  }

  supportsMultimodal(): boolean {
    return this.modelData.supportsMultimodal ?? true;
  }
}
