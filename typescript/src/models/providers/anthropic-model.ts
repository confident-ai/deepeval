import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "../base-model";
import {
  computeCost,
  extractJson,
  importOptional,
  requireApiKey,
} from "../utils";

const DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6";
const DEFAULT_MAX_TOKENS = 4096;

export interface AnthropicModelOptions {
  model?: string;
  apiKey?: string;
  temperature?: number;
  maxTokens?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class AnthropicModel extends DeepEvalBaseLLM {
  private readonly apiKey: string;
  private readonly temperature?: number;
  private readonly maxTokens: number;
  private readonly costPerInputToken?: number;
  private readonly costPerOutputToken?: number;
  private client?: any;

  constructor(options: AnthropicModelOptions = {}) {
    super(
      options.model ??
        process.env.ANTHROPIC_MODEL_NAME ??
        DEFAULT_ANTHROPIC_MODEL,
    );
    this.apiKey = options.apiKey ?? process.env.ANTHROPIC_API_KEY ?? "";
    // Left undefined unless explicitly set — some models (e.g. reasoning models)
    // reject `temperature`, so we only send it when the caller provides it.
    this.temperature = options.temperature;
    this.maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
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

    const message = await client.messages.create({
      model: this.modelName,
      max_tokens: this.maxTokens,
      ...(this.temperature !== undefined && { temperature: this.temperature }),
      messages: [{ role: "user", content: prompt }],
    });

    const text: string = (message.content ?? [])
      .filter((block: any) => block.type === "text")
      .map((block: any) => block.text)
      .join("");
    const cost = computeCost(
      message.usage?.input_tokens,
      message.usage?.output_tokens,
      this.costPerInputToken,
      this.costPerOutputToken,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? DEFAULT_ANTHROPIC_MODEL;
  }

  supportsMultimodal(): boolean {
    return true;
  }
}
