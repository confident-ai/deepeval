import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type ExtraGenerationParams,
  type GenerationResult,
} from "@/models/base-model";
import { extractJson, importOptional, requireApiKey } from "@/models/utils";
import { anthropicContent } from "@/models/multimodal";
import { defaultModelName, type ModelNamespace } from "@/models/registry";

const DEFAULT_MAX_TOKENS = 4096;

/** Any other key is forwarded to `messages.create(...)`. */
export interface AnthropicModelOptions extends ExtraGenerationParams {
  model?: string;
  apiKey?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  maxTokens?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class AnthropicModel extends DeepEvalBaseLLM {
  private readonly apiKey: string;
  private readonly maxTokens: number;
  private readonly extraParams: ExtraGenerationParams;
  private client?: any;
  protected registryNamespace: ModelNamespace = "anthropic";

  constructor(options: AnthropicModelOptions = {}) {
    const {
      model,
      apiKey,
      temperature,
      maxTokens,
      costPerInputToken,
      costPerOutputToken,
      ...extraParams
    } = options;

    super(
      model ??
        process.env.ANTHROPIC_MODEL_NAME ??
        defaultModelName("anthropic"),
    );
    this.apiKey = apiKey ?? process.env.ANTHROPIC_API_KEY ?? "";
    this.temperature = temperature;
    this.maxTokens = maxTokens ?? DEFAULT_MAX_TOKENS;
    this.costPerInputToken = costPerInputToken;
    this.costPerOutputToken = costPerOutputToken;
    this.extraParams = extraParams;
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
      ...this.extraParams,
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
