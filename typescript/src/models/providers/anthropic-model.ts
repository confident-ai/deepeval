import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type ExtraGenerationParams,
  type GenerationResult,
} from "@/models/base-model";
import { parseBool } from "@/config/utils";
import { extractJson, importOptional, requireApiKey } from "@/models/utils";
import { anthropicContent } from "@/models/multimodal";
import { defaultModelName, type ModelNamespace } from "@/models/registry";

const DEFAULT_MAX_TOKENS = 4096;
// Anthropic's `max_tokens` caps thinking plus response text, and its minimum
// thinking budget is 1024, so a thinking request needs headroom for both.
const DEFAULT_THINKING_MAX_TOKENS = 8192;
const MIN_THINKING_BUDGET_TOKENS = 1024;

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
  private readonly maxTokens?: number;
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
    this.maxTokens = maxTokens;
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

  /**
   * Read at resolve time, not in the constructor, so `editSettings` mid-run is
   * picked up.
   */
  private thinkingEnabled(): boolean {
    return (
      (parseBool(process.env.DEEPEVAL_MODEL_THINKING) ?? false) &&
      this.modelData.supportsThinking === true
    );
  }

  /**
   * The budget and the `thinking` block, sized together because `max_tokens`
   * caps thinking and response text as one. `thinking` is left out where it is
   * not ours to set: models that always think reject a disabled block, and
   * older ones reject the parameter outright.
   */
  protected resolveThinking(): {
    maxTokens: number;
    thinking?: Record<string, unknown>;
  } {
    const enabled = this.thinkingEnabled();
    const maxTokens =
      this.maxTokens ??
      (enabled ? DEFAULT_THINKING_MAX_TOKENS : DEFAULT_MAX_TOKENS);

    if (this.modelData.supportsThinking !== true) return { maxTokens };
    if (!enabled) return { maxTokens, thinking: { type: "disabled" } };

    const budgetTokens = Math.max(
      MIN_THINKING_BUDGET_TOKENS,
      Math.floor(maxTokens / 2),
    );
    if (maxTokens <= budgetTokens) {
      throw new Error(
        `Thinking needs at least ${MIN_THINKING_BUDGET_TOKENS} tokens of ` +
          `budget on top of the response itself, but maxTokens is ` +
          `${maxTokens} and caps thinking and response together. Raise ` +
          `maxTokens above ${MIN_THINKING_BUDGET_TOKENS * 2} or unset ` +
          `DEEPEVAL_MODEL_THINKING.`,
      );
    }
    return {
      maxTokens,
      thinking: { type: "enabled", budget_tokens: budgetTokens },
    };
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    const client = await this.getClient();

    const { maxTokens, thinking } = this.resolveThinking();
    // A thinking request only accepts the default temperature.
    const temperature =
      thinking?.type === "enabled" ? undefined : this.resolveTemperature();
    const message = await client.messages.create({
      model: this.modelName,
      max_tokens: maxTokens,
      ...(temperature !== undefined && { temperature }),
      ...(thinking !== undefined && { thinking }),
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
