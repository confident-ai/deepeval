import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "../base-model";
import { computeCost } from "../utils";
import { aiSdkContent } from "../multimodal";
import { checkIfMultimodal } from "../../test-case/mllm-image";

export interface AISDKModelOptions {
  /**
   * A Vercel AI SDK `LanguageModel` instance, e.g. `openai("gpt-4o")` from
   * `@ai-sdk/openai` or any other AI SDK provider model.
   */
  model: any;
  temperature?: number;
  maxOutputTokens?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class AISDKModel extends DeepEvalBaseLLM {
  private readonly aiModel: any;
  private readonly temperature?: number;
  private readonly maxOutputTokens?: number;
  private readonly costPerInputToken?: number;
  private readonly costPerOutputToken?: number;

  constructor(options: AISDKModelOptions) {
    super(
      typeof options.model === "string"
        ? options.model
        : options.model?.modelId,
    );
    this.aiModel = options.model;
    this.temperature = options.temperature;
    this.maxOutputTokens = options.maxOutputTokens;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    // `ai` is an optional peer dependency; typed loosely to avoid coupling to
    // its generics. The runtime call shape matches AI SDK v5/v6.
    const ai: any = await import("ai");

    // Plain `prompt` for text; AI SDK `messages` (text + image parts) for multimodal.
    const input = checkIfMultimodal(prompt)
      ? { messages: [{ role: "user", content: aiSdkContent(prompt) }] }
      : { prompt };

    if (schema) {
      const { object, usage } = await ai.generateObject({
        model: this.aiModel,
        schema,
        ...input,
        ...(this.temperature !== undefined && {
          temperature: this.temperature,
        }),
        maxOutputTokens: this.maxOutputTokens,
      });
      const cost = computeCost(
        usage?.inputTokens,
        usage?.outputTokens,
        this.costPerInputToken,
        this.costPerOutputToken,
      );
      return { output: object as T, cost };
    }

    const { text, usage } = await ai.generateText({
      model: this.aiModel,
      ...input,
      ...(this.temperature !== undefined && { temperature: this.temperature }),
      maxOutputTokens: this.maxOutputTokens,
    });
    const cost = computeCost(
      usage?.inputTokens,
      usage?.outputTokens,
      this.costPerInputToken,
      this.costPerOutputToken,
    );
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? "ai-sdk";
  }

  supportsStructuredOutputs(): boolean {
    return true;
  }

  supportsMultimodal(): boolean {
    return true;
  }
}
