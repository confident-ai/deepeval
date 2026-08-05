import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type ExtraGenerationParams,
  type GenerationResult,
} from "@/models/base-model";
import { aiSdkContent } from "@/models/multimodal";
import { checkIfMultimodal } from "@/test-case/mllm-image";
import type { ModelNamespace } from "@/models/registry";

// Keyed by the segment before the first dot of an AI SDK provider id, which
// looks like `openai.chat` or `google.generative-ai`.
const NAMESPACE_BY_PROVIDER_ID: Record<string, ModelNamespace> = {
  openai: "openai",
  azure: "openai",
  anthropic: "anthropic",
  google: "gemini",
  "google-vertex": "gemini",
  gemini: "gemini",
  xai: "grok",
  grok: "grok",
  deepseek: "deepseek",
  moonshot: "kimi",
  kimi: "kimi",
  "amazon-bedrock": "bedrock",
  bedrock: "bedrock",
  ollama: "ollama",
};

/** `undefined` for providers Python has no data for, e.g. Mistral. */
export function resolveAiSdkNamespace(
  model: unknown,
): ModelNamespace | undefined {
  const provider =
    typeof model === "object" && model !== null
      ? (model as { provider?: unknown }).provider
      : undefined;
  if (typeof provider !== "string") {
    return undefined;
  }
  return NAMESPACE_BY_PROVIDER_ID[provider.split(".")[0]];
}

/** Any other key is forwarded to `generateText(...)` / `generateObject(...)`. */
export interface AISDKModelOptions extends ExtraGenerationParams {
  /** A Vercel AI SDK `LanguageModel`, e.g. `openai("gpt-4o")`. */
  model: any;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  maxOutputTokens?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class AISDKModel extends DeepEvalBaseLLM {
  private readonly aiModel: any;
  private readonly maxOutputTokens?: number;
  private readonly extraParams: ExtraGenerationParams;

  constructor(options: AISDKModelOptions) {
    const {
      model,
      temperature,
      maxOutputTokens,
      costPerInputToken,
      costPerOutputToken,
      ...extraParams
    } = options;

    super(typeof model === "string" ? model : model?.modelId);
    this.aiModel = model;
    this.temperature = temperature;
    this.maxOutputTokens = maxOutputTokens;
    this.costPerInputToken = costPerInputToken;
    this.costPerOutputToken = costPerOutputToken;
    this.extraParams = extraParams;
    this.registryNamespace = resolveAiSdkNamespace(model);
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    // Optional peer dependency, typed loosely to avoid coupling to its
    // generics. Call shape matches AI SDK v5/v6.
    const ai: any = await import("ai");

    const input = checkIfMultimodal(prompt)
      ? { messages: [{ role: "user", content: aiSdkContent(prompt) }] }
      : { prompt };
    const temperature = this.resolveTemperature();

    if (schema) {
      const { object, usage } = await ai.generateObject({
        model: this.aiModel,
        schema,
        ...input,
        ...(temperature !== undefined && { temperature }),
        maxOutputTokens: this.maxOutputTokens,
        ...this.extraParams,
      });
      const cost = this.resolveCost(usage?.inputTokens, usage?.outputTokens);
      return { output: object as T, cost };
    }

    const { text, usage } = await ai.generateText({
      model: this.aiModel,
      ...input,
      ...(temperature !== undefined && { temperature }),
      maxOutputTokens: this.maxOutputTokens,
      ...this.extraParams,
    });
    const cost = this.resolveCost(usage?.inputTokens, usage?.outputTokens);
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? "ai-sdk";
  }

  supportsStructuredOutputs(): boolean {
    return this.modelData.supportsStructuredOutputs ?? true;
  }

  supportsMultimodal(): boolean {
    return this.modelData.supportsMultimodal ?? true;
  }
}
