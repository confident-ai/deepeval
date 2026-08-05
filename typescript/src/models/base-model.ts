import type { ZodType } from "zod";
import { parseNumber } from "@/config/utils";
import {
  getModelData,
  multimodalModelNames,
  type ModelData,
  type ModelNamespace,
} from "@/models/registry";
import { computeCost } from "@/models/utils";
import { observeMethods } from "@/tracing/internal";
import { SpanType } from "@/tracing/tracing";

export interface GenerationResult<T = string> {
  output: T;
  cost: number | null;
}

export interface TokenLogProb {
  token: string;
  logprob: number;
}

/** One generated token, plus the alternatives the model considered for it. */
export interface ContentTokenLogProbs extends TokenLogProb {
  topLogProbs: TokenLogProb[];
}

/**
 * A generation that also exposes per-token log probabilities. Normalized away
 * from any one provider's wire format so consumers (GEval) stay provider-
 * agnostic, unlike Python which passes OpenAI's `ChatCompletion` straight
 * through.
 */
export interface RawGenerationResult {
  output: string;
  cost: number | null;
  logProbs?: ContentTokenLogProbs[];
}

export interface RawGenerationOptions {
  topLogprobs?: number;
}

/**
 * Provider params with no first-class option. Every model's options type
 * extends this, so any key the underlying SDK accepts is passed inline next to
 * the options and collected by rest destructuring — where Python needs a
 * separate `generation_kwargs` dict. Merged last into the request, so a key
 * given here overrides the equivalent first-class option.
 */
export interface ExtraGenerationParams {
  [key: string]: unknown;
}

export const DEFAULT_TEMPERATURE = 0;

/**
 * The `<PREFIX>_COST_PER_{INPUT,OUTPUT}_TOKEN` prefix each provider reads.
 * Azure shares OpenAI's prefix; Ollama has none because local models are free.
 */
const COST_ENV_PREFIX_BY_NAMESPACE: Partial<Record<ModelNamespace, string>> = {
  openai: "OPENAI",
  anthropic: "ANTHROPIC",
  gemini: "GEMINI",
  grok: "GROK",
  kimi: "MOONSHOT",
  deepseek: "DEEPSEEK",
  bedrock: "AWS_BEDROCK",
};

export abstract class DeepEvalBaseLLM {
  modelName?: string;

  /** Unset by providers Python has no data for; those resolve to defaults. */
  protected registryNamespace?: ModelNamespace;

  /** Set when the registry key differs from `modelName`, as on Azure. */
  protected registryModelName?: string;

  /** Take precedence over the registry's prices. */
  protected costPerInputToken?: number;
  protected costPerOutputToken?: number;

  /** Set by gateways that price separately from any registry namespace. */
  protected costEnvPrefix?: string;

  /** Unset means `DEFAULT_TEMPERATURE`; `null` means "never send temperature". */
  protected temperature?: number | null;

  private cachedModelData?: ModelData;

  constructor(modelName?: string) {
    observeMethods(this, {
      spanType: SpanType.LLM,
      methods: ["generate", "generateRaw", "generateSamples", "batchGenerate"],
    });
    this.modelName = this.parseModelName(modelName);
  }

  protected parseModelName(modelName?: string): string | undefined {
    return modelName;
  }

  // Lazy: subclasses set `registryNamespace` in field initializers, which run
  // after this constructor.
  protected get modelData(): ModelData {
    this.cachedModelData ??= getModelData(
      this.registryNamespace,
      this.registryModelName ?? this.modelName,
    );
    return this.cachedModelData;
  }

  /** When `schema` is given, the model returns JSON parsed into `output`. */
  abstract generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>>;

  /**
   * Generate while asking for per-token log probabilities. Optional on purpose:
   * its absence is how a caller learns this provider can't do log-probs, the
   * same signal Python gets from `generate_raw_response` raising
   * `AttributeError`. Callers must always have a non-log-prob fallback.
   */
  generateRaw?(
    prompt: string,
    options?: RawGenerationOptions,
  ): Promise<RawGenerationResult>;

  abstract getModelName(): string;

  /**
   * Read at resolve time, not in the constructor, so `editSettings` mid-run is
   * picked up.
   */
  private envCost(direction: "INPUT" | "OUTPUT"): number | undefined {
    const prefix =
      this.costEnvPrefix ??
      (this.registryNamespace
        ? COST_ENV_PREFIX_BY_NAMESPACE[this.registryNamespace]
        : undefined);
    if (!prefix) return undefined;
    const value = parseNumber(
      process.env[`${prefix}_COST_PER_${direction}_TOKEN`],
    );
    return value !== undefined && value >= 0 ? value : undefined;
  }

  /** `undefined` means omit the field, as reasoning models reject it. */
  protected resolveTemperature(): number | undefined {
    if (this.temperature === null) {
      return undefined;
    }
    if (this.modelData.supportsTemperature === false) {
      return undefined;
    }
    return (
      this.temperature ??
      parseNumber(process.env.TEMPERATURE) ??
      DEFAULT_TEMPERATURE
    );
  }

  /** `null` when neither the caller, the env, nor the registry knows the rates. */
  protected resolveCost(
    inputTokens: number | null | undefined,
    outputTokens: number | null | undefined,
  ): number | null {
    return computeCost(
      inputTokens,
      outputTokens,
      this.costPerInputToken ??
        this.envCost("INPUT") ??
        this.modelData.inputPrice,
      this.costPerOutputToken ??
        this.envCost("OUTPUT") ??
        this.modelData.outputPrice,
    );
  }

  supportsMultimodal(): boolean | null {
    return this.modelData.supportsMultimodal ?? null;
  }

  /**
   * Vision-capable models from this model's own provider, so a "not a vision
   * model" error can suggest a drop-in replacement. Empty for providers with no
   * registry.
   */
  multimodalAlternatives(): string[] {
    return multimodalModelNames(this.registryNamespace);
  }

  supportsStructuredOutputs(): boolean | null {
    return this.modelData.supportsStructuredOutputs ?? null;
  }

  supportsLogProbs(): boolean | null {
    return this.modelData.supportsLogProbs ?? null;
  }

  maxLogProbs(): number | null {
    return this.modelData.maxLogProbs ?? null;
  }

  /** Clamp a requested `top_logprobs` to what the model actually allows. */
  protected capTopLogprobs(topLogprobs: number): number {
    const max = this.maxLogProbs();
    return max == null ? topLogprobs : Math.min(topLogprobs, max);
  }

  supportsTemperature(): boolean | null {
    return this.modelData.supportsTemperature ?? null;
  }
}
