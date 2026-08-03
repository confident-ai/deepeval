import type { ZodType } from "zod";
import {
  getModelData,
  type ModelData,
  type ModelNamespace,
} from "@/models/registry";
import { computeCost } from "@/models/utils";

export interface GenerationResult<T = string> {
  output: T;
  cost: number | null;
}

/**
 * Extra provider-specific generation parameters, the counterpart of Python's
 * `generation_kwargs`. Merged last, so a key set here overrides the equivalent
 * first-class option.
 */
export type GenerationKwargs = Record<string, unknown>;

export const DEFAULT_TEMPERATURE = 0;

export abstract class DeepEvalBaseLLM {
  modelName?: string;

  /** Unset by providers Python has no data for; those resolve to defaults. */
  protected registryNamespace?: ModelNamespace;

  /** Set when the registry key differs from `modelName`, as on Azure. */
  protected registryModelName?: string;

  /** Take precedence over the registry's prices. */
  protected costPerInputToken?: number;
  protected costPerOutputToken?: number;

  /** Unset means `DEFAULT_TEMPERATURE`; `null` means "never send temperature". */
  protected temperature?: number | null;

  private cachedModelData?: ModelData;

  constructor(modelName?: string) {
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

  abstract getModelName(): string;

  /** `undefined` means omit the field, as reasoning models reject it. */
  protected resolveTemperature(): number | undefined {
    if (this.temperature === null) {
      return undefined;
    }
    if (this.modelData.supportsTemperature === false) {
      return undefined;
    }
    return this.temperature ?? DEFAULT_TEMPERATURE;
  }

  /** `null` when neither the caller nor the registry knows the rates. */
  protected resolveCost(
    inputTokens: number | null | undefined,
    outputTokens: number | null | undefined,
  ): number | null {
    return computeCost(
      inputTokens,
      outputTokens,
      this.costPerInputToken ?? this.modelData.inputPrice,
      this.costPerOutputToken ?? this.modelData.outputPrice,
    );
  }

  supportsMultimodal(): boolean | null {
    return this.modelData.supportsMultimodal ?? null;
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

  supportsTemperature(): boolean | null {
    return this.modelData.supportsTemperature ?? null;
  }
}
