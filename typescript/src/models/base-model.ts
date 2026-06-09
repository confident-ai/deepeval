import type { ZodType } from "zod";

/**
 * The result of an LLM generation: the produced output plus the USD cost.
 */
export interface GenerationResult<T = string> {
  output: T;
  cost: number | null;
}

export abstract class DeepEvalBaseLLM {
  modelName?: string;

  constructor(modelName?: string) {
    this.modelName = this.parseModelName(modelName);
  }

  protected parseModelName(modelName?: string): string | undefined {
    return modelName;
  }

  /**
   * Runs the model to produce an output (and its cost).
   *
   * @param prompt The prompt to send to the model.
   * @param schema Optional zod schema; when provided, the model is asked to
   *   return JSON and the parsed, validated value is returned as `output`.
   */
  abstract generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>>;

  abstract getModelName(): string;

  supportsMultimodal(): boolean | null {
    return null;
  }

  supportsStructuredOutputs(): boolean | null {
    return null;
  }

  supportsLogProbs(): boolean | null {
    return null;
  }
}
