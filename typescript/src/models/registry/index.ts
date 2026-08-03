// `models.json` is generated from `deepeval/models/llms/constants.py`, the
// source of truth. Do not hand-edit it; run:
//
//   python scripts/compile_model_registry.py
import generated from "@/models/registry/models.json";

/** Mirrors Python's `DeepEvalModelData`. */
export interface ModelData {
  supportsLogProbs?: boolean;
  maxLogProbs?: number;
  supportsMultimodal?: boolean;
  supportsStructuredOutputs?: boolean;
  supportsJson?: boolean;
  /** USD per input token. */
  inputPrice?: number;
  /** USD per output token. */
  outputPrice?: number;
  supportsTemperature?: boolean;
}

/** One per Python registry. */
export type ModelNamespace =
  | "openai"
  | "anthropic"
  | "gemini"
  | "grok"
  | "kimi"
  | "deepseek"
  | "ollama"
  | "bedrock";

// The compile script omits any field equal to its default, so both sides must
// agree on these.
export const DEFAULT_MODEL_DATA: ModelData = { supportsTemperature: true };

const GENERATED = generated as unknown as Record<
  string,
  Record<string, ModelData>
>;

export const GENERATED_MODEL_DATA: Record<
  string,
  Record<string, ModelData>
> = Object.fromEntries(
  Object.entries(GENERATED).filter(([key]) => key !== "_meta"),
);

/** Unknown models resolve to `DEFAULT_MODEL_DATA`, as they do in Python. */
export function getModelData(
  namespace: ModelNamespace | string | undefined,
  modelName: string | undefined,
): ModelData {
  const entry =
    namespace && modelName
      ? GENERATED_MODEL_DATA[namespace]?.[modelName]
      : undefined;
  return { ...DEFAULT_MODEL_DATA, ...entry };
}
