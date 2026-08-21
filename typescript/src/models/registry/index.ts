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
  /** Set only when thinking can be switched on and off per request. */
  supportsThinking?: boolean;
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
  // Underscored keys are metadata (`_meta`, `_defaults`), not namespaces.
  Object.entries(GENERATED).filter(([key]) => !key.startsWith("_")),
);

/**
 * Namespaces whose provider falls back to a default model. Narrower than
 * `ModelNamespace`: it includes `openrouter`, which has no pricing registry, and
 * excludes the providers that require an explicit `*_MODEL_NAME`.
 *
 * Typed off the generated JSON, so dropping a namespace from Python's
 * `DEFAULT_MODELS` breaks the build at the provider that reads it.
 */
export type DefaultModelNamespace = keyof typeof generated._defaults;

/**
 * The model a provider evaluates with when given neither a `model` option nor a
 * `*_MODEL_NAME` env var.
 *
 * Generated from Python's `DEFAULT_MODELS`, so the two SDKs cannot fall back to
 * different judges. Never hardcode a default in a provider — change
 * `deepeval/models/llms/constants.py` and recompile.
 */
export function defaultModelName(namespace: DefaultModelNamespace): string {
  return generated._defaults[namespace];
}

/** Vision-capable model names in a namespace — for "use one of these" errors. */
export function multimodalModelNames(
  namespace: ModelNamespace | string | undefined,
): string[] {
  const models = namespace ? GENERATED_MODEL_DATA[namespace] : undefined;
  if (!models) return [];
  return Object.entries(models)
    .filter(([, data]) => data.supportsMultimodal)
    .map(([name]) => name);
}

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
