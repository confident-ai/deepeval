/**
 * The model each provider falls back to when a metric is given no model.
 *
 * Mirrors `DEFAULT_MODELS` in `deepeval/models/llms/constants.py`. That dict is
 * the single source both SDKs generate their defaults from, so one value per
 * provider is correct for Python and TypeScript alike — there is deliberately no
 * per-language split here.
 *
 * This is the only place in the docs that spells a default model out. Pages
 * render it through `<DefaultLLMModel />`, so changing the default is a one-line
 * edit rather than a sweep. Keep it in step with `constants.py`.
 */
export const DEFAULT_MODELS = {
  openai: "gpt-5.4",
  anthropic: "claude-opus-5",
  gemini: "gemini-3.6-flash",
  // Derived from the OpenAI default in `constants.py`, hence the prefix.
  openrouter: "openai/gpt-5.4",
} as const;

export type DefaultModelProvider = keyof typeof DEFAULT_MODELS;
