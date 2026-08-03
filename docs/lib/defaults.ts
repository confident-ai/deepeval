import type { Language } from "./lang/languages";

/**
 * The judge model each SDK falls back to when a metric is given no model.
 *
 * Per-language because the two SDKs ship different defaults; rendering one
 * language's value to the other is a wrong answer, not a rounding error.
 */
export const DEFAULT_LLM_MODEL: Record<Language, string> = {
  python: "gpt-5.4",
  typescript: "gpt-4.1",
};
