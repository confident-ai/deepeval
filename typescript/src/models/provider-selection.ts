import { getSettings } from "@/config/settings";

/** Matches the `set-<id>` / `unset-<id>` command suffixes in `cli/providers.ts`. */
export type ProviderId =
  | "openai"
  | "gemini"
  | "portkey"
  | "ollama"
  | "local-model"
  | "azure-openai"
  | "moonshot"
  | "grok"
  | "deepseek"
  | "openrouter"
  | "anthropic"
  | "bedrock";

/**
 * The provider the `USE_*` settings select, or null when none is configured.
 *
 * Order mirrors Python's `initialize_model`, so the same environment picks the
 * same provider in both SDKs. LiteLLM is skipped: it has no TS equivalent.
 */
export function selectProvider(): ProviderId | null {
  const settings = getSettings();

  if (settings.USE_OPENAI_MODEL) return "openai";
  if (settings.USE_GEMINI_MODEL) return "gemini";
  if (settings.USE_PORTKEY_MODEL) return "portkey";
  // The placeholder key `set-ollama` writes is what marks Ollama, as in Python.
  if (settings.LOCAL_MODEL_API_KEY === "ollama") return "ollama";
  if (settings.USE_LOCAL_MODEL) return "local-model";
  if (settings.USE_AZURE_OPENAI) return "azure-openai";
  if (settings.USE_MOONSHOT_MODEL) return "moonshot";
  if (settings.USE_GROK_MODEL) return "grok";
  if (settings.USE_DEEPSEEK_MODEL) return "deepseek";
  if (settings.USE_OPENROUTER_MODEL) return "openrouter";
  if (settings.USE_ANTHROPIC_MODEL) return "anthropic";
  if (settings.USE_AWS_BEDROCK_MODEL) return "bedrock";

  return null;
}
