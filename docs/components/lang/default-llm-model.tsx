"use client";

import { DEFAULT_MODELS, type DefaultModelProvider } from "@/lib/defaults";

/**
 * Renders a provider's default judge model, e.g. `<DefaultLLMModel />` or
 * `<DefaultLLMModel provider="anthropic" />`.
 *
 * Not language-aware: both SDKs resolve their defaults from the same generated
 * table, so answering differently per language would invent a difference that no
 * longer exists.
 */
export const DefaultLLMModel = ({
  provider = "openai",
}: {
  provider?: DefaultModelProvider;
}) => <code>{DEFAULT_MODELS[provider]}</code>;
