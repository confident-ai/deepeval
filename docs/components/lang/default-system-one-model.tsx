"use client";

import { DEFAULT_SYSTEM_ONE_MODEL } from "@/lib/defaults";

/**
 * Renders the default System One model, e.g. `<DefaultSystemOneModel />`.
 *
 * Counterpart to `DefaultLLMModel` for metrics that decide with Jev instead of
 * a generative LLM.
 */
export const DefaultSystemOneModel = () => (
  <code>{DEFAULT_SYSTEM_ONE_MODEL}</code>
);
