// Shared OpenRouter helpers, used by both routes into OpenRouter:
//
//   1. The OpenAI SDK pointed at `https://openrouter.ai/api/v1` (`@/openai`).
//      Responses are raw JSON parsed by the OpenAI SDK, so OpenRouter's extra
//      fields arrive in their original **snake_case**.
//   2. The official `@openrouter/sdk` (`@/openrouter`), which is Speakeasy-
//      generated and normalizes every field to **camelCase**.
//
// Port of `deepeval/model_integrations/openrouter.py`, with one deliberate
// difference: the TypeScript span classes carry no `provider`/`integration`
// fields, so everything here lands in the span's `metadata` instead.

import { getCurrentSpan } from "@/tracing";

// Namespace OpenRouter's detail under a single key so it can never collide
// with metadata the user set via `updateCurrentSpan({ metadata })`.
export const OPENROUTER_METADATA_KEY = "openrouter";

export const OPENROUTER_PROVIDER = "OpenRouter";

// Base-URL host -> provider label, for OpenAI-compatible gateways we can
// recognize on sight. Add an entry to teach `@/openai` about another gateway;
// anything unrecognized is left alone.
const HOST_PROVIDERS: Record<string, string> = {
  "openrouter.ai": OPENROUTER_PROVIDER,
};

/**
 * Resolve a provider from a client's base URL, or undefined if unrecognized.
 * Accepts a URL object, a string, or nothing — a base URL is never worth
 * throwing over.
 */
export function detectProviderFromBaseUrl(baseUrl: unknown): string | undefined {
  if (!baseUrl) return undefined;

  let host: string;
  try {
    host = new URL(String(baseUrl)).hostname.toLowerCase();
  } catch {
    return undefined;
  }

  for (const [knownHost, provider] of Object.entries(HOST_PROVIDERS)) {
    // Suffix match so regional/vanity subdomains resolve too, but a lookalike
    // domain that merely contains the name does not.
    if (host === knownHost || host.endsWith(`.${knownHost}`)) {
      return provider;
    }
  }
  return undefined;
}

/**
 * Read a field that may arrive under either casing.
 *
 * The same logical field is `prompt_tokens_details` over the OpenAI-compatible
 * endpoint and `promptTokensDetails` through the official SDK, so every read
 * has to try both.
 */
function get(obj: any, snake: string, camel: string): any {
  if (obj == null || typeof obj !== "object") return undefined;
  const value = obj[snake] ?? obj[camel];
  return value ?? undefined;
}

/** Plain-data rendering, so span metadata stays JSON-safe. */
function toPlain(value: any): any {
  if (value == null) return undefined;
  if (typeof value !== "object") return value;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return String(value);
  }
}

export interface OpenRouterMetadata {
  provider: string;
  generationId?: string;
  upstreamProvider?: string;
  cost?: number;
  costDetails?: Record<string, any>;
  isByok?: boolean;
  cachedTokens?: number;
  cacheWriteTokens?: number;
  reasoningTokens?: number;
  routing?: Record<string, any>;
}

/**
 * Pull OpenRouter's non-standard response fields into a metadata object.
 *
 * Captures what OpenRouter knows that a plain provider response does not:
 * which upstream actually served the request, what it cost, and how the router
 * got there. Always includes `provider`, since the TypeScript spans have no
 * dedicated field to carry it.
 */
export function extractOpenRouterMetadata(
  response: any,
  providerLabel: string = OPENROUTER_PROVIDER,
): OpenRouterMetadata {
  const metadata: OpenRouterMetadata = { provider: providerLabel };

  try {
    // Generation id ("gen-..."), queryable against /api/v1/generation.
    const generationId = response?.id;
    if (generationId) metadata.generationId = generationId;

    // The upstream OpenRouter routed to (e.g. "Anthropic"). Present on the
    // OpenAI-compatible endpoint only.
    const upstream = response?.provider;
    if (typeof upstream === "string" && upstream) {
      metadata.upstreamProvider = upstream;
    }

    const usage = response?.usage;
    if (usage) {
      const cost = usage.cost;
      if (typeof cost === "number") metadata.cost = cost;

      const costDetails = get(usage, "cost_details", "costDetails");
      if (costDetails) metadata.costDetails = toPlain(costDetails);

      const isByok = get(usage, "is_byok", "isByok");
      if (typeof isByok === "boolean") metadata.isByok = isByok;

      // Chat Completions calls these prompt/completion; the Responses API
      // calls the same things input/output.
      const promptDetails =
        get(usage, "prompt_tokens_details", "promptTokensDetails") ??
        get(usage, "input_tokens_details", "inputTokensDetails");
      if (promptDetails) {
        const cached = get(promptDetails, "cached_tokens", "cachedTokens");
        if (cached) metadata.cachedTokens = cached;
        const cacheWrite = get(
          promptDetails,
          "cache_write_tokens",
          "cacheWriteTokens",
        );
        if (cacheWrite) metadata.cacheWriteTokens = cacheWrite;
      }

      const completionDetails =
        get(usage, "completion_tokens_details", "completionTokensDetails") ??
        get(usage, "output_tokens_details", "outputTokensDetails");
      if (completionDetails) {
        const reasoning = get(
          completionDetails,
          "reasoning_tokens",
          "reasoningTokens",
        );
        if (reasoning) metadata.reasoningTokens = reasoning;
      }
    }

    // Routing detail, present only on the official SDK's response.
    const router = get(
      response,
      "openrouter_metadata",
      "openrouterMetadata",
    );
    if (router) {
      const routing: Record<string, any> = {};
      for (const field of ["strategy", "summary", "attempt", "region"]) {
        const value = router[field];
        if (value != null) routing[field] = toPlain(value);
      }
      if (Object.keys(routing).length > 0) metadata.routing = routing;

      // `isByok` also lives here; only fall back to it if usage lacked one.
      if (metadata.isByok === undefined) {
        const isByok = get(router, "is_byok", "isByok");
        if (typeof isByok === "boolean") metadata.isByok = isByok;
      }
    }
  } catch {
    // Metadata is strictly additive — never let it break a traced call.
  }

  return metadata;
}

/**
 * Merge OpenRouter metadata onto the current span.
 *
 * Written straight onto the span rather than through `updateCurrentSpan`,
 * because that replaces the metadata object wholesale — a user who set their
 * own metadata on this span would otherwise lose it, or we would lose ours.
 */
export function mergeOpenRouterMetadata(metadata: OpenRouterMetadata): void {
  const currentSpan = getCurrentSpan();
  if (!currentSpan) return;

  currentSpan.metadata = {
    ...(currentSpan.metadata ?? {}),
    [OPENROUTER_METADATA_KEY]: metadata,
  };
}
