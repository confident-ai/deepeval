// Unit tests for the shared OpenRouter helpers. No SDK, no network.

import {
  detectProviderFromBaseUrl,
  extractOpenRouterMetadata,
} from "@/openrouter/utils";

describe("detectProviderFromBaseUrl", () => {
  test.each([
    "https://openrouter.ai/api/v1",
    "https://OpenRouter.ai/api/v1",
    // Suffix match, so regional/vanity subdomains still resolve.
    "https://eu.openrouter.ai/api/v1",
  ])("detects OpenRouter for %s", (baseUrl) => {
    expect(detectProviderFromBaseUrl(baseUrl)).toBe("OpenRouter");
  });

  test.each([
    undefined,
    null,
    "",
    "https://api.openai.com/v1",
    "https://my-proxy.internal/v1",
    // Must not match a lookalike domain that merely contains the name.
    "https://openrouter.ai.evil.com/v1",
    "not a url",
  ])("does not detect for %s", (baseUrl) => {
    expect(detectProviderFromBaseUrl(baseUrl)).toBeUndefined();
  });

  test("accepts a URL object as well as a string", () => {
    expect(
      detectProviderFromBaseUrl(new URL("https://openrouter.ai/api/v1")),
    ).toBe("OpenRouter");
  });
});

describe("extractOpenRouterMetadata", () => {
  test("reads the snake_case shape the OpenAI-compatible endpoint returns", () => {
    const metadata = extractOpenRouterMetadata({
      id: "gen-123",
      provider: "Anthropic",
      usage: {
        cost: 0.5,
        is_byok: true,
        cost_details: { upstream_inference_cost: 0.1 },
        prompt_tokens_details: { cached_tokens: 4, cache_write_tokens: 9 },
        completion_tokens_details: { reasoning_tokens: 2 },
      },
    });

    expect(metadata).toEqual({
      generationId: "gen-123",
      upstreamProvider: "Anthropic",
      cost: 0.5,
      costDetails: { upstream_inference_cost: 0.1 },
      isByok: true,
      cachedTokens: 4,
      cacheWriteTokens: 9,
      reasoningTokens: 2,
    });
  });

  test("reads the camelCase shape the official SDK returns", () => {
    const metadata = extractOpenRouterMetadata({
      id: "gen-456",
      usage: {
        cost: 0.25,
        isByok: false,
        promptTokensDetails: { cachedTokens: 7 },
        completionTokensDetails: { reasoningTokens: 3 },
      },
      openrouterMetadata: {
        strategy: "price",
        summary: "routed to Anthropic",
        attempt: 1,
        region: "us-east",
      },
    });

    expect(metadata.generationId).toBe("gen-456");
    expect(metadata.cost).toBe(0.25);
    expect(metadata.isByok).toBe(false);
    expect(metadata.cachedTokens).toBe(7);
    expect(metadata.reasoningTokens).toBe(3);
    expect(metadata.routing).toEqual({
      strategy: "price",
      summary: "routed to Anthropic",
      attempt: 1,
      region: "us-east",
    });
  });

  test("reads the Responses API's input/output token detail names", () => {
    const metadata = extractOpenRouterMetadata({
      id: "gen-789",
      usage: {
        inputTokensDetails: { cachedTokens: 11 },
        outputTokensDetails: { reasoningTokens: 5 },
      },
    });

    expect(metadata.cachedTokens).toBe(11);
    expect(metadata.reasoningTokens).toBe(5);
  });

  test("omits fields the response did not carry", () => {
    const metadata = extractOpenRouterMetadata({
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });

    expect(metadata).toEqual({});
  });

  test("never throws on a hostile response", () => {
    const hostile = {
      get id(): string {
        throw new Error("boom");
      },
    };

    expect(() => extractOpenRouterMetadata(hostile)).not.toThrow();
    expect(extractOpenRouterMetadata(hostile)).toEqual({});
  });
});
