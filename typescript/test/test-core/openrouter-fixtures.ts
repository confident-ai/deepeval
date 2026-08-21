// Wire-shaped OpenRouter responses, shared by the native-SDK and OpenAI-SDK
// suites.
//
// These are deliberately raw snake_case JSON, exactly as OpenRouter puts it on
// the wire. The OpenAI SDK hands that through untouched; the official SDK
// camelCases it on the way in. Both casings therefore get exercised from the
// same fixture, which is the point.

export const CHAT_RESPONSE = {
  id: "gen-1750000000-abcdef123",
  model: "anthropic/claude-sonnet-4.5",
  object: "chat.completion",
  created: 1750000000,
  system_fingerprint: null,
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello there!" },
      finish_reason: "stop",
      logprobs: null,
    },
  ],
  usage: {
    prompt_tokens: 12,
    completion_tokens: 5,
    total_tokens: 17,
    cost: 0.000123,
    is_byok: false,
    cost_details: {
      upstream_inference_cost: 0.0001,
      upstream_inference_prompt_cost: 0.00006,
      upstream_inference_completions_cost: 0.00004,
    },
    prompt_tokens_details: { cached_tokens: 4, cache_write_tokens: 9 },
    completion_tokens_details: { reasoning_tokens: 2 },
  },
  openrouter_metadata: {
    attempt: 1,
    endpoints: { available: [], total: 1 },
    is_byok: false,
    region: "us-east",
    requested: "anthropic/claude-sonnet-4.5",
    strategy: "price",
    summary: "routed to Anthropic",
  },
};

export const TOOL_CALL_RESPONSE = {
  ...structuredClone(CHAT_RESPONSE),
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: {
              name: "get_weather",
              arguments: '{"city": "Paris"}',
            },
          },
        ],
      },
      finish_reason: "tool_calls",
      logprobs: null,
    },
  ],
};

export const WEATHER_TOOL = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Look up the weather",
    parameters: { type: "object", properties: {} },
  },
};

// What OpenRouter returns over its OpenAI-compatible endpoint: the same extras,
// plus a top-level `provider` naming the upstream that actually served it.
export const OPENAI_COMPAT_RESPONSE = {
  ...structuredClone(CHAT_RESPONSE),
  provider: "Anthropic",
  openrouter_metadata: undefined as unknown as undefined,
};
delete (OPENAI_COMPAT_RESPONSE as any).openrouter_metadata;

// A plain OpenAI response, to prove the OpenRouter handling stays dormant.
export const VANILLA_OPENAI_RESPONSE = {
  id: "chatcmpl-abc123",
  model: "gpt-4o",
  object: "chat.completion",
  created: 1750000000,
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello there!" },
      finish_reason: "stop",
      logprobs: null,
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 },
};

/** A `fetch` that always answers with `payload`, so suites run offline. */
export function mockFetch(payload: unknown) {
  return async () =>
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
}
