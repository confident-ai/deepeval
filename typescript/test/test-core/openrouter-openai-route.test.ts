// Part B: reaching OpenRouter through the OpenAI SDK.
//
// `@/openai` already traced these calls; what it could not do is say who served
// them. Only the client's base URL can. These tests pin that, and pin that a
// plain OpenAI client is unaffected by it. All HTTP is stubbed.

import { OpenAI } from "openai";

import { patchOpenAI, unpatchOpenAI } from "@/openai/patch";
import { setTracingContext } from "@/tracing/trace-context";
import { traceManager } from "@/tracing/tracing";
import { Environment } from "@/tracing/utils";

import {
  OPENAI_COMPAT_RESPONSE,
  VANILLA_OPENAI_RESPONSE,
  mockFetch,
} from "./openrouter-fixtures";

function openAIClient(baseURL: string | undefined, payload: unknown) {
  return new OpenAI({
    apiKey: "test-key",
    baseURL,
    fetch: mockFetch(payload) as any,
  });
}

/** Run `fn`, then return the trace deepeval serialized for it. */
async function captureTrace(fn: () => Promise<unknown>) {
  traceManager.clearTraces();
  await fn();

  const traces = traceManager.getAllTraces();
  expect(traces.length).toBeGreaterThan(0);
  return (traceManager as any).createTraceApi(traces[0]);
}

function onlyLlmSpan(trace: any) {
  expect(trace.llmSpans).toHaveLength(1);
  return trace.llmSpans[0];
}

beforeEach(() => {
  traceManager.clearTraces();
  traceManager.configure({
    environment: Environment.TESTING,
    tracingEnabled: true,
  });
});

describe("OpenRouter through the OpenAI SDK", () => {
  let client: OpenAI;

  afterEach(() => {
    if (client) unpatchOpenAI(client);
  });

  test("an OpenRouter base URL attaches OpenRouter metadata", async () => {
    client = openAIClient(
      "https://openrouter.ai/api/v1",
      OPENAI_COMPAT_RESPONSE,
    );
    patchOpenAI(client);

    const trace = await captureTrace(() =>
      client.chat.completions.create({
        model: "anthropic/claude-sonnet-4.5",
        messages: [{ role: "user", content: "hi" }],
      }),
    );

    const span = onlyLlmSpan(trace);
    // The integration is the SDK instrumented; the provider is who served it.
    expect(span.integration).toBe("OpenAI");
    expect(span.provider).toBe("OpenRouter");

    const metadata = span.metadata.openrouter;
    expect(metadata.generationId).toBe("gen-1750000000-abcdef123");
    // Only the OpenAI-compatible endpoint names the upstream that served it.
    expect(metadata.upstreamProvider).toBe("Anthropic");
    expect(metadata.cost).toBe(0.000123);
    expect(metadata.cachedTokens).toBe(4);
    expect(metadata.reasoningTokens).toBe(2);
  });

  test("a plain OpenAI client is unaffected", async () => {
    client = openAIClient(undefined, VANILLA_OPENAI_RESPONSE);
    patchOpenAI(client);

    const trace = await captureTrace(() =>
      client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      }),
    );

    const span = onlyLlmSpan(trace);
    expect(span.output).toBe("Hello there!");
    expect(span.integration).toBe("OpenAI");
    expect(span.provider).toBe("OpenAI");
    // No OpenRouter metadata is invented for a non-OpenRouter call.
    expect(span.metadata?.openrouter).toBeUndefined();
  });

  test("an override does not discard OpenRouter's metadata", async () => {
    client = openAIClient(
      "https://openrouter.ai/api/v1",
      OPENAI_COMPAT_RESPONSE,
    );
    patchOpenAI(client);

    const trace = await captureTrace(() =>
      setTracingContext({ llmSpanContext: { provider: "CustomLabel" } }, () =>
        client.chat.completions.create({
          model: "anthropic/claude-sonnet-4.5",
          messages: [{ role: "user", content: "hi" }],
        }),
      ),
    );

    const span = onlyLlmSpan(trace);
    expect(span.provider).toBe("CustomLabel");
    expect(span.integration).toBe("OpenAI");
    expect(span.metadata.openrouter.cost).toBe(0.000123);
  });

  test("user metadata on the span is preserved alongside ours", async () => {
    client = openAIClient(
      "https://openrouter.ai/api/v1",
      OPENAI_COMPAT_RESPONSE,
    );
    patchOpenAI(client);

    const { observe, updateCurrentSpan } = await import("@/tracing");

    const trace = await captureTrace(
      observe({
        type: "llm",
        name: "wrapper",
        fn: async () => {
          updateCurrentSpan({ metadata: { mine: "kept" } });
          return client.chat.completions.create({
            model: "anthropic/claude-sonnet-4.5",
            messages: [{ role: "user", content: "hi" }],
          });
        },
      }) as () => Promise<unknown>,
    );

    // The wrapper keeps the user's metadata; the inner call's span carries
    // ours. Neither clobbers the other.
    const wrapper = trace.llmSpans.find((s: any) => s.name === "wrapper");
    expect(wrapper.metadata).toMatchObject({ mine: "kept" });
    expect(wrapper.metadata.openrouter).toBeUndefined();

    const inner = trace.llmSpans.find((s: any) => s.name !== "wrapper");
    expect(inner.provider).toBe("OpenRouter");
  });
});
