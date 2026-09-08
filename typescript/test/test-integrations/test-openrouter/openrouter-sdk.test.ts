// Tracing tests for the native `@openrouter/sdk` wrapper.
//
// All HTTP is stubbed, so this runs offline — it lives here rather than in
// test-core only because `@openrouter/sdk` is ESM-only and needs the babel
// transform in jest.openrouter.config.cjs to load under Jest's CJS runtime.
//
// Run: npx jest -c jest.openrouter.config.cjs

import { HTTPClient, OpenRouter } from "@openrouter/sdk";

import { EvaluationDataset, Golden } from "@/dataset";
import { instrumentOpenRouter } from "@/openrouter";
import { unpatchOpenRouter } from "@/openrouter/patch";
import { setTracingContext } from "@/tracing/trace-context";
import { traceManager } from "@/tracing/tracing";
import { Environment } from "@/tracing/utils";

import {
  CHAT_RESPONSE,
  TOOL_CALL_RESPONSE,
  WEATHER_TOOL,
  mockFetch,
} from "../../test-core/openrouter-fixtures";

function openRouterClient(payload: unknown) {
  return new OpenRouter({
    apiKey: "test-key",
    httpClient: new HTTPClient({ fetcher: mockFetch(payload) as any }),
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

describe("native @openrouter/sdk wrapper", () => {
  let client: OpenRouter;

  afterEach(() => {
    if (client) unpatchOpenRouter(client as any);
  });

  test("chat.send produces an LLM span", async () => {
    client = openRouterClient(CHAT_RESPONSE);
    instrumentOpenRouter(client);

    const trace = await captureTrace(() =>
      client.chat.send({
        chatRequest: {
          model: "anthropic/claude-sonnet-4.5",
          messages: [{ role: "user", content: "hi" }],
        },
      }),
    );

    const span = onlyLlmSpan(trace);
    expect(span.model).toBe("anthropic/claude-sonnet-4.5");
    expect(span.output).toBe("Hello there!");
    expect(span.inputTokenCount).toBe(12);
    expect(span.outputTokenCount).toBe(5);
    // The instrumented SDK and the serving provider are both OpenRouter here.
    expect(span.integration).toBe("OpenRouter");
    expect(span.provider).toBe("OpenRouter");
  });

  test("OpenRouter's extras land in span metadata", async () => {
    client = openRouterClient(CHAT_RESPONSE);
    instrumentOpenRouter(client);

    const trace = await captureTrace(() =>
      client.chat.send({
        chatRequest: {
          model: "anthropic/claude-sonnet-4.5",
          messages: [{ role: "user", content: "hi" }],
        },
      }),
    );

    const metadata = onlyLlmSpan(trace).metadata.openrouter;
    expect(metadata.generationId).toBe("gen-1750000000-abcdef123");
    expect(metadata.cost).toBe(0.000123);
    expect(metadata.isByok).toBe(false);
    expect(metadata.cachedTokens).toBe(4);
    expect(metadata.cacheWriteTokens).toBe(9);
    expect(metadata.reasoningTokens).toBe(2);
    // Routing detail only the native SDK exposes.
    expect(metadata.routing).toMatchObject({
      strategy: "price",
      summary: "routed to Anthropic",
    });
  });

  test("tool calls are captured on the LLM span", async () => {
    client = openRouterClient(TOOL_CALL_RESPONSE);
    instrumentOpenRouter(client);

    const trace = await captureTrace(() =>
      client.chat.send({
        chatRequest: {
          model: "anthropic/claude-sonnet-4.5",
          messages: [{ role: "user", content: "weather in Paris?" }],
          tools: [WEATHER_TOOL] as any,
        },
      }),
    );

    const span = onlyLlmSpan(trace);
    expect(span.toolsCalled).toHaveLength(1);
    expect(span.toolsCalled[0]).toMatchObject({
      name: "get_weather",
      inputParameters: { city: "Paris" },
      // Carried over from the request's tool declaration.
      description: "Look up the weather",
    });
    // NOTE: no assertion on `trace.toolSpans` here. `createChildToolSpans`
    // pushes a child span that does not survive into the serialized trace —
    // a pre-existing gap in the TS tracer that the OpenAI integration hits
    // identically (Python does emit these). Pinning the current, working
    // behaviour rather than a fix this integration does not own.
    expect(trace.toolSpans).toEqual([]);
  });

  test("the provider label can be overridden by the user", async () => {
    client = openRouterClient(CHAT_RESPONSE);
    instrumentOpenRouter(client);

    const trace = await captureTrace(() =>
      setTracingContext({ llmSpanContext: { provider: "MyRouter" } }, () =>
        client.chat.send({
          chatRequest: {
            model: "anthropic/claude-sonnet-4.5",
            messages: [{ role: "user", content: "hi" }],
          },
        }),
      ),
    );

    const span = onlyLlmSpan(trace);
    expect(span.provider).toBe("MyRouter");
    // Overriding the provider must not relabel which SDK was instrumented.
    expect(span.integration).toBe("OpenRouter");
    // Relabelling is cosmetic; the cost data must survive it.
    expect(span.metadata.openrouter.cost).toBe(0.000123);
  });

  test("streaming calls are passed through untouched", async () => {
    // Streaming is out of scope: the span would close before content arrived.
    client = openRouterClient(CHAT_RESPONSE);
    instrumentOpenRouter(client);

    traceManager.clearTraces();
    const result = await client.chat.send({
      chatRequest: {
        model: "anthropic/claude-sonnet-4.5",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      },
    } as any);

    // Handed straight back, and no LLM span invented for it.
    expect(result).toBeDefined();
    const traces = traceManager.getAllTraces();
    const llmSpans = traces.flatMap(
      (t: any) => (traceManager as any).createTraceApi(t).llmSpans,
    );
    expect(llmSpans).toHaveLength(0);
  });

  test("nests under an enclosing observe() span", async () => {
    client = openRouterClient(CHAT_RESPONSE);
    instrumentOpenRouter(client);

    const { observe } = await import("@/tracing");

    const trace = await captureTrace(
      observe({
        type: "agent",
        name: "my-agent",
        fn: async () =>
          client.chat.send({
            chatRequest: {
              model: "anthropic/claude-sonnet-4.5",
              messages: [{ role: "user", content: "hi" }],
            },
          }),
      }) as () => Promise<unknown>,
    );

    expect(trace.agentSpans).toHaveLength(1);
    const agent = trace.agentSpans[0];
    const llm = onlyLlmSpan(trace);
    expect(llm.parentUuid).toBe(agent.uuid);
    expect(llm.provider).toBe("OpenRouter");
    // The enclosing span is attributed to the integration that produced it.
    expect(agent.integration).toBe("OpenRouter");
  });

  test("spans reach traceManager under evalsIterator", async () => {
    client = openRouterClient(CHAT_RESPONSE);
    instrumentOpenRouter(client);

    traceManager.clearTraces();
    const dataset = new EvaluationDataset({
      goldens: [new Golden({ input: "hi there" })],
    });

    for await (const golden of dataset.evalsIterator({
      displayConfig: { showIndicator: false, printResults: false } as any,
    })) {
      await client.chat.send({
        chatRequest: {
          model: "anthropic/claude-sonnet-4.5",
          messages: [{ role: "user", content: golden.input! }],
        },
      });
    }

    const llmSpans = traceManager
      .getAllTraces()
      .flatMap((t: any) => (traceManager as any).createTraceApi(t).llmSpans);

    expect(llmSpans.length).toBeGreaterThan(0);
    expect(llmSpans[0].provider).toBe("OpenRouter");
  }, 30000);

  test("unpatch restores the original method", async () => {
    client = openRouterClient(CHAT_RESPONSE);
    const original = client.chat.send;
    instrumentOpenRouter(client);
    expect(client.chat.send).not.toBe(original);

    unpatchOpenRouter(client as any);
    expect(client.chat.send).toBe(original);
  });
});
