import {
  inferProviderFromModel,
  normalizeSpanProviderForPlatform,
  Environment,
} from "../../src/tracing/utils";
import { Integration, Provider } from "../../src/tracing/integrations";
import {
  observe,
  updateCurrentSpan,
  getCurrentSpan,
  traceManager,
  LlmSpan,
  SpanType,
} from "../../src/tracing/tracing";

describe("Provider inference utilities", () => {
  describe("inferProviderFromModel", () => {
    const cases: [string, string | undefined][] = [
      ["gpt-4o-mini", Provider.OPEN_AI],
      ["gpt-4.1", Provider.OPEN_AI],
      ["o1-preview", Provider.OPEN_AI],
      ["o3-mini", Provider.OPEN_AI],
      ["o4-mini", Provider.OPEN_AI],
      ["text-embedding-3-small", Provider.OPEN_AI],
      ["claude-3-5-sonnet-latest", Provider.ANTHROPIC],
      ["claude-opus-4-20250514", Provider.ANTHROPIC],
      ["gemini-1.5-pro", Provider.GEMINI],
      ["mistral-large-latest", Provider.MISTRAL],
      ["mixtral-8x7b", Provider.MISTRAL],
      ["grok-2", Provider.X_AI],
      ["deepseek-chat", Provider.DEEP_SEEK],
      // provider prefixes / separators are stripped down to the model id
      ["openai/gpt-4o", Provider.OPEN_AI],
      ["anthropic:claude-3-haiku", Provider.ANTHROPIC],
      ["accounts/fireworks/models/mixtral-8x7b", Provider.MISTRAL],
      // unknown / empty
      ["some-unknown-model", undefined],
      ["", undefined],
    ];

    test.each(cases)("infers %s -> %s", (model, expected) => {
      expect(inferProviderFromModel(model)).toBe(expected);
    });

    test("returns undefined for null / undefined", () => {
      expect(inferProviderFromModel(null)).toBeUndefined();
      expect(inferProviderFromModel(undefined)).toBeUndefined();
    });
  });

  describe("normalizeSpanProviderForPlatform", () => {
    test("canonicalizes known provider spellings and enum keys", () => {
      expect(normalizeSpanProviderForPlatform("openai")).toBe(Provider.OPEN_AI);
      expect(normalizeSpanProviderForPlatform("OpenAI")).toBe(Provider.OPEN_AI);
      expect(normalizeSpanProviderForPlatform("OPEN_AI")).toBe(Provider.OPEN_AI);
      expect(normalizeSpanProviderForPlatform("anthropic")).toBe(
        Provider.ANTHROPIC,
      );
      expect(normalizeSpanProviderForPlatform("azure")).toBe(Provider.AZURE);
    });

    test("matches on the leading path/segment head", () => {
      expect(normalizeSpanProviderForPlatform("openai/deployment-name")).toBe(
        Provider.OPEN_AI,
      );
    });

    test("passes through unknown providers unchanged", () => {
      expect(normalizeSpanProviderForPlatform("acme-corp")).toBe("acme-corp");
    });

    test("returns undefined for null / empty", () => {
      expect(normalizeSpanProviderForPlatform(null)).toBeUndefined();
      expect(normalizeSpanProviderForPlatform("")).toBeUndefined();
      expect(normalizeSpanProviderForPlatform("   ")).toBeUndefined();
    });
  });

  describe("integration + provider on spans", () => {
    beforeEach(() => {
      traceManager.clearTraces();
      traceManager.configure({
        environment: Environment.TESTING,
        samplingRate: 1,
      });
    });

    test("updateCurrentSpan sets the integration on the current span", async () => {
      let captured: any;
      const fn = observe({
        type: SpanType.LLM,
        model: "gpt-4o-mini",
        name: "llm",
        fn: async () => {
          updateCurrentSpan({ integration: Integration.LANGCHAIN });
          captured = getCurrentSpan();
          return "ok";
        },
      });
      await fn();

      expect(captured.integration).toBe(Integration.LANGCHAIN);
    });

    test("integration + provider serialize into the API span", async () => {
      // Integrations set these directly on the LLM span (as the LangChain /
      // AI SDK / OpenAI processors do); assert they survive serialization.
      const fn = observe({
        type: SpanType.LLM,
        model: "gpt-4o-mini",
        name: "llm",
        fn: async () => {
          const span = getCurrentSpan() as LlmSpan;
          span.integration = Integration.OPEN_AI;
          span.provider = Provider.OPEN_AI;
          return "ok";
        },
      });
      await fn();

      const trace = traceManager.getAllTraces()[0];
      const apiTrace = (traceManager as any).createTraceApi(trace);
      const llmApiSpan = apiTrace.llmSpans[0];

      expect(llmApiSpan.integration).toBe(Integration.OPEN_AI);
      expect(llmApiSpan.provider).toBe(Provider.OPEN_AI);
    });
  });
});
