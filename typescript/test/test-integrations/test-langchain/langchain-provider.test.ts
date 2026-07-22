import { DeepEvalCallbackHandler } from "../../../src/integrations/langchain/callback-handler";
import { safeExtractProvider } from "../../../src/integrations/langchain/utils";
import { traceManager } from "../../../src/tracing/tracing";
import { Environment } from "../../../src/tracing/utils";
import { Integration, Provider } from "../../../src/tracing/integrations";

// A minimal Serialized-shaped object; the handler only reads `.name`.
const serialized = (name?: string): any => ({
  lc: 1,
  type: "not_implemented",
  id: ["test"],
  ...(name ? { name } : {}),
});

const CHAIN_RUN = "11111111-1111-1111-1111-111111111111";
const LLM_RUN = "22222222-2222-2222-2222-222222222222";
const TOOL_RUN = "33333333-3333-3333-3333-333333333333";
const RETRIEVER_RUN = "44444444-4444-4444-4444-444444444444";

describe("LangChain integration: provider + integration tagging", () => {
  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
      samplingRate: 1,
    });
  });

  describe("safeExtractProvider", () => {
    test("reads model_provider from invocation params (camel + snake case)", () => {
      expect(
        safeExtractProvider(undefined, {
          invocationParams: { model_provider: "anthropic" },
        }),
      ).toBe("anthropic");
      expect(
        safeExtractProvider(undefined, {
          invocation_params: { model_provider: "openai" },
        }),
      ).toBe("openai");
    });

    test("reads ls_provider / model_provider from metadata", () => {
      expect(safeExtractProvider({ ls_provider: "openai" })).toBe("openai");
      expect(safeExtractProvider({ model_provider: "mistral" })).toBe("mistral");
    });

    test("returns undefined when no hint present", () => {
      expect(safeExtractProvider({}, {})).toBeUndefined();
      expect(safeExtractProvider()).toBeUndefined();
    });
  });

  describe("callback handler span tagging", () => {
    test("chain, llm, tool and retriever spans are tagged LangChain; provider is inferred", async () => {
      const handler = new DeepEvalCallbackHandler({
        name: "test-graph",
        threadId: "provider-thread-1",
      });

      // Root chain span.
      await handler.handleChainStart(
        serialized(),
        { question: "hi" },
        CHAIN_RUN,
      );
      const rootSpan = traceManager.getSpanByUuid(CHAIN_RUN);
      expect(rootSpan?.integration).toBe(Integration.LANGCHAIN);

      // LLM span with only a model name -> provider inferred from the model.
      await handler.handleLLMStart(
        serialized("claude-3-5-haiku-latest"),
        ["hello"],
        LLM_RUN,
        CHAIN_RUN,
      );
      const llmSpan: any = traceManager.getSpanByUuid(LLM_RUN);
      expect(llmSpan.integration).toBe(Integration.LANGCHAIN);
      expect(llmSpan.provider).toBe(Provider.ANTHROPIC);

      // Tool span.
      await handler.handleToolStart(
        serialized(),
        "tool input",
        TOOL_RUN,
        CHAIN_RUN,
      );
      expect(traceManager.getSpanByUuid(TOOL_RUN)?.integration).toBe(
        Integration.LANGCHAIN,
      );

      // Retriever span.
      await handler.handleRetrieverStart(
        serialized(),
        "query",
        RETRIEVER_RUN,
        CHAIN_RUN,
      );
      expect(traceManager.getSpanByUuid(RETRIEVER_RUN)?.integration).toBe(
        Integration.LANGCHAIN,
      );
    });

    test("handleLLMEnd overrides the inferred provider with the response's model_provider", async () => {
      const handler = new DeepEvalCallbackHandler({
        name: "test-graph-2",
        threadId: "provider-thread-2",
      });

      await handler.handleChainStart(serialized(), {}, CHAIN_RUN);
      await handler.handleLLMStart(
        serialized("gpt-4o-mini"),
        ["hello"],
        LLM_RUN,
        CHAIN_RUN,
      );

      // Hold a reference; handleLLMEnd mutates the span in place.
      const llmSpan: any = traceManager.getSpanByUuid(LLM_RUN);
      expect(llmSpan.provider).toBe(Provider.OPEN_AI); // inferred at start

      await handler.handleLLMEnd(
        {
          generations: [
            [
              {
                text: "hi there",
                message: {
                  content: "hi there",
                  tool_calls: [],
                  invalid_tool_calls: [],
                  response_metadata: {
                    model_name: "claude-3-5-sonnet-latest",
                    model_provider: "anthropic",
                    tokenUsage: { promptTokens: 5, completionTokens: 7 },
                  },
                },
              },
            ],
          ],
          llmOutput: {},
        } as any,
        LLM_RUN,
      );

      expect(llmSpan.provider).toBe(Provider.ANTHROPIC); // response metadata wins
      expect(llmSpan.model).toBe("claude-3-5-sonnet-latest");
      expect(llmSpan.inputTokenCount).toBe(5);
      expect(llmSpan.outputTokenCount).toBe(7);
    });
  });
});
