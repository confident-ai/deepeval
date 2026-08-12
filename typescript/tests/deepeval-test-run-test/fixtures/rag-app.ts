import { observe, updateCurrentSpan, updateCurrentTrace, SpanType } from "deepeval/tracing";
import { LLMTestCase } from "deepeval";
import { answerRelevancy, faithfulness, contextualRelevancy } from "./metrics";

const KB: Record<string, { context: string[]; answer: string }> = {
  "What is the capital of France?": {
    context: ["France is a country in Europe. Its capital city is Paris."],
    answer: "The capital of France is Paris.",
  },
  "Who wrote Romeo and Juliet?": {
    context: ["Romeo and Juliet is a tragedy written by William Shakespeare."],
    answer: "Romeo and Juliet was written by William Shakespeare.",
  },
};

const lookup = (query: string) =>
  KB[query] ?? {
    context: ["No relevant information was found."],
    answer: "I don't know.",
  };

const retrieve = observe({
  type: SpanType.RETRIEVER,
  name: "retriever",
  embedder: "text-embedding-3-small",
  metrics: [contextualRelevancy()],
  fn: async (query: string): Promise<string[]> => {
    const { context } = lookup(query);
    updateCurrentSpan({ input: query, retrievalContext: context });
    return context;
  },
});

const generate = observe({
  type: SpanType.LLM,
  name: "generator",
  model: "gpt-4o-mini",
  metrics: [answerRelevancy(), faithfulness()],
  fn: async (query: string, context: string[]): Promise<string> => {
    const { answer } = lookup(query);
    updateCurrentSpan({
      testCase: new LLMTestCase({
        input: query,
        actualOutput: answer,
        retrievalContext: context,
      }),
    });
    return answer;
  },
});

export const ragApp = observe({
  type: SpanType.AGENT,
  name: "rag_app",
  fn: async (query: string): Promise<string> => {
    const context = await retrieve(query);
    const answer = await generate(query, context);
    updateCurrentTrace({ input: query, output: answer });
    return answer;
  },
});
