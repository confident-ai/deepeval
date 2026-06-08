import { observe, updateCurrentSpan } from "../../src/tracing/tracing";

const observedRetriever = observe({
  fn: (_query: string) => {
    const retrieved_chunks = ["chunk1", "chunk2"];
    updateCurrentSpan({ retrievalContext: retrieved_chunks });
    return retrieved_chunks.join("\n");
  },
});

const observedLLMApp = observe({
  fn: (query: string) => {
    const _retrieval_context = observedRetriever(query);
    const res = "res";
    updateCurrentSpan({ input: query, output: res });
    return res;
  },
});

observedLLMApp("What is weather typically like in San Francisco?");
