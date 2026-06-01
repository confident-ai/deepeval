import {
  observe,
  updateRetrieverSpan,
  updateLlmSpan,
} from "../../src/tracing/tracing";

///////////////////////////////////////////////
// Example
///////////////////////////////////////////////

const outerFunction = () => {
  const observedInnerFunction = observe({
    type: "retriever",
    fn: () => {
      updateRetrieverSpan({
        chunkSize: 10,
      });
    },
  });

  observedInnerFunction();
};

const observedOuterFunction = observe({ type: "custom", fn: outerFunction });

observedOuterFunction();

///////////////////////////////////////////////
//  LLM Span
///////////////////////////////////////////////

const generateResponse = (prompt: string) => {
  const output = `Generated response for prompt ${prompt}`;
  updateLlmSpan({
    inputTokenCount: 10,
    outputTokenCount: 25,
    costPerInputToken: 0,
    costPerOutputToken: 0,
  });
  return output;
};

const observedGenerateResponse = observe({
  type: "llm",
  model: "gpt-4",
  fn: generateResponse,
});
observedGenerateResponse("What is weather in San Francisco?");

///////////////////////////////////////////////
// Retriever Span
///////////////////////////////////////////////

const retrieveDocuments = (query: string) => {
  const fetchedDocuments = ["doc1", "doc2", query];
  updateRetrieverSpan({
    embedder: "text-embedding-ada-002",
    chunkSize: 10,
    topK: 5,
  });
  return fetchedDocuments;
};

const observedRetrieveDocuments = observe({
  type: "retriever",
  embedder: "text-embedding-ada-002",
  fn: retrieveDocuments,
});
observedRetrieveDocuments("What is weather in San Francisco?");
