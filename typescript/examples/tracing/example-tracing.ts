import { LLMTestCase, ToolCall } from "../../src/test-case";
import {
  observe,
  updateCurrentSpan,
  traceManager,
  updateCurrentTrace,
} from "../../src/tracing/tracing";
import { wait } from "../../src/utils";

const maskingFunction = (data: any): any => {
  if (typeof data === "string") {
    const redactedData = data.replace(
      /\b(?:\d{4}[- ]?){3}\d{4}\b/g,
      "[REDACTED CARD]",
    );
    return redactedData;
  }
  return data;
};

traceManager.configure({
  environment: "development",
  samplingRate: 1,
  mask: maskingFunction,
});

// Tool
const webSearch = observe({
  type: "tool",
  name: "web_search",
  fn: async (query: string) => {
    // <--Include implementation to search web here-->
    return "Latest search results for: " + query;
  },
});

// Retriever
const retrieveDocuments = observe({
  type: "retriever",
  embedder: "text-embedding-ada-002",
  fn: async (query: string) => {
    // <--Include implementation to fetch from vector database here-->
    const fetchedDocuments = [
      "Document 1: This is relevant information about the query.",
      "Document 2: More relevant information here.",
      "Document 3: Additional context that might be useful.",
    ];

    updateCurrentSpan({
      input: query,
      retrievalContext: fetchedDocuments,
    });

    return fetchedDocuments;
  },
});

// LLM
const generateResponse = observe({
  type: "llm",
  model: "gpt-4",
  fn: async (input: string) => {
    // <--Include format prompts and call your LLM provider here-->
    const output = "Generated response based on the prompt: " + input;

    await wait(1000);

    updateCurrentSpan({
      input: input,
      output: output,
      // inputTokenCount: 10,
      // outputTokenCount: 20,
    });

    return output;
  },
});

// Custom span wrapping the RAG pipeline
const ragPipeline = observe({
  type: "Custom Type",
  name: "RAG Pipeline",
  metricCollection: "My Metrics",
  fn: async (query: string) => {
    // Retrieve
    const docs = await retrieveDocuments(query);
    const context = docs.join("\n");

    // Generate
    const response = await generateResponse(
      `Context: ${context}\nQuery: ${query}`,
    );
    updateCurrentSpan({
      input: query,
      output: response,
      name: "sample span name",

      testCase: new LLMTestCase({
        input: query,
        actualOutput: response,
        expectedOutput: response,
        retrievalContext: ["Context"],
        context: ["Context"],
        toolsCalled: [
          new ToolCall({
            name: "web_search",
          }),
        ],
        expectedTools: [
          new ToolCall({
            name: "web_search",
          }),
        ],
      }),
      metadata: {
        testMetadataOne: "1234567890",
        testMetadataTwo: "1234567890",
      },
    });
    return response;
  },
});

// Agent that does RAG + tool calling
const researchAgent = observe({
  type: "agent",
  name: "Research Agent - Example",
  availableTools: ["web_search"],
  metricCollection: "My Metrics",
  fn: async (query: string) => {
    const initialResponse = await ragPipeline(query);
    const searchResults = await webSearch(initialResponse);
    const finalResponse = await generateResponse(
      `Initial response: ${initialResponse}\n` +
        `Additional search results: ${searchResults}\n` +
        `Query: ${query}`,
    );

    updateCurrentSpan({
      name: "sample span name",
      input: query,
      output: finalResponse,
      testCase: new LLMTestCase({
        input: query,
        actualOutput: finalResponse,
        expectedOutput: finalResponse,
        retrievalContext: ["Context"],
        context: ["Context"],
        toolsCalled: [
          new ToolCall({
            name: "web_search",
          }),
        ],
        expectedTools: [
          new ToolCall({
            name: "web_search",
          }),
        ],
      }),
      metadata: {
        testMetadataOne: "1234567890",
        testMetadataTwo: "1234567890",
      },
    });

    updateCurrentTrace({
      name: "sample trace name",
      input: "sample trace input",
      output: "sample trace output",
      metadata: {
        testMetadataOne: "1234567890",
        testMetadataTwo: "1234567890",
      },
      tags: ["sample tag one", "sample tag two"],
      threadId: "sample thread id",
      userId: "sample user id",
    });

    return finalResponse;
  },
});

(async () => {
  try {
    await researchAgent("What is the weather like in San Francisco?");
    await researchAgent("What is the weather like in San Francisco?");
    await researchAgent("What is the weather like in San Francisco?");
    await researchAgent("What is the weather like in San Francisco?");
  } catch (error) {
    console.error("Error running agent:", error);
  }
})();
