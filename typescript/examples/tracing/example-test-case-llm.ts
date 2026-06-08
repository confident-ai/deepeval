import { observe, updateCurrentSpan } from "../../src/tracing/tracing";
import { ToolCall, LLMTestCase } from "../../src/test-case";

///////////////////////////////////////////////
// Agent Span
///////////////////////////////////////////////

const toolCallingAgent = (query: string) => {
  updateCurrentSpan({
    testCase: new LLMTestCase({
      input: query,
      actualOutput: "Agent response",
      expectedOutput: "Agent response",
      context: ["context", "context_2"],
      retrievalContext: ["retrieval_context", "retrieval_context_2"],
      expectedTools: [
        new ToolCall({
          name: "web_search",
          inputParameters: { query: query },
        }),
      ],
      toolsCalled: [
        new ToolCall({
          name: "web_search",
          inputParameters: { query: query },
        }),
      ],
    }),
  });
  return "Agent response";
};

const observedToolCallingAgent = observe({
  type: "agent",
  name: "tool_calling_agent",
  metricCollection: "metric_collection",
  fn: toolCallingAgent,
});

observedToolCallingAgent("What is weather in San Francisco?");

///////////////////////////////////////////////
// LLM Span
///////////////////////////////////////////////

const llmSpan = (query: string) => {
  updateCurrentSpan({
    testCase: new LLMTestCase({
      input: query,
      output: "LLM response",
      expectedOutput: "LLM response",
      context: ["context", "context_2"],
      retrievalContext: ["retrieval_context", "retrieval_context_2"],
      expectedTools: [
        new ToolCall({
          name: "web_search",
          inputParameters: { query: query },
        }),
      ],
      toolsCalled: [
        new ToolCall({
          name: "web_search",
          inputParameters: { query: query },
        }),
      ],
    }),
  });
  return "LLM response";
};

const observedLLMSpan = observe({
  type: "llm",
  model: "gpt-3.5-turbo",
  metricCollection: "metric_collection",
  fn: llmSpan,
});

observedLLMSpan("What is weather in San Francisco?");
