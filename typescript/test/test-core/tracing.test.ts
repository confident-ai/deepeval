import { config } from "dotenv";

import {
  getCurrentSpan,
  observe,
  traceManager,
  updateCurrentSpan,
  updateCurrentTrace,
  updateLlmSpan,
  updateRetrieverSpan,
  SpanType,
  TraceSpanStatus,
} from "../../src/tracing/tracing";
import { TraceSpanApiStatus } from "../../src/tracing/api";
import { LLMTestCase } from "../../src/test-case";
import { Environment } from "../../src/tracing/utils";

config();
process.env.CONFIDENT_TRACE_VERBOSE = "YES";

describe("Tracing Module", () => {
  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      samplingRate: 1,
      mask: (data) => data,
    });
  });

  test("Should create a trace with root span", async () => {
    const testFunction = observe({
      type: "CUSTOM",
      name: "testFunction",
      fn: async () => "test result",
    });
    const result = await testFunction();

    expect(result).toBe("test result");
    expect(traceManager.getAllTraces().length).toBeGreaterThan(0);
  });

  test("Should handle nested spans correctly", async () => {
    const innerFunction = observe({
      type: SpanType.RETRIEVER,
      name: "innerFunction",
      fn: async () => {
        updateCurrentSpan({
          input: "test input",
          output: "test output",
          retrievalContext: ["context1", "context2"],
        });
        return "inner result";
      },
    });
    const outerFunction = observe({
      type: SpanType.LLM,
      model: "test-model",
      name: "outerFunction",
      fn: async () => {
        const result = await innerFunction();
        return `outer result: ${result}`;
      },
    });
    const result = await outerFunction();

    expect(result).toBe("outer result: inner result");
  });

  test("Should track LLM span with attributes", async () => {
    const llmFunction = observe({
      type: SpanType.LLM,
      model: "gpt-5-nano",
      name: "llmFunction",
      fn: async (prompt: string) => {
        const response = `Response to: ${prompt}`;
        updateLlmSpan({
          inputTokenCount: 10,
          outputTokenCount: 20,
        });
        return response;
      },
    });
    const result = await llmFunction("Test prompt");

    expect(result).toBe("Response to: Test prompt");
  });

  // test("Should track tool span with ToolCall attributes", async () => {
  //   const toolFunction = observe({
  //     type: SpanType.TOOL,
  //     name: "toolFunction",
  //     fn: async (query: string) => {
  //       const resultData = `Result for: ${query}`;
  //       const toolCall = new ToolCall({
  //         name: "toolFunction",
  //         inputParameters: { query },
  //         output: resultData,
  //       });
  //       updateCurrentSpan({
  //         toolsCalled: [toolCall],
  //         input: query,
  //         output: resultData,
  //         expectedTools: [toolCall],
  //       });

  //       const currentSpan = getCurrentSpan();
  //       return {
  //         data: resultData,
  //         toolsCalled: currentSpan?.toolsCalled?.[0],
  //         expectedToolCall: currentSpan?.expectedTools?.[0],
  //       };
  //     },
  //   });

  //   const result = await toolFunction("test query");
  //   const currentSpan = getCurrentSpan();
  //   const expectedToolCall = new ToolCall({
  //     name: "toolFunction",
  //     inputParameters: {
  //       query: "test query",
  //     },
  //     output: "Result for: test query",
  //   });

  //   expect(result).toEqual({
  //     data: "Result for: test query",
  //     toolsCalled: expectedToolCall,
  //     expectedToolCall: expectedToolCall,
  //   });

  //   expect(currentSpan?.toolsCalled?.[0]).toEqual(expectedToolCall);
  //   expect(currentSpan?.expectedTools?.[0]).toEqual(expectedToolCall);
  // });

  test("Should track retriever span with attributes", async () => {
    const retrieverFunction = observe({
      type: SpanType.RETRIEVER,
      name: "retrieverFunction",
      fn: async (query: string) => {
        const docs = ["Doc 1", "Doc 2", query];
        updateRetrieverSpan({
          embedder: "test-embedder",
          topK: 8,
          chunkSize: 10,
        });
        return docs;
      },
    });
    const result = await retrieverFunction("test query");

    expect(result).toEqual(["Doc 1", "Doc 2", "test query"]);
  });

  test("Should track agent span with attributes", async () => {
    const agentFunction = observe({
      type: SpanType.AGENT,
      name: "agentFunction",
      availableTools: ["tool1", "tool2"],
      fn: async (query: string) => {
        const response = `Agent response to: ${query}`;
        updateCurrentSpan({
          input: query,
          output: response,
          testCase: new LLMTestCase({
            input: query,
            actualOutput: response,
          }),
        });
        return response;
      },
    });
    const result = await agentFunction("test query");

    expect(result).toBe("Agent response to: test query");
  });

  test("Should silently drop trace when drop is true", async () => {
    const testFunction = observe({
      type: "CUSTOM",
      name: "droppedFunction",
      fn: async () => {
        updateCurrentTrace({ drop: true });
        return "result";
      },
    });
    await testFunction();

    const traces = traceManager.getAllTraces();
    expect(traces.length).toBeGreaterThan(0);
    const trace = traces[traces.length - 1];
    expect(trace.drop).toBe(true);

    traceManager.configure({ tracingEnabled: true });
    const result = traceManager.postTrace(trace);
    expect(result).toBeUndefined();
  });

  test("Should mark trace ERRORED when error propagates to root observe (uncaught)", async () => {
    const innerFunction = observe({
      type: SpanType.TOOL,
      name: "innerThrows",
      fn: async () => {
        throw new Error("inner failure");
      },
    });
    const outerFunction = observe({
      type: SpanType.AGENT,
      name: "outerUncaught",
      fn: async () => {
        return await innerFunction();
      },
    });

    await expect(outerFunction()).rejects.toThrow("inner failure");

    const traces = traceManager.getAllTraces();
    expect(traces.length).toBeGreaterThan(0);
    const trace = traces[traces.length - 1];
    expect(trace.status).toBe(TraceSpanStatus.ERRORED);
    expect(trace.rootSpans[0].status).toBe(TraceSpanStatus.ERRORED);

    const traceApi = (traceManager as any).createTraceApi(trace);
    expect(traceApi.status).toBe(TraceSpanApiStatus.ERROR);
  });

  test("Should keep trace SUCCESS when child observe error is caught", async () => {
    const innerFunction = observe({
      type: SpanType.TOOL,
      name: "innerThrows",
      fn: async () => {
        throw new Error("inner failure");
      },
    });
    const outerFunction = observe({
      type: SpanType.AGENT,
      name: "outerCatches",
      fn: async () => {
        try {
          await innerFunction();
        } catch {
          return "recovered";
        }
      },
    });

    const result = await outerFunction();
    expect(result).toBe("recovered");

    const traces = traceManager.getAllTraces();
    expect(traces.length).toBeGreaterThan(0);
    const trace = traces[traces.length - 1];
    expect(trace.status).toBe(TraceSpanStatus.SUCCESS);
    expect(trace.rootSpans[0].status).toBe(TraceSpanStatus.SUCCESS);

    const erroredChild = trace.rootSpans[0].children.find(
      (child) => child.status === TraceSpanStatus.ERRORED,
    );
    expect(erroredChild).toBeDefined();
    expect(erroredChild?.error).toContain("inner failure");

    const traceApi = (traceManager as any).createTraceApi(trace);
    expect(traceApi.status).toBe(TraceSpanApiStatus.SUCCESS);
  });

  test("Should update trace with input and output", async () => {
    const testFunction = observe({
      type: "CUSTOM",
      name: "testFunction",
      fn: async (query: string) => {
        updateCurrentTrace({
          input: `trace input: ${query}`,
          output: `trace output for ${query}`,
          tags: ["tag1", "tag2"],
        });
        return "result";
      },
    });
    const QUERY = "SOME_QUERY";
    await testFunction(QUERY);
    const traces = traceManager.getAllTraces();

    expect(traces.length).toBeGreaterThan(0);
    expect(traces[0].input).toBe(`trace input: ${QUERY}`);
    expect(traces[0].output).toBe(`trace output for ${QUERY}`);
    expect(traces[0].tags).toEqual(["tag1", "tag2"]);
  });
});
