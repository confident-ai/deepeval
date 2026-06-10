import * as path from "path";
import { DeepEvalCallbackHandler } from "../../../src/integrations/langchain";
import { traceManager } from "../../../src/tracing/tracing";
import { Environment } from "../../../src/tracing/utils";
import { generateTraceJson, assertTraceJson } from "../utils";

import { invokeSimpleApp } from "./apps/langchain-simple-app";
import { invokeSingleToolApp } from "./apps/langchain-single-tool-app";
import { invokeCityInfo } from "./apps/langchain-multiple-tools-app";
import { invokeStreamingSingle } from "./apps/langchain-streaming-app";
import {
  invokeResearch,
  invokeFactCheck,
} from "./apps/langchain-conditional-app";
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { invokeParallelWeather } from "./apps/langchain-parallel-tools-app";
import { invokeRagApp } from "./apps/langchain-retriever-app";
import { invokeMultiStepAgent } from "./apps/langchain-agent-app";
import { invokeMetricCollectionApp } from "./apps/langchain-metric-collection-app";

const FIXTURES_DIR = path.join(__dirname, "fixtures");
const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

const traceTest = async (
  schemaName: string,
  executeFn: () => Promise<void>,
) => {
  const jsonPath = path.join(FIXTURES_DIR, schemaName);
  if (GENERATE_SCHEMAS) {
    await generateTraceJson(jsonPath, executeFn);
  } else {
    await assertTraceJson(jsonPath, executeFn);
  }
};

describe("LangChain Integration Tests", () => {
  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture simple LLM app trace", async () => {
    await traceTest("langchain_simple_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-simple-test",
        tags: ["langchain", "simple"],
        metadata: { test_type: "simple" },
        threadId: "simple-123",
        userId: "test-user",
      });

      await invokeSimpleApp(
        {
          messages: [
            { role: "user", content: "Say hello in one short sentence." },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture single tool app trace", async () => {
    await traceTest("langchain_single_tool_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-single-tool-test",
        tags: ["langchain", "single-tool"],
        threadId: "single-tool-123",
      });

      await invokeSingleToolApp(
        {
          messages: [
            {
              role: "user",
              content:
                "Use the get_weather tool to get weather for San Francisco.",
            },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture multi-tool app trace", async () => {
    await traceTest("langchain_multiple_tools_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-multi-tool-test",
        tags: ["langchain", "multiple-tools"],
      });

      await invokeCityInfo(
        {
          messages: [
            { role: "user", content: "What's the weather in Tokyo?" },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture streaming app trace", async () => {
    await traceTest("langchain_streaming_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-streaming-test",
        tags: ["langchain", "streaming"],
      });

      await invokeStreamingSingle(
        {
          messages: [
            { role: "user", content: "What is the stock price for Apple?" },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture conditional routing app trace", async () => {
    await traceTest("langchain_conditional_research_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-conditional-test",
        tags: ["langchain", "conditional"],
      });

      await invokeResearch(
        {
          messages: [
            {
              role: "user",
              content: "Research the current state of Quantum Computing.",
            },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture parallel tool app trace", async () => {
    await traceTest("langchain_parallel_weather_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-parallel-test",
        tags: ["langchain", "parallel"],
      });

      await invokeParallelWeather(
        {
          messages: [
            {
              role: "user",
              content: "What's the weather in Tokyo and New York?",
            },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture RAG app trace", async () => {
    await traceTest("langchain_retriever_python_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-rag-test",
        tags: ["langchain", "rag"],
      });

      await invokeRagApp(
        {
          messages: [{ role: "user", content: "Tell me about Python." }] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture multi-step agent app trace", async () => {
    await traceTest("langchain_agent_multi_step_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "langchain-agent-test",
        tags: ["langchain", "agent"],
      });

      await invokeMultiStepAgent(
        {
          messages: [
            {
              role: "user",
              content: "Search for Apple stock price and explain it.",
            },
          ] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);

  test("Should capture all Confident AI attributes in LangChain trace", async () => {
    await traceTest("langchain_metric_collection_schema.json", async () => {
      const callback = new DeepEvalCallbackHandler({
        name: "LangChain Full Attributes Trace",
        threadId: "thread-xyz-789",
        userId: "user-alpha-123",
        tags: ["langchain", "integration-test", "full-attributes"],
        metricCollection: "root-trace-evals",
        metadata: {
          deployment: "test-env",
          runner: "jest",
        },
      });

      await invokeMetricCollectionApp(
        {
          messages: [{ role: "user", content: "What is 125 * 5?" }] as any,
        },
        { callbacks: [callback as unknown as BaseCallbackHandler] },
      );
    });
  }, 30000);
});
