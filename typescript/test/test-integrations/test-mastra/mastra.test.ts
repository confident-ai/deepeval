import * as path from "path";
import { DeepEvalExporter } from "../../../src/integrations/mastra";
import { traceManager } from "../../../src/tracing/tracing";
import { Environment } from "../../../src/tracing/utils";
import { generateTraceJson, assertTraceJson } from "../utils";

import { runSimpleApp } from "./apps/mastra-simple-app";
import { runToolApp } from "./apps/mastra-tool-app";
import { runMultiToolApp } from "./apps/mastra-multi-tool-app";

const FIXTURES_DIR = path.join(__dirname, "fixtures");
const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

const settleTraces = async (timeoutMs = 15000): Promise<void> => {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (
      traceManager.getActiveSpans().size === 0 &&
      traceManager.getAllTraces().length > 0
    ) {
      return;
    }
    await new Promise((r) => setTimeout(r, 50));
  }
};

const traceTest = async (schemaName: string, executeFn: () => Promise<void>) => {
  const jsonPath = path.join(FIXTURES_DIR, schemaName);
  if (GENERATE_SCHEMAS) {
    await generateTraceJson(jsonPath, executeFn);
  } else {
    await assertTraceJson(jsonPath, executeFn);
  }
};

describe("Mastra Integration Tests", () => {
  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture simple agent trace", async () => {
    await traceTest("mastra_simple_schema.json", async () => {
      const exporter = new DeepEvalExporter({
        name: "mastra-simple-test",
        tags: ["mastra", "simple"],
        metadata: { test_type: "simple" },
        threadId: "simple-123",
        userId: "test-user",
      });

      await runSimpleApp(exporter, "Say hello in one short sentence.");
      await settleTraces();
    });
  }, 60000);

  test("Should capture single tool agent trace", async () => {
    await traceTest("mastra_tool_schema.json", async () => {
      const exporter = new DeepEvalExporter({
        name: "mastra-tool-test",
        tags: ["mastra", "tool"],
      });

      await runToolApp(
        exporter,
        "Use the get_weather tool to get the weather in San Francisco.",
      );
      await settleTraces();
    });
  }, 60000);

  test("Should capture multi-tool agent trace", async () => {
    await traceTest("mastra_multi_tool_schema.json", async () => {
      const exporter = new DeepEvalExporter({
        name: "mastra-multi-tool-test",
        tags: ["mastra", "multi-tool"],
      });

      await runMultiToolApp(
        exporter,
        "What is the weather in Tokyo, and what is 125 * 5?",
      );
      await settleTraces();
    });
  }, 60000);

  test("Should capture agent trace with full DeepEval attributes and metric collections", async () => {
    await traceTest("mastra_full_attributes_schema.json", async () => {
      const exporter = new DeepEvalExporter({
        name: "Mastra Full Attributes Trace",
        threadId: "thread-xyz-789",
        userId: "user-alpha-123",
        tags: ["mastra", "integration-test", "full-attributes"],
        metadata: { deployment: "test-env", runner: "jest" },
        metricCollection: "root-trace-evals",
        llmMetricCollection: "llm-evals",
        agentMetricCollection: "agent-evals",
        toolMetricCollectionMap: { get_weather: "weather-tool-evals" },
      });

      await runToolApp(
        exporter,
        "Use the get_weather tool to get the weather in London.",
      );
      await settleTraces();
    });
  }, 60000);
});
