// Set node env to development for openai agents to create traces
process.env.NODE_ENV = "development";

import * as path from "path";
import { run, addTraceProcessor } from "@openai/agents";
import { DeepEvalTracingProcessor } from "../../../src/integrations/openai-agents";
import { traceManager } from "../../../src/tracing/tracing";
import { setTracingContext } from "../../../src/tracing/trace-context";
import { Prompt } from "../../../src/prompt";
import { Environment } from "../../../src/tracing/utils";
import { generateTraceJson, assertTraceJson } from "../utils";

// App Imports
import { simpleAgent } from "./apps/openai-agents-simple-app";
import { toolAgent } from "./apps/openai-agents-tool-app";
import { streamingAgent } from "./apps/openai-agents-streaming-app";
import { triageAgent } from "./apps/openai-agents-handoff-app";
import { evalAgent } from "./apps/openai-agents-eval-app";

const FIXTURES_DIR = path.join(__dirname, "fixtures");
const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

describe("OpenAI Agents Integration Tests", () => {
  let processor;

  beforeAll(() => {
    processor = new DeepEvalTracingProcessor();
    addTraceProcessor(processor);
  });

  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture simple agent trace", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "openai_agents_simple_schema.json",
    );

    const executeTest = async () => {
      await run(simpleAgent, "Say hello concisely.");
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture tool agent trace", async () => {
    const jsonPath = path.join(FIXTURES_DIR, "openai_agents_tool_schema.json");

    const executeTest = async () => {
      await setTracingContext(
        {
          llmSpanContext: {
            toolsMetricCollection: "tool-level-collection",
          },
        },
        async () => {
          await run(
            toolAgent,
            "What is the weather in London? Use your tools.",
          );
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture streaming agent trace", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "openai_agents_streaming_schema.json",
    );

    const executeTest = async () => {
      const result = await run(
        streamingAgent,
        "Write a short poem about artificial intelligence.",
        { stream: true },
      );

      for await (const chunk of result) {
        // Chunking happening
      }
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture handoff agent trace", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "openai_agents_handoff_schema.json",
    );

    const executeTest = async () => {
      await run(triageAgent, "Hola, ¿cómo estás?");
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture agent trace with full DeepEval context and metrics", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "openai_agents_evals_app_schema.json",
    );

    const executeTest = async () => {
      const prompt = new Prompt({ alias: "eval-test-prompt" });
      prompt.hash = "test-hash-123";
      prompt.version = "00.00.01";
      prompt.label = "prompt-label-1";

      await setTracingContext(
        {
          name: "OpenAI Agents Evaluation Trace",
          threadId: "thread-abc-123",
          userId: "user-999",
          metricCollection: "trace-level-metrics",
          llmSpanContext: {
            prompt: prompt,
            metricCollection: "llm-level-metrics",
            toolsMetricCollection: "tool-level-collection",
          },
          agentSpanContext: {
            metricCollection: "agent-level-collection",
          },
        },
        async () => {
          await run(evalAgent, "Verify metric collection propagation.");
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);
});
