import { generateText, streamText, Output } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import * as path from "path";
import { Prompt } from "../../../src/prompt";

import { configureAiSdkTracing } from "../../../src/integrations/ai-sdk";
import { setTracingContext } from "../../../src/tracing/trace-context";
import { traceManager } from "../../../src/tracing/tracing";
import { Environment } from "../../../src/tracing/utils";
import { generateTraceJson, assertTraceJson } from "../utils";

const FIXTURES_DIR = path.join(__dirname, "fixtures");
const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

const tracer = configureAiSdkTracing({
  isTestMode: true,
  environment: "testing",
});

describe("Vercel AI SDK Core Integration Tests", () => {
  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture basic generateText with trace context", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_ai_sdk_generate_text.json",
    );

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "vercel-thread-1",
          name: "Vercel Text Gen Trace",
          tags: ["vercel", "generateText"],
        },
        async () => {
          await generateText({
            model: openai("gpt-4o-mini"),
            prompt: "Why is the sky blue? Answer in one short sentence.",
            experimental_telemetry: {
              isEnabled: true,
              tracer: tracer!,
            },
          });
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture streamText with trace context", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_ai_sdk_stream_text.json",
    );

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "vercel-thread-2",
          name: "Vercel Stream Gen Trace",
          tags: ["vercel", "streamText"],
        },
        async () => {
          const { textStream } = streamText({
            model: openai("gpt-4o-mini"),
            prompt: "Count from 1 to 5.",
            experimental_telemetry: {
              isEnabled: true,
              tracer: tracer!,
            },
          });
          for await (const chunk of textStream) {
            // consuming chunks...
          }
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture structured output via generateText", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_ai_sdk_structured_output.json",
    );

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "vercel-thread-3",
          name: "Vercel Object Gen Trace",
          tags: ["vercel", "generateText", "structuredSchema"],
        },
        async () => {
          await generateText({
            model: openai("gpt-4o-mini"),
            prompt:
              "Generate a recipe for a classic margarita. You MUST include EXACTLY 3 ingredients and EXACTLY 3 steps.",
            temperature: 0,
            output: Output.object({
              schema: z.object({
                recipe: z.object({
                  name: z.string(),
                  ingredients: z.array(
                    z.object({ name: z.string(), amount: z.string() }),
                  ),
                  steps: z.array(z.string()),
                }),
              }),
            }),
            experimental_telemetry: {
              isEnabled: true,
              tracer: tracer!,
            },
          });
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 30000);

  test("Should capture Confident AI specific attributes via setTracingContext", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_ai_sdk_confident_attributes.json",
    );

    // Create a mock prompt to test prompt versioning/tracking
    const mockPrompt = new Prompt({ alias: "ai-sdk-test-prompt" });
    mockPrompt.version = "01.00.00";
    mockPrompt.label = "testing";
    mockPrompt.hash = "abcdef123";

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "test_thread_id",
          userId: "test_user_id",
          name: "DeepEval Attributes Trace",
          tags: ["vercel", "attributes-test", "evaluation"],
          metadata: { custom_business_id: "biz_789" },
          metricCollection: "global-trace-metrics",
          llmSpanContext: {
            prompt: mockPrompt,
            metricCollection: "llm-specific-metrics",
          },
        },
        async () => {
          await generateText({
            model: openai("gpt-4o-mini"),
            prompt: "What is the capital of France?",
            experimental_telemetry: {
              isEnabled: true,
              tracer: tracer!,
            },
          });
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
