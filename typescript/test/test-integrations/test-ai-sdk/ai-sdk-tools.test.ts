import { generateText, tool, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import * as path from "path";

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

describe("Vercel AI SDK Tool Calling Tests", () => {
  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture full tool execution loop during generateText", async () => {
    const jsonPath = path.join(FIXTURES_DIR, "expected_ai_sdk_tools.json");

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "vercel-tools-thread",
          name: "Vercel Tool Gen Trace",
          tags: ["vercel", "tool-call"],
        },
        async () => {
          await generateText({
            model: openai("gpt-4o-mini"),
            prompt: "What is the weather in San Francisco? Answer in celsius.",

            tools: {
              get_weather: tool({
                description: "Get the weather in a location",
                inputSchema: z.object({
                  location: z
                    .string()
                    .describe("The city name, e.g., 'San Francisco'"),
                  unit: z.enum(["c", "f"]).optional(),
                }),
                execute: async ({ location, unit }) => {
                  const temp = unit === "f" ? 65 : 18;
                  return {
                    location,
                    temperature: temp,
                    unit: unit || "c",
                    condition: "partly cloudy",
                  };
                },
              }),
            },
            stopWhen: stepCountIs(2),
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
