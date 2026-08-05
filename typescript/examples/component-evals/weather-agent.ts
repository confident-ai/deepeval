/**
 * Shared app for the component-eval examples: a Mastra agent with one tool.
 *
 * SEAM 1 lives here — inside a tool body your code IS the span, so you can reach
 * it directly. See `evals-iterator.ts` / `matcher.test.ts` for SEAM 2.
 */
import { Mastra } from "@mastra/core/mastra";
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { Observability } from "@mastra/observability";
import { z } from "zod";

import { metrics } from "deepeval";
import { ToolCall } from "deepeval/testCase";
import { DeepEvalExporter } from "deepeval/integrations/mastra";
import { updateCurrentSpan } from "deepeval/tracing";

const getWeather = createTool({
  id: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ weather: z.string() }),
  execute: async ({ city }) => {
    // Attach the metric to THIS tool span, plus the per-call data the
    // metric needs. Nothing can infer `expectedTools` for you.
    updateCurrentSpan({
      metrics: [new metrics.ToolCorrectnessMetric()],
      toolsCalled: [
        new ToolCall({ name: "get_weather", inputParameters: { city } }),
      ],
      expectedTools: [new ToolCall({ name: "get_weather" })],
    });

    return {
      weather: city.toLowerCase() === "tokyo" ? "Sunny, 72F" : "Rainy, 55F",
    };
  },
});

const weatherAgent = new Agent({
  id: "weather-agent",
  name: "Weather Agent",
  instructions:
    "Answer weather questions. Always use get_weather, then reply in one sentence.",
  model: "openai/gpt-4o-mini",
  tools: { get_weather: getWeather },
});

export const mastra = new Mastra({
  agents: { weatherAgent },
  observability: new Observability({
    configs: {
      deepeval: {
        serviceName: "component-evals-example",
        exporters: [new DeepEvalExporter({ name: "component-evals-example" })],
      },
    },
  }),
});

export const ask = (question: string) =>
  mastra.getAgent("weatherAgent").generate(question);
