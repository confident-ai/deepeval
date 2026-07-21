import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import type { TracingOptions } from "@mastra/core/observability";
import { z } from "zod";
import type { DeepEvalExporter } from "../../../../src/integrations/mastra";
import { buildMastra } from "./mastra-harness";

const WEATHER: Record<string, string> = {
  "san francisco": "Foggy, 58F",
  "new york": "Sunny, 72F",
  london: "Rainy, 55F",
  tokyo: "Cloudy, 68F",
};

const getWeather = createTool({
  id: "get_weather",
  description: "Returns the current weather in a city.",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ weather: z.string() }),
  execute: async ({ city }) => ({
    weather:
      WEATHER[city.toLowerCase()] ?? `Weather data not available for ${city}`,
  }),
});

const weatherAgent = new Agent({
  id: "weather-agent",
  name: "Weather Agent",
  instructions:
    "Use the get_weather tool to answer weather questions. Do not answer from memory.",
  model: "openai/gpt-4o-mini",
  tools: { get_weather: getWeather },
});

export async function runToolApp(exporter: DeepEvalExporter, prompt: string) {
  const mastra = buildMastra(exporter, { agents: { weatherAgent } });
  return await mastra.getAgent("weatherAgent").generate(prompt);
}

export async function runToolAppWithTracing(
  exporter: DeepEvalExporter,
  prompt: string,
  tracingOptions: TracingOptions,
) {
  const mastra = buildMastra(exporter, { agents: { weatherAgent } });
  return await mastra
    .getAgent("weatherAgent")
    .generate(prompt, { tracingOptions });
}
