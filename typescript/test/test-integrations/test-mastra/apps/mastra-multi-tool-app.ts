import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
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
    weather: WEATHER[city.toLowerCase()] ?? `Weather data not available for ${city}`,
  }),
});

const calculate = createTool({
  id: "calculate",
  description: "Evaluates a simple arithmetic expression.",
  inputSchema: z.object({ expression: z.string() }),
  outputSchema: z.object({ result: z.string() }),
  execute: async ({ expression }) => {
    const allowed = new Set("0123456789+-*/.() ");
    if ([...expression].every((c) => allowed.has(c))) {
      // eslint-disable-next-line no-eval
      return { result: `${expression} = ${eval(expression)}` };
    }
    return { result: "Invalid expression" };
  },
});

const assistantAgent = new Agent({
  id: "assistant-agent",
  name: "Assistant Agent",
  instructions:
    "Use the get_weather tool for weather and the calculate tool for math. Do not answer from memory.",
  model: "openai/gpt-4o-mini",
  tools: { get_weather: getWeather, calculate },
});

export async function runMultiToolApp(
  exporter: DeepEvalExporter,
  prompt: string,
) {
  const mastra = buildMastra(exporter, { agents: { assistantAgent } });
  return await mastra.getAgent("assistantAgent").generate(prompt);
}
