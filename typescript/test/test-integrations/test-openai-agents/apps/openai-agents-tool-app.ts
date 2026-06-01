import { Agent, tool } from "@openai/agents";
import { z } from "zod";

const getWeather = tool({
  name: "get_weather",
  description: "Returns the current weather in a city.",
  parameters: z.object({
    city: z.string().describe("The city to get weather for"),
  }),
  execute: async ({ city }) => {
    const weatherData: Record<string, string> = {
      "san francisco": "Foggy, 58°F",
      "new york": "Sunny, 72°F",
      london: "Rainy, 55°F",
      tokyo: "Cloudy, 68°F",
    };
    return (
      weatherData[city.toLowerCase()] ||
      `Weather data not available for ${city}`
    );
  },
});

const calculate = tool({
  name: "calculate",
  description: "Evaluates a mathematical expression.",
  parameters: z.object({
    expression: z.string().describe("The math expression to evaluate"),
  }),
  execute: async ({ expression }) => {
    try {
      const allowed = new Set("0123456789+-*/.() ");
      if ([...expression].every((c) => allowed.has(c))) {
        return `${expression} = ${eval(expression)}`;
      }
      return "Invalid expression";
    } catch {
      return "Error";
    }
  },
});

export const toolAgent = new Agent({
  name: "ToolAgent",
  instructions:
    "You are a helper. Use tools for weather or math. Do not answer from memory.",
  model: "gpt-4o",
  tools: [getWeather, calculate],
});
