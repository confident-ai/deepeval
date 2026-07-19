import { Mastra } from "@mastra/core/mastra";
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { Observability } from "@mastra/observability";
import { z } from "zod";

import { DeepEvalExporter } from "../../../src/integrations/mastra";

const getWeather = createTool({
  id: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ weather: z.string() }),
  execute: async ({ city }) => {
    const data: Record<string, string> = {
      "san francisco": "Foggy, 58F",
      tokyo: "Sunny, 72F",
      london: "Rainy, 55F",
    };
    return { weather: data[city.toLowerCase()] ?? `No data for ${city}` };
  },
});

// 2. A Mastra agent that uses the tool.
const weatherAgent = new Agent({
  id: "weather-agent",
  name: "Weather Agent",
  instructions:
    "You answer weather questions. Always use the get_weather tool, then reply in one short sentence.",
  model: "openai/gpt-4o-mini",
  tools: { get_weather: getWeather },
});

// 3. The DeepEval exporter. Reads CONFIDENT_API_KEY from the environment.
const exporter = new DeepEvalExporter({
  name: "weather-demo",
  environment: "development",
  tags: ["mastra", "demo"],
});

// 4. Register it on the Mastra instance's observability config.
const mastra = new Mastra({
  agents: { weatherAgent },
  observability: new Observability({
    configs: {
      deepeval: {
        serviceName: "mastra-deepeval-demo",
        exporters: [exporter],
      },
    },
  }),
});

async function main() {
  const result = await mastra
    .getAgent("weatherAgent")
    .generate("What's the weather in Tokyo?");

  console.log("\n🤖 Agent:", (result as any).text);

  // Mastra delivers span events asynchronously; give them a moment to land,
  // then flush so the trace is posted to Confident before the process exits.
  await new Promise((r) => setTimeout(r, 2000));
  await exporter.flush();

  console.log(
    "\n✅ Trace sent. Open the Observatory to view it: https://app.confident.ai",
  );
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
