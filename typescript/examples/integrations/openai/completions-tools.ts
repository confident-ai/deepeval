import { instrumentOpenAI } from "deepeval-ts/openai";
import { setTracingContext } from "deepeval-ts/tracing";
import { Prompt } from "deepeval-ts";

import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
instrumentOpenAI(openai);

const prompt = new Prompt({ alias: "asd" });
prompt.pull({ version: "00.00.01" });

const tools = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get current weather for a city",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string" },
        },
        required: ["city"],
      },
    },
  },
];

async function main() {
  const resp = await setTracingContext(
    {
      name: "chat.completions.create (with tools)",
      llmSpanContext: {
        expectedTools: [
          {
            name: "get_weather",
            inputParameters: { city: "SF" },
          },
        ],
        expectedOutput: "Likely a function call to get_weather.",
        context: [],
        retrievalContext: [],
        metricCollection: "default",
        prompt: prompt,
      },
    },
    async () => {
      return await openai.chat.completions.create({
        model: "gpt-5-nano",
        messages: [
          { role: "user", content: "What's the weather in San Francisco?" },
        ],
        tools: tools,
      });
    },
  );

  console.log(resp.choices[0].message.tool_calls);
}

main();
