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
    name: "get_country_info",
    description: "Get information about a country",
    parameters: {
      type: "object",
      properties: {
        country: { type: "string" },
      },
      required: ["country"],
    },
    strict: false,
  },
];

async function main() {
  const resp = await setTracingContext(
    {
      name: "responses.create (with tools)",
      llmSpanContext: {
        expectedTools: [
          {
            name: "get_country_info",
            inputParameters: { country: "Japan" },
          },
        ],
        expectedOutput: "Likely a function call to get_country_info.",
        context: [],
        retrievalContext: [],
        metricCollection: "default",
        prompt: prompt,
      },
    },
    async () => {
      return await openai.responses.create({
        model: "gpt-5-nano",
        input: "Tell me about Japan.",
        instructions: "Provide relevant details about the country.",
        tools: tools,
      });
    },
  );

  console.log(resp.output);
}

main();
