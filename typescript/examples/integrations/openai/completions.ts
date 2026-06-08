import { instrumentOpenAI } from "deepeval-ts/openai";
import { setTracingContext } from "deepeval-ts/tracing";
import { Prompt } from "deepeval-ts";

import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
instrumentOpenAI(openai);

const prompt = new Prompt({ alias: "asd" });
prompt.pull({ version: "00.00.01" });

async function main() {
  const result = await setTracingContext(
    {
      name: "chat.completions.create (without tools)",
      tags: ["example", "openai"],
      metricCollection: "default",
      metadata: {
        key1: "value1",
      },
      llmSpanContext: {
        prompt: prompt,
        expectedOutput: "A concise explanation.",
        context: ["user asked about bananas"],
        retrievalContext: [],
        expectedTools: [],
      },
    },
    async () => {
      return await openai.chat.completions.create({
        model: "gpt-5-nano",
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "user", content: "Explain photosynthesis in 2 lines." },
        ],
      });
    },
  );
  console.log(result.choices[0].message.content);
}

main();
