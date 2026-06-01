import { instrumentOpenAI } from "deepeval-ts/openai";
import { setTracingContext } from "deepeval-ts/tracing";

import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
instrumentOpenAI(openai);

async function main() {
  const resp = await setTracingContext(
    {
      name: "responses.create (without tools)",
      llmSpanContext: {
        expectedOutput: "A JSON object with a short answer.",
        context: ["ask about oceans"],
      },
    },
    async () => {
      return await openai.responses.create({
        model: "gpt-5-nano",
        input: "How many oceans are there?",
        instructions: "Answer succinctly.",
      });
    },
  );

  console.log(resp.output_text);
}

main();
