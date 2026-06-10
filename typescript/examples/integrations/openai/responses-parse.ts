import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

import { instrumentOpenAI } from "deepeval-ts/openai";
import { setTracingContext } from "deepeval-ts/tracing";
import { Prompt } from "deepeval-ts";

import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
instrumentOpenAI(openai);

const prompt = new Prompt({ alias: "asd" });
prompt.pull({ version: "00.00.01" });

const CalendarEvent = z.object({
  name: z.string(),
  date: z.string(),
  participants: z.array(z.string()),
});

async function main() {
  const parseResult = await setTracingContext(
    {
      name: "responses.parse",
      tags: ["example", "openai", "parse"],
      metricCollection: "default",
      metadata: {
        key2: "value2",
      },
      llmSpanContext: {
        prompt: prompt,
        expectedOutput: "A simple parsing of a sentence.",
        context: ["user asked to parse a sentence"],
        retrievalContext: [],
        expectedTools: [],
      },
    },
    async () => {
      return await openai.responses.parse({
        model: "gpt-5-nano",
        input: [
          { role: "system", content: "Extract the event information." },
          {
            role: "user",
            content: "Alice and Bob are going to a science fair on Friday.",
          },
        ],
        text: {
          format: zodTextFormat(CalendarEvent, "event"),
        },
      });
    },
  );
  console.log(parseResult.output);
}

main();
