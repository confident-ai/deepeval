import { zodResponseFormat } from "openai/helpers/zod";
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
      name: "completions.parse",
      tags: ["example", "openai", "parse"],
      metricCollection: "default",
      metadata: {
        key3: "value3",
      },
      llmSpanContext: {
        prompt: prompt,
        expectedOutput: "A simple parsing of a sentence into an event.",
        context: ["user input: event description"],
        retrievalContext: [],
        expectedTools: [],
      },
    },
    async () => {
      return await openai.chat.completions.parse({
        model: "gpt-5-nano",
        messages: [
          { role: "system", content: "Extract event info from the sentence." },
          {
            role: "user",
            content:
              "Bob and Carol are going to a birthday party next Tuesday.",
          },
        ],
        response_format: zodResponseFormat(CalendarEvent, "calendar_event"),
      });
    },
  );

  console.log(parseResult.choices[0].message);
}

main();
