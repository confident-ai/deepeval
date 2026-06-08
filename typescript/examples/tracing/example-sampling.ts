import { observe, traceManager } from "../../src/tracing/tracing";
import OpenAI from "openai";

traceManager.configure({ samplingRate: 0.5 });

const openai = new OpenAI();

const llmApp = async (query: string) => {
  const res = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  return res.choices[0].message.content;
};

const observedLlmApp = observe({ fn: llmApp });

for (let i = 0; i < 10; i++) {
  observedLlmApp("Write me a poem.");
}
