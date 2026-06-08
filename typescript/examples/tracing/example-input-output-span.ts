import OpenAI from "openai";
import { observe, updateCurrentSpan } from "../../src/tracing/tracing";

const openai = new OpenAI();

const llmApp = async (query: string) => {
  const res = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  updateCurrentSpan({ input: "test", output: "test" });
  return res.choices[0].message.content;
};

const observedLlmApp = observe({
  fn: llmApp,
});

observedLlmApp("Write me a poem.");
