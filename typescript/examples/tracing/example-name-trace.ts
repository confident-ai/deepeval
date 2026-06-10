import OpenAI from "openai";
import { observe, updateCurrentTrace } from "../../src/tracing/tracing";

const openai = new OpenAI();

const llmApp = async (query: string) => {
  updateCurrentTrace({ name: "Call LLM" });
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  return response.choices[0].message.content;
};

const observedLlmApp = observe({
  fn: llmApp,
});

observedLlmApp("Write me a poem.");
