import OpenAI from "openai";
import { observe, traceManager } from "../../src/tracing/tracing";

traceManager.configure({ environment: "production" });
const openai = new OpenAI();

const llmApp = async (query: string) => {
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  return response.choices[0].message.content;
};

const observedLlmApp = observe({
  type: "llm",
  model: "gpt-4o",
  fn: llmApp,
});

observedLlmApp("Write me a poem.");
