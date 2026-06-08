import OpenAI from "openai";
import { observe, updateCurrentSpan } from "../../src/tracing/tracing";

const openai = new OpenAI();

const llmApp = async (query: string) => {
  updateCurrentSpan({ name: "Call LLM" });
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  return response.choices[0].message.content;
};

const observedLlmApp = observe({
  fn: llmApp,
  name: "not_llm_app",
});

observedLlmApp("Write me a poem.");
