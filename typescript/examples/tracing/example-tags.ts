import { observe, updateCurrentTrace } from "../../src/tracing/tracing";
import OpenAI from "openai";

const llmApp = async (query: string) => {
  updateCurrentTrace({ tags: ["Causal Chit-Chat"] });
  const openai = new OpenAI();
  const res = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  return res.choices[0].message.content;
};

const observedLlmApp = observe({
  type: "agent",
  fn: llmApp,
});

observedLlmApp("Write me a poem.");
