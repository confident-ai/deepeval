import { observe, updateCurrentTrace } from "../../src/tracing/tracing";
import OpenAI from "openai";

const llmApp = async (query: string) => {
  const openai = new OpenAI();
  const res = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  updateCurrentTrace({
    threadId: "your-thread-id",
    input: query,
    output: res.choices[0].message.content,
  });
  return res.choices[0].message.content;
};

const observedLlmApp = observe({
  fn: llmApp,
});

observedLlmApp("Write me a poem.");
