import { observe, updateCurrentTrace } from "../../src/tracing/tracing";
import OpenAI from "openai";

const randomChoice = (arr: string[]) => {
  return arr[Math.floor(Math.random() * arr.length)];
};

const llmApp = async (query: string) => {
  const openai = new OpenAI();
  const res = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  const apiKey = randomChoice(["api-key-1", "api-key-2"]);
  updateCurrentTrace({
    threadId: "your-thread-id",
    input: query,
    output: res.choices[0].message.content,
    confidentApiKey: apiKey,
  });
  return res.choices[0].message.content;
};

const observedLlmApp = observe({
  fn: llmApp,
});

observedLlmApp("Write me a poem.");
observedLlmApp("Write me a poem.");
observedLlmApp("Write me a poem.");
