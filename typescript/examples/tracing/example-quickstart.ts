import OpenAI from "openai";
import { observe } from "../../src/tracing/tracing";

// const llmApp = async (query: string) => {
//   const openai = new OpenAI();
//   const res = await openai.chat.completions.create({
//     model: "gpt-4o",
//     messages: [{ role: "user", content: query }],
//   });
//   return res.choices[0].message.content;
// };

// const observedLlmApp = observe({
//   fn: llmApp,
// });

// // Call app to send trace to Confident AI
// observedLlmApp("Write me a poem.");

///////////////////////////////////

const llmApp = async (query: string) => {
  const openai = new OpenAI();
  const res = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: query }],
  });
  return res.choices[0].message.content;
};

const observedLlmApp = observe({
  metricCollection: "My Metrics",
  fn: llmApp,
});

// Call app to send trace to Confident AI
observedLlmApp("Write me a poem.");
