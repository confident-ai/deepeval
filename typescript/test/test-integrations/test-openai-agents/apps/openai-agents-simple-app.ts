import { Agent } from "@openai/agents";

export const simpleAgent = new Agent({
  name: "SimpleAgent",
  instructions:
    "You are a helpful assistant. Answer the user's question concisely. Do not use any tools.",
  model: "gpt-4o",
});
