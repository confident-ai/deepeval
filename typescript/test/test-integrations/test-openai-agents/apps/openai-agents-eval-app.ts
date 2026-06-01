import { Agent } from "@openai/agents";

export const evalAgent = new Agent({
  name: "EvalAgent",
  instructions: "You are a helpful assistant.",
  model: "gpt-4o",
});
