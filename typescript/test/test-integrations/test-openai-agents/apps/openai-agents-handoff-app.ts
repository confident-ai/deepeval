import { Agent } from "@openai/agents";

const spanishAgent = new Agent({
  name: "SpanishAgent",
  instructions: "You speak Spanish. Answer 'Hola' to everything.",
  model: "gpt-4o",
});

const englishAgent = new Agent({
  name: "EnglishAgent",
  instructions: "You speak English. Answer 'Hello' to everything.",
  model: "gpt-4o",
});

export const triageAgent = new Agent({
  name: "TriageAgent",
  instructions:
    "If input is Spanish, handoff to SpanishAgent. Else EnglishAgent.",
  model: "gpt-4o",
  handoffs: [spanishAgent, englishAgent],
});
