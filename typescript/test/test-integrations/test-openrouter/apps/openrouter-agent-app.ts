import { observe } from "@/tracing";
import { MODEL, getOpenRouterClient } from "./openrouter-clients";

const retrieveDocs = observe({
  type: "retriever",
  name: "retrieveDocs",
  fn: async (_query: string) => ["Paris is the capital of France."],
});

/**
 * A retriever span and an LLM span under one agent span — the shape most
 * user apps produce, and the one that exercises parent nesting.
 */
export const invokeAgentApp = observe({
  type: "agent",
  name: "openrouter-agent",
  fn: async (prompt: string) => {
    const client = getOpenRouterClient();

    const docs = (await retrieveDocs(prompt)) as string[];

    return await client.chat.send({
      chatRequest: {
        model: MODEL,
        messages: [
          { role: "system", content: `Context: ${docs.join(" ")}` },
          { role: "user", content: prompt },
        ],
      },
    });
  },
}) as (prompt: string) => Promise<unknown>;
