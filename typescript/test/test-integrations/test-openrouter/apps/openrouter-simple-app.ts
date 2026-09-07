import { MODEL, getOpenRouterClient } from "./openrouter-clients";

export async function invokeSimpleApp(prompt: string) {
  return await getOpenRouterClient().chat.send({
    chatRequest: {
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
    },
  });
}
