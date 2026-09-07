import { MODEL, getOpenAIRouteClient } from "./openrouter-clients";

/**
 * The other way into OpenRouter: the OpenAI SDK pointed at its base URL.
 * `instrumentOpenAI` recognizes the host, so the span still carries
 * OpenRouter's cost and generation detail.
 */
export async function invokeOpenAIRouteApp(prompt: string) {
  return await getOpenAIRouteClient().chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: prompt }],
  });
}
