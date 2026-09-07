import { observe } from "@/tracing";
import { MODEL, getOpenRouterClient } from "./openrouter-clients";

const WEATHER_TOOL = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Look up the current weather for a city.",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
};

function getWeather(city: string) {
  return { city, temperature: 18, unit: "C", condition: "partly cloudy" };
}

/**
 * Two turns: the model asks for the tool, we answer, it summarizes. Produces
 * one LLM span carrying `toolsCalled` and a second carrying the final text.
 *
 * Wrapped in `observe` so both turns share one trace — a bare `chat.send`
 * opens a trace of its own, which would split the loop across two.
 */
export const invokeToolApp = observe({
  type: "agent",
  name: "openrouter-tool-agent",
  fn: async (prompt: string) => {
    const client = getOpenRouterClient();

    const messages: any[] = [{ role: "user", content: prompt }];

    const first: any = await client.chat.send({
      chatRequest: { model: MODEL, messages, tools: [WEATHER_TOOL] } as any,
    });

    const message = first?.choices?.[0]?.message;
    const toolCalls = message?.toolCalls ?? message?.tool_calls ?? [];
    if (toolCalls.length === 0) return first;

    messages.push(message);
    for (const call of toolCalls) {
      const fn = call.function ?? {};
      const args = JSON.parse(fn.arguments || "{}");
      messages.push({
        role: "tool",
        toolCallId: call.id,
        content: JSON.stringify(getWeather(args.city ?? "Paris")),
      });
    }

    return await client.chat.send({
      chatRequest: { model: MODEL, messages, tools: [WEATHER_TOOL] } as any,
    });
  },
}) as (prompt: string) => Promise<unknown>;
