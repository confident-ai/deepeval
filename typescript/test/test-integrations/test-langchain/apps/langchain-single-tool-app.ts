import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { BaseMessage, ToolMessage, AIMessage } from "@langchain/core/messages";

const getWeather = tool(
  (input) => {
    const weatherData: Record<string, string> = {
      "san francisco": "Foggy, 58F",
      "new york": "Sunny, 72F",
      london: "Rainy, 55F",
    };
    return (
      weatherData[input.city.toLowerCase()] ||
      `Weather data not available for ${input.city}`
    );
  },
  {
    name: "get_weather",
    description: "Returns the current weather in a city.",
    schema: z.object({ city: z.string() }),
  },
);

const tools = [getWeather];
const toolsByName = Object.fromEntries(tools.map((t) => [t.name, t]));

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
const llmWithTools = llm.bindTools(tools);

const runToolChain = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  const messages = inputs.messages || [];
  // Use 'as any' on invoke to bypass symbol checks
  const response = (await (llmWithTools.invoke as any)(
    messages,
    config,
  )) as AIMessage;
  const messagesWithResponse = [...messages, response];

  if (response.tool_calls && response.tool_calls.length > 0) {
    for (const toolCall of response.tool_calls) {
      const t = toolsByName[toolCall.name];
      if (t) {
        const result = await t.invoke(toolCall, config);
        messagesWithResponse.push(
          new ToolMessage({
            content:
              typeof result === "string" ? result : JSON.stringify(result),
            tool_call_id: toolCall.id!,
          }),
        );
      }
    }
    const finalResponse = await (llmWithTools.invoke as any)(
      messagesWithResponse,
      config,
    );
    return { messages: [...messagesWithResponse, finalResponse] };
  }
  return { messages: messagesWithResponse };
};

const singleToolChain = new RunnableLambda({ func: runToolChain }).withConfig({
  runName: "single_tool_chain",
});

export const invokeSingleToolApp = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await singleToolChain.invoke(inputs, config);
};
