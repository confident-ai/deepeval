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
      tokyo: "Cloudy, 68F",
      paris: "Partly cloudy, 62F",
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

const getPopulation = tool(
  (input) => {
    const populationData: Record<string, string> = {
      "san francisco": "874,000",
      "new york": "8,336,000",
      london: "8,982,000",
      tokyo: "13,960,000",
      paris: "2,161,000",
    };
    return (
      populationData[input.city.toLowerCase()] ||
      `Population data not available for ${input.city}`
    );
  },
  {
    name: "get_population",
    description: "Returns the population of a city.",
    schema: z.object({ city: z.string() }),
  },
);

const getTimezone = tool(
  (input) => {
    const timezoneData: Record<string, string> = {
      "san francisco": "PST (UTC-8)",
      "new york": "EST (UTC-5)",
      london: "GMT (UTC+0)",
      tokyo: "JST (UTC+9)",
      paris: "CET (UTC+1)",
    };
    return (
      timezoneData[input.city.toLowerCase()] ||
      `Timezone data not available for ${input.city}`
    );
  },
  {
    name: "get_timezone",
    description: "Returns the timezone of a city.",
    schema: z.object({ city: z.string() }),
  },
);

const calculate = tool(
  (input) => {
    try {
      const allowedChars = new Set("0123456789+-*/.() ".split(""));
      if ([...input.expression].every((c) => allowedChars.has(c))) {
        return `${input.expression} = ${eval(input.expression)}`;
      }
      return "Invalid expression";
    } catch (e: any) {
      return `Error: ${e.message}`;
    }
  },
  {
    name: "calculate",
    description: "Evaluates a mathematical expression and returns the result.",
    schema: z.object({ expression: z.string() }),
  },
);

const cityInfoTools = [getWeather, getPopulation, getTimezone];
const cityInfoToolsByName = Object.fromEntries(
  cityInfoTools.map((t) => [t.name, t]),
);

const mixedTools = [getWeather, calculate];
const mixedToolsByName = Object.fromEntries(mixedTools.map((t) => [t.name, t]));

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
const llmCityInfo = llm.bindTools(cityInfoTools);
const llmMixed = llm.bindTools(mixedTools);

const runMultiToolChain = async (
  inputs: { messages: BaseMessage[] },
  llmWithTools: any,
  toolsByName: Record<string, any>,
  config?: RunnableConfig,
) => {
  const messages = inputs.messages || [];
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

const cityInfoChain = new RunnableLambda({
  func: async (inputs: { messages: BaseMessage[] }, config?: RunnableConfig) =>
    runMultiToolChain(inputs, llmCityInfo, cityInfoToolsByName, config),
}).withConfig({ runName: "city_info_chain" });

const mixedToolsChain = new RunnableLambda({
  func: async (inputs: { messages: BaseMessage[] }, config?: RunnableConfig) =>
    runMultiToolChain(inputs, llmMixed, mixedToolsByName, config),
}).withConfig({ runName: "mixed_tools_chain" });

export const invokeCityInfo = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await cityInfoChain.invoke(inputs, config);
};

export const invokeMixedTools = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await mixedToolsChain.invoke(inputs, config);
};
