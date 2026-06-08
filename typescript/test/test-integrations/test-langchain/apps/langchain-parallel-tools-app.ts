import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { BaseMessage, ToolMessage, AIMessage } from "@langchain/core/messages";

const getWeather = tool(
  (input) => {
    const weather: Record<string, string> = {
      tokyo: "Sunny, 72F",
      "new york": "Cloudy, 58F",
      london: "Rainy, 52F",
      paris: "Partly cloudy, 65F",
      sydney: "Clear, 78F",
    };
    return (
      weather[input.city.toLowerCase()] || `No weather data for ${input.city}`
    );
  },
  {
    name: "get_weather",
    description: "Get weather for a city.",
    schema: z.object({ city: z.string() }),
  },
);

const getStockPrice = tool(
  (input) => {
    const prices: Record<string, string> = {
      AAPL: "$178.50",
      GOOGL: "$142.30",
      MSFT: "$378.90",
      TSLA: "$245.60",
      AMZN: "$185.20",
    };
    return prices[input.symbol.toUpperCase()] || `No price for ${input.symbol}`;
  },
  {
    name: "get_stock_price",
    description: "Get stock price for a symbol.",
    schema: z.object({ symbol: z.string() }),
  },
);

const getExchangeRate = tool(
  (input) => {
    const rates: Record<string, number> = {
      USD_EUR: 0.92,
      USD_GBP: 0.79,
      USD_JPY: 149.5,
      EUR_USD: 1.09,
    };
    const key = `${input.from_currency.toUpperCase()}_${input.to_currency.toUpperCase()}`;
    return rates[key]
      ? `1 ${input.from_currency.toUpperCase()} = ${rates[key]} ${input.to_currency.toUpperCase()}`
      : `No rate for ${input.from_currency} to ${input.to_currency}`;
  },
  {
    name: "get_exchange_rate",
    description: "Get exchange rate between currencies.",
    schema: z.object({ from_currency: z.string(), to_currency: z.string() }),
  },
);

const calculate = tool(
  (input) => {
    try {
      const allowed = new Set("0123456789+-*/.() ".split(""));
      if ([...input.expression].every((c) => allowed.has(c)))
        return `${input.expression} = ${eval(input.expression)}`;
      return "Invalid expression";
    } catch {
      return "Calculation error";
    }
  },
  {
    name: "calculate",
    description: "Calculate a math expression.",
    schema: z.object({ expression: z.string() }),
  },
);

const weatherTools = [getWeather];
const weatherToolsByName = Object.fromEntries(
  weatherTools.map((t) => [t.name, t]),
);

const mixedTools = [getWeather, getStockPrice, getExchangeRate, calculate];
const mixedToolsByName = Object.fromEntries(mixedTools.map((t) => [t.name, t]));

const stockTools = [getStockPrice];
const stockToolsByName = Object.fromEntries(stockTools.map((t) => [t.name, t]));

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
const llmWeather = llm.bindTools(weatherTools, { parallel_tool_calls: true });
const llmMixed = llm.bindTools(mixedTools, { parallel_tool_calls: true });
const llmStocks = llm.bindTools(stockTools, { parallel_tool_calls: true });

const runParallelChain = async (
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

const parallelWeatherChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runParallelChain(inputs, llmWeather, weatherToolsByName, config),
}).withConfig({ runName: "parallel_weather_chain" });
const parallelMixedChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runParallelChain(inputs, llmMixed, mixedToolsByName, config),
}).withConfig({ runName: "parallel_mixed_chain" });
const parallelStocksChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runParallelChain(inputs, llmStocks, stockToolsByName, config),
}).withConfig({ runName: "parallel_stocks_chain" });

export const invokeParallelWeather = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await parallelWeatherChain.invoke(inputs, config);
export const invokeParallelMixed = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await parallelMixedChain.invoke(inputs, config);
export const invokeParallelStocks = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await parallelStocksChain.invoke(inputs, config);
