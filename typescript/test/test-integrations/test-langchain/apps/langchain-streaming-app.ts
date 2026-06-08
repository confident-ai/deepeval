import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { BaseMessage, ToolMessage, AIMessage } from "@langchain/core/messages";

const getStockPrice = tool(
  (input) => {
    const prices: Record<string, string> = {
      AAPL: "$178.50 (+1.2%)",
      GOOGL: "$142.30 (-0.5%)",
      MSFT: "$378.90 (+0.8%)",
      TSLA: "$245.60 (+2.1%)",
      AMZN: "$185.20 (-0.3%)",
    };
    return (
      prices[input.symbol.toUpperCase()] ||
      `Stock price not available for ${input.symbol}`
    );
  },
  {
    name: "get_stock_price",
    description: "Get the current stock price for a ticker symbol.",
    schema: z.object({ symbol: z.string() }),
  },
);

const getCompanyInfo = tool(
  (input) => {
    const info: Record<string, string> = {
      AAPL: "Apple Inc. - Technology company, Market Cap: $2.8T",
      GOOGL: "Alphabet Inc. - Technology company, Market Cap: $1.8T",
      MSFT: "Microsoft Corporation - Technology company, Market Cap: $2.9T",
      TSLA: "Tesla Inc. - Electric vehicles, Market Cap: $780B",
      AMZN: "Amazon.com Inc. - E-commerce/Cloud, Market Cap: $1.9T",
    };
    return (
      info[input.symbol.toUpperCase()] ||
      `Company info not available for ${input.symbol}`
    );
  },
  {
    name: "get_company_info",
    description: "Get company information for a ticker symbol.",
    schema: z.object({ symbol: z.string() }),
  },
);

const singleTools = [getStockPrice];
const singleToolsByName = Object.fromEntries(
  singleTools.map((t) => [t.name, t]),
);

const multiTools = [getStockPrice, getCompanyInfo];
const multiToolsByName = Object.fromEntries(multiTools.map((t) => [t.name, t]));

const llmStreaming = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
  streaming: true,
});
const llmSingle = llmStreaming.bindTools(singleTools);
const llmMulti = llmStreaming.bindTools(multiTools);

const runStreamingChain = async (
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

const streamingSingleChain = new RunnableLambda({
  func: async (inputs: { messages: BaseMessage[] }, config?: RunnableConfig) =>
    runStreamingChain(inputs, llmSingle, singleToolsByName, config),
}).withConfig({ runName: "streaming_single_chain" });

export const invokeStreamingSingle = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await streamingSingleChain.invoke(inputs, config);
};
