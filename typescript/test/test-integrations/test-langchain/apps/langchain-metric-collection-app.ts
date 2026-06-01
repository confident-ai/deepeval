import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import {
  BaseMessage,
  AIMessage,
  ToolMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { Prompt } from "../../../../src/prompt";

const testPrompt = new Prompt({ alias: "langchain-metric-collection-test" });
testPrompt.version = "01.00.00";
testPrompt.label = "production";
testPrompt.hash = "bab04ec";

const calculate = tool(
  (input) => {
    try {
      const allowed = new Set("0123456789+-*/.(). ".split(""));
      if (![...input.expression].every((c) => allowed.has(c)))
        return "Error: Invalid characters";
      return String(eval(input.expression));
    } catch (e: any) {
      return `Error: ${e.message}`;
    }
  },
  {
    name: "calculate",
    description: "Evaluates a math expression.",
    schema: z.object({ expression: z.string() }),
  },
);

const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
  metadata: {
    prompt: testPrompt,
    metricCollection: "llm-span-evals",
  },
});

const llmWithTools = llm.bindTools([calculate]);

const runMetricCollectionApp = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  const systemMsg = new SystemMessage("You are a calculator assistant.");
  const allMessages = [systemMsg, ...(inputs.messages || [])];

  for (let i = 0; i < 2; i++) {
    const response = (await (llmWithTools.invoke as any)(
      allMessages,
      config,
    )) as AIMessage;
    allMessages.push(response);

    if (!response.tool_calls || response.tool_calls.length === 0) break;

    for (const toolCall of response.tool_calls) {
      if (toolCall.name === "calculate") {
        const result = await calculate.invoke(toolCall, config);
        allMessages.push(
          new ToolMessage({
            content:
              typeof result === "string" ? result : JSON.stringify(result),
            tool_call_id: toolCall.id!,
          }),
        );
      }
    }
  }
  return { messages: allMessages.slice(1) };
};

const metricCollectionChain = new RunnableLambda({
  func: runMetricCollectionApp,
}).withConfig({ runName: "metric_collection_chain" });

export const invokeMetricCollectionApp = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await metricCollectionChain.invoke(inputs, config);
};
