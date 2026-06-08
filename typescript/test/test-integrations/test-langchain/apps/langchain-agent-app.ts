import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { BaseMessage, ToolMessage, AIMessage } from "@langchain/core/messages";

const searchWeb = tool(
  (input) => {
    const results: Record<string, string> = {
      "weather san francisco":
        "San Francisco weather: Foggy, 58F, humidity 75%",
      "population tokyo":
        "Tokyo population: approximately 13.96 million people",
      "stock price apple": "Apple (AAPL) stock: $178.50, up 1.2%",
      "exchange rate usd eur": "USD to EUR: 1 USD = 0.92 EUR",
    };
    const query = input.query.toLowerCase();
    for (const [key, value] of Object.entries(results)) {
      if (key.split(" ").every((word) => query.includes(word))) return value;
    }
    return `Search results for '${input.query}': No specific data found.`;
  },
  {
    name: "search_web",
    description: "Search the web for information.",
    schema: z.object({ query: z.string() }),
  },
);

const calculator = tool(
  (input) => {
    try {
      const allowed = new Set("0123456789+-*/.() ".split(""));
      if ([...input.expression].every((c) => allowed.has(c)))
        return `Calculation: ${input.expression} = ${eval(input.expression)}`;
      return "Invalid expression";
    } catch (e: any) {
      return `Calculation error: ${e.message}`;
    }
  },
  {
    name: "calculator",
    description: "Perform mathematical calculations.",
    schema: z.object({ expression: z.string() }),
  },
);

const getCurrentTime = tool(() => "Current time: 2024-01-15 10:30:00 UTC", {
  name: "get_current_time",
  description: "Get the current time.",
  schema: z.object({}),
});

const simpleTools = [searchWeb];
const simpleToolsByName = Object.fromEntries(
  simpleTools.map((t) => [t.name, t]),
);

const multiStepTools = [searchWeb, calculator];
const multiStepToolsByName = Object.fromEntries(
  multiStepTools.map((t) => [t.name, t]),
);

const complexTools = [searchWeb, calculator, getCurrentTime];
const complexToolsByName = Object.fromEntries(
  complexTools.map((t) => [t.name, t]),
);

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
const llmSimple = llm.bindTools(simpleTools);
const llmMultiStep = llm.bindTools(multiStepTools);
const llmComplex = llm.bindTools(complexTools);

const runAgentLoop = async (
  inputs: { messages: BaseMessage[] },
  llmWithTools: any,
  toolsByName: Record<string, any>,
  config?: RunnableConfig,
  maxIterations = 5,
) => {
  const allMessages = [...(inputs.messages || [])];

  for (let i = 0; i < maxIterations; i++) {
    const response = (await (llmWithTools.invoke as any)(
      allMessages,
      config,
    )) as AIMessage;
    allMessages.push(response);

    if (!response.tool_calls || response.tool_calls.length === 0) break;

    for (const toolCall of response.tool_calls) {
      const t = toolsByName[toolCall.name];
      if (t) {
        const result = await t.invoke(toolCall, config);
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
  return { messages: allMessages };
};

const simpleAgentChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runAgentLoop(inputs, llmSimple, simpleToolsByName, config),
}).withConfig({ runName: "simple_agent" });
const multiStepAgentChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runAgentLoop(inputs, llmMultiStep, multiStepToolsByName, config),
}).withConfig({ runName: "multi_step_agent" });
const complexAgentChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runAgentLoop(inputs, llmComplex, complexToolsByName, config),
}).withConfig({ runName: "complex_agent" });

export const invokeSimpleAgent = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await simpleAgentChain.invoke(inputs, config);
export const invokeMultiStepAgent = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await multiStepAgentChain.invoke(inputs, config);
export const invokeComplexAgent = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await complexAgentChain.invoke(inputs, config);
