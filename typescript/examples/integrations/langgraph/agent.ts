import * as z from "zod";
import { type BaseMessage } from "@langchain/core/messages";

import { isAIMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { StateGraph, START, END } from "@langchain/langgraph";
import { SystemMessage } from "@langchain/core/messages";

import { DeepEvalCallbackHandler } from "deepeval-ts/integrations";

const handler = new DeepEvalCallbackHandler({
  name: "Langgraph Agent",
  tags: ["tag1", "deepeval", "langgraph-agent"],
  metadata: {
    key1: "value1",
  },
});

const model = new ChatOpenAI({
  model: "gpt-5-nano",
  metadata: {
    metric_collection: "test_collection_1",
    prompt: prompt,
  },
});

// Define tools
const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

// Augment the LLM with tools
const toolsByName = {
  [add.name]: add,
  [multiply.name]: multiply,
  [divide.name]: divide,
};
const tools = Object.values(toolsByName);
const modelWithTools = model.bindTools(tools);

const MessagesState = z.object({
  messages: z.array(z.custom<BaseMessage>()).describe("messages"),
  llmCalls: z.number().optional(),
});

async function llmCall(state: z.infer<typeof MessagesState>) {
  return {
    messages: await modelWithTools.invoke([
      new SystemMessage(
        "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
      ),
      ...state.messages,
    ]),
    llmCalls: (state.llmCalls ?? 0) + 1,
  };
}

async function toolNode(state: z.infer<typeof MessagesState>) {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage == null || !isAIMessage(lastMessage)) {
    return { messages: [] };
  }

  const result: ToolMessage[] = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name];

    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  return { messages: result };
}

async function shouldContinue(state: z.infer<typeof MessagesState>) {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage == null || !isAIMessage(lastMessage)) return END;

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  return END;
}

const graph = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile();

// Invoke
(async () => {
  const result = await graph.invoke(
    {
      messages: [new HumanMessage("Add 3 and 4.")],
    },
    {
      callbacks: [handler],
    },
  );

  console.log(`[result.messages]: ${result.messages}`);
})().catch(console.error);
