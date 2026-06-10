import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { BaseMessage } from "@langchain/core/messages";

const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
});

const runSimpleChain = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  const response = await (llm.invoke as any)(inputs.messages, config);
  return { messages: [...inputs.messages, response] };
};

const simpleChain = new RunnableLambda({
  func: runSimpleChain,
}).withConfig({ runName: "simple_chain" });

export const invokeSimpleApp = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await simpleChain.invoke(inputs, config);
};
