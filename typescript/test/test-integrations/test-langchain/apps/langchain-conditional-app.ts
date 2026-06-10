import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { BaseMessage, ToolMessage, AIMessage } from "@langchain/core/messages";

const researchTopic = tool(
  (input) => {
    const topic = input.topic.toLowerCase();
    if (topic.includes("ai"))
      return "AI research shows rapid advancement in large language models and neural networks.";
    if (topic.includes("climate"))
      return "Climate research indicates rising global temperatures and sea levels.";
    if (topic.includes("space"))
      return "Space research reveals new exoplanets in habitable zones.";
    if (topic.includes("quantum"))
      return "Quantum computing achieves new milestone in error correction.";
    return `Research findings for ${input.topic}: General information available.`;
  },
  {
    name: "research_topic",
    description: "Research a topic and return findings.",
    schema: z.object({ topic: z.string() }),
  },
);

const summarizeText = tool(
  (input) => {
    if (input.text.length > 100)
      return `Summary: ${input.text.substring(0, 100)}...`;
    return `Summary: ${input.text}`;
  },
  {
    name: "summarize_text",
    description: "Summarize the given text.",
    schema: z.object({ text: z.string() }),
  },
);

const factCheck = tool(
  (input) => {
    const claim = input.claim.toLowerCase();
    if (
      claim.includes("true") ||
      claim.includes("correct") ||
      claim.includes("round")
    ) {
      return "Fact check: VERIFIED - This claim appears to be accurate.";
    }
    if (claim.includes("false") || claim.includes("wrong")) {
      return "Fact check: FALSE - This claim is inaccurate.";
    }
    return "Fact check: UNVERIFIED - Unable to confirm this claim.";
  },
  {
    name: "fact_check",
    description: "Fact check a claim.",
    schema: z.object({ claim: z.string() }),
  },
);

const researchTools = [researchTopic];
const researchToolsByName = Object.fromEntries(
  researchTools.map((t) => [t.name, t]),
);

const summarizeTools = [summarizeText];
const summarizeToolsByName = Object.fromEntries(
  summarizeTools.map((t) => [t.name, t]),
);

const factCheckTools = [factCheck];
const factCheckToolsByName = Object.fromEntries(
  factCheckTools.map((t) => [t.name, t]),
);

const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });
const llmResearch = llm.bindTools(researchTools);
const llmSummarize = llm.bindTools(summarizeTools);
const llmFactCheck = llm.bindTools(factCheckTools);

const runConditionalChain = async (
  inputs: { messages: BaseMessage[] },
  llmWithTools: any,
  toolsByName: Record<string, any>,
  config?: RunnableConfig,
) => {
  const messages = inputs.messages || [];
  // Fix: Cast invoke to any to bypass Symbol check
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

const researchChain = new RunnableLambda({
  func: async (inputs: { messages: BaseMessage[] }, config?: RunnableConfig) =>
    runConditionalChain(inputs, llmResearch, researchToolsByName, config),
}).withConfig({ runName: "research_chain" });

const summarizeChain = new RunnableLambda({
  func: async (inputs: { messages: BaseMessage[] }, config?: RunnableConfig) =>
    runConditionalChain(inputs, llmSummarize, summarizeToolsByName, config),
}).withConfig({ runName: "summarize_chain" });

const factCheckChain = new RunnableLambda({
  func: async (inputs: { messages: BaseMessage[] }, config?: RunnableConfig) =>
    runConditionalChain(inputs, llmFactCheck, factCheckToolsByName, config),
}).withConfig({ runName: "fact_check_chain" });

const generalChain = new RunnableLambda({
  func: async (
    inputs: { messages: BaseMessage[] },
    config?: RunnableConfig,
  ) => {
    const messages = inputs.messages || [];
    const response = await (llm.invoke as any)(messages, config);
    return { messages: [...messages, response] };
  },
}).withConfig({ runName: "general_chain" });

export const invokeResearch = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await researchChain.invoke(inputs, config);
};

export const invokeSummarize = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await summarizeChain.invoke(inputs, config);
};

export const invokeFactCheck = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await factCheckChain.invoke(inputs, config);
};

export const invokeGeneral = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => {
  return await generalChain.invoke(inputs, config);
};
