import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { DeepEvalCallbackHandler } from "deepeval-ts/integrations";
import { RunnableSequence } from "@langchain/core/runnables";

const model = new ChatOpenAI({
  modelName: "gpt-5-nano",
});

const promptTemplate = PromptTemplate.fromTemplate(
  "Tell me a joke about {topic}",
);

const handler = new DeepEvalCallbackHandler({
  name: "Langchain",
  tags: ["tag1", "langchain", "langchain-prompt-template"],
  metadata: {
    key1: "value1",
  },
});

async function generatJoke(topic: string) {
  const chain = RunnableSequence.from([promptTemplate, model]);
  const response = await chain.invoke(
    { topic: topic },
    { callbacks: [handler] },
  );

  console.log(response.content);
}

async function main() {
  await generatJoke("cats");
}

main().catch(console.error);
