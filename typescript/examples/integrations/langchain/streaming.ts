import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { DeepEvalCallbackHandler } from "deepeval-ts/integrations";

const model = new ChatOpenAI({
  modelName: "gpt-5-nano",
});

const promptTemplate = PromptTemplate.fromTemplate(
  "Tell me a joke about {topic}",
);

const handler = new DeepEvalCallbackHandler({
  name: "DeepEvalCallbackHandler",
  tags: ["tag1", "tag2"],
  metadata: {
    key1: "value1",
  },
});

async function generateStream(topic: string) {
  const chain = promptTemplate.pipe(model);

  const stream = await chain.stream({ topic: topic }, { callbacks: [handler] });

  for await (const chunk of stream) {
    console.log(chunk?.content);
  }
}

async function main() {
  await generateStream("bears");
}

main().catch(console.error);
