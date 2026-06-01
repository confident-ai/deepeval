import { BaseRetriever, BaseRetrieverInput } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";
import {
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { RunnableLambda, RunnableConfig } from "@langchain/core/runnables";
import { CallbackManagerForRetrieverRun } from "@langchain/core/callbacks/manager";

class DeterministicRetriever extends BaseRetriever {
  lc_namespace = ["langchain", "retrievers"];
  documents: Record<string, Document[]> = {
    python: [
      new Document({
        pageContent:
          "Python is a high-level programming language known for its simplicity.",
        metadata: { source: "doc1" },
      }),
      new Document({
        pageContent:
          "Python supports multiple programming paradigms including procedural and OOP.",
        metadata: { source: "doc2" },
      }),
    ],
    langchain: [
      new Document({
        pageContent:
          "LangChain is a framework for developing applications powered by language models.",
        metadata: { source: "doc3" },
      }),
      new Document({
        pageContent:
          "LangChain provides tools for chaining LLM calls and integrating with external data.",
        metadata: { source: "doc4" },
      }),
    ],
    default: [
      new Document({
        pageContent:
          "This is a general document about AI and machine learning.",
        metadata: { source: "doc5" },
      }),
      new Document({
        pageContent:
          "Machine learning enables computers to learn from data without explicit programming.",
        metadata: { source: "doc6" },
      }),
    ],
  };

  constructor(
    fields?: BaseRetrieverInput & { metadata?: Record<string, any> },
  ) {
    super(fields);
    if (fields?.metadata) this.metadata = fields.metadata;
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    const queryLower = query.toLowerCase();
    if (queryLower.includes("python")) return this.documents.python;
    if (queryLower.includes("langchain")) return this.documents.langchain;
    return this.documents.default;
  }
}

const retriever = new DeterministicRetriever();
const retrieverWithMetricCollection = new DeterministicRetriever({
  metadata: { metric_collection: "retriever_quality" },
});
const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

const runRagChain = async (
  inputs: { messages: BaseMessage[] },
  currentRetriever: any,
  config?: RunnableConfig,
) => {
  const messages = inputs.messages || [];
  const query =
    messages.length > 0 ? messages[messages.length - 1].content.toString() : "";
  const docs = await (currentRetriever.invoke as any)(query, config);
  const context = docs.map((d: any) => d.pageContent).join("\n\n");

  const augmentedMessages = [
    {
      role: "system",
      content:
        "You are a helpful assistant. Answer the user's question based ONLY on the provided context. Be concise and factual.",
    },
    ...messages.slice(0, -1),
    {
      role: "user",
      content: `Context:\n${context}\n\nAnswer based on the context above.\nQuery: ${query}`,
    },
  ];

  const response = await (llm.invoke as any)(augmentedMessages, config);
  return { messages: [...messages, response], context, source_documents: docs };
};

const ragChain = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runRagChain(inputs, retriever, config),
}).withConfig({ runName: "rag_chain" });
const ragChainWithMetricCollection = new RunnableLambda({
  func: (inputs: any, config?: RunnableConfig) =>
    runRagChain(inputs, retrieverWithMetricCollection, config),
}).withConfig({ runName: "rag_chain" });

export const invokeRagApp = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await ragChain.invoke(inputs, config);
export const invokeRagAppWithMetricCollection = async (
  inputs: { messages: BaseMessage[] },
  config?: RunnableConfig,
) => await ragChainWithMetricCollection.invoke(inputs, config);
