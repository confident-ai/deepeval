import { observe } from "../../src/tracing/tracing";

const ragPipeline = (query: string): string => {
  console.log(query);
  return "";
};

const generateResponse = (prompt: string): string => {
  console.log(prompt);
  return "";
};

const retrieveDocuments = (query: string): string[] => {
  console.log(query);
  return [];
};

const webSearch = (query: string): string => {
  console.log(query);
  return "";
};

const supervisorAgent = (query: string): string => {
  console.log(query);
  return "";
};

const observedRagPipeline = observe({
  name: "RAG Pipeline",
  fn: ragPipeline,
});

const observedGenerateResponse = observe({
  type: "llm",
  model: "gpt-4",
  fn: generateResponse,
});

const observedRetrieveDocuments = observe({
  type: "retriever",
  embedder: "text-embedding-ada-002",
  fn: retrieveDocuments,
});

const observedWebSearch = observe({
  type: "tool",
  fn: webSearch,
});

const observedSupervisorAgent = observe({
  type: "agent",
  availableTools: ["search", "calculator"],
  agentHandoffs: ["research_agent", "math_agent"],
  fn: supervisorAgent,
});

observedRagPipeline("What is the capital of France?");
observedGenerateResponse("What is the capital of France?");
observedRetrieveDocuments("What is the capital of France?");
observedWebSearch("What is the capital of France?");
observedSupervisorAgent("What is the capital of France?");
