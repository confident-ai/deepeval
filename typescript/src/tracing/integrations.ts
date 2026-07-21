export const Integration = {
  LANGCHAIN: "LangChain",
  CREW_AI: "CrewAI",
  LLAMA_INDEX: "LlamaIndex",
  OPENAI_AGENTS: "OpenAI Agents",
  OPEN_AI: "OpenAI",
  ANTHROPIC: "Anthropic",
  PYDANTIC_AI: "PydanticAI",
  GOOGLE_ADK: "Google ADK",
  OPEN_INFERENCE: "OpenInference",
  AGENTCORE: "AgentCore",
  AI_SDK: "AI SDK",
  MASTRA: "Mastra",
} as const;
export type Integration = (typeof Integration)[keyof typeof Integration];

export const Provider = {
  OPEN_AI: "OpenAI",
  ANTHROPIC: "Anthropic",
  GEMINI: "Gemini",
  X_AI: "XAI",
  DEEP_SEEK: "DeepSeek",
  MISTRAL: "Mistral",
  PERPLEXITY: "Perplexity",
  BEDROCK: "Bedrock",
  VERTEX_AI: "VertexAI",
  AZURE: "Azure",
  OPEN_ROUTER: "OpenRouter",
  PORTKEY: "Portkey",
  TRUE_FOUNDRY: "TrueFoundry",
  MOONSHOT: "Moonshot",
} as const;
export type Provider = (typeof Provider)[keyof typeof Provider];
