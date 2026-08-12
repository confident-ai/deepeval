// Canonical integration and provider strings for tracing payloads. Port of
// `deepeval/tracing/integrations.py`; the values are shared with Confident AI
// and with telemetry, so they must match Python exactly.

export enum Integration {
  LANGCHAIN = "LangChain",
  CREW_AI = "CrewAI",
  LLAMA_INDEX = "LlamaIndex",
  OPENAI_AGENTS = "OpenAI Agents",
  OPEN_AI = "OpenAI",
  ANTHROPIC = "Anthropic",
  PYDANTIC_AI = "PydanticAI",
  GOOGLE_ADK = "Google ADK",
  STRANDS = "Strands",
  OTEL = "OpenTelemetry",
  OPEN_INFERENCE = "OpenInference",
  AGENTCORE = "AgentCore",
  // Only this SDK ships these two, but the vocabulary is shared.
  AI_SDK = "AI SDK",
  MASTRA = "Mastra",
}
