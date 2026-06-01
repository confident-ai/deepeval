"""Canonical integration and provider strings for tracing payloads."""

from enum import Enum


class Integration(str, Enum):
    LANGCHAIN = "LangChain"
    CREW_AI = "CrewAI"
    LLAMA_INDEX = "LlamaIndex"
    OPENAI_AGENTS = "OpenAI Agents"
    OPEN_AI = "OpenAI"
    ANTHROPIC = "Anthropic"
    PYDANTIC_AI = "PydanticAI"
    GOOGLE_ADK = "Google ADK"
    STRANDS = "Strands"
    OTEL = "OpenTelemetry"
    OPEN_INFERENCE = "OpenInference"
    AGENTCORE = "AgentCore"


class Provider(str, Enum):
    OPEN_AI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GEMINI = "Gemini"
    X_AI = "XAI"
    DEEP_SEEK = "DeepSeek"
    MISTRAL = "Mistral"
    PERPLEXITY = "Perplexity"
    BEDROCK = "Bedrock"
    VERTEX_AI = "VertexAI"
    AZURE = "Azure"
    OPEN_ROUTER = "OpenRouter"
    PORTKEY = "Portkey"
    TRUE_FOUNDRY = "TrueFoundry"
    MOONSHOT = "Moonshot"
