"""LLM commands, plus the embedding commands of providers that serve both.

Import order is the order `deepeval --help` lists them in.
"""

from deepeval.cli.providers.llms import (  # noqa: F401
    openai,
    azure,
    anthropic,
    bedrock,
    ollama,
    local_model,
    grok,
    moonshot,
    deepseek,
    local_embeddings,
    gemini,
    litellm,
    portkey,
    openrouter,
)
