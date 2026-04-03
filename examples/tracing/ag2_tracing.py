"""
AG2 Multi-Agent Tracing with DeepEval

Demonstrates how to capture traces from AG2 agent conversations
for evaluation using DeepEval metrics.

Requirements:
    pip install deepeval "ag2[openai]>=0.11.4,<1.0"

Usage:
    export OPENAI_API_KEY=your-key
    python ag2_tracing.py
"""

import os

from deepeval.integrations.ag2 import instrument_ag2

# Instrument AG2 before creating agents
instrument_ag2()

from autogen import AssistantAgent, UserProxyAgent, LLMConfig  # noqa: E402

llm_config = LLMConfig(
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
)

assistant = AssistantAgent(
    name="Assistant",
    system_message="You are a helpful assistant. Answer questions concisely.",
    llm_config=llm_config,
)

executor = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False,
)

# Run conversation - traces are captured automatically
# max_turns=1 means: User sends message, Assistant replies once, done.
executor.run(
    assistant,
    message="What are the three laws of thermodynamics? Explain each briefly.",
    max_turns=1,
).process()

# After conversation, you can evaluate the traces using DeepEval metrics
# See: https://docs.confident-ai.com/docs/metrics-introduction
