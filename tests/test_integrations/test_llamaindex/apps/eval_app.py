"""
DeepEval Features LlamaIndex App
Based on official DeepEval + LlamaIndex documentation using FunctionAgent.
"""

import nest_asyncio
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


def get_evals_agent():
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    return FunctionAgent(
        tools=[multiply],
        llm=llm,
        system_prompt="You are a helpful assistant that can perform calculations. You MUST use the tools provided.",
    )
