import os
import json

import pytest
import asyncio
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
from deepeval.tracing import trace
from deepeval.tracing.trace_context import LlmSpanContext, AgentSpanContext

from deepeval.integrations.llama_index import instrument_llama_index
from deepeval.prompt.prompt import Prompt

# import llama_index.core.instrumentation as instrument
# instrument_llama_index(instrument.get_dispatcher())


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="test_collection_1",
)

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"


async def llm_app(input: str):
    agent_span_context = AgentSpanContext(
        metric_collection="test_collection_1",
    )
    llm_span_context = LlmSpanContext(
        metric_collection="test_collection_1",
        prompt=prompt,
    )
    with trace(
        agent_span_context=agent_span_context, llm_span_context=llm_span_context
    ):
        await agent.run(input)


################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "llama_index.json")


# @generate_trace_json(json_path)
@assert_trace_json(json_path)
async def test_json_schema():
    await llm_app("What is 3 * 12?")
