"""Strands evals fixture — trace-level setup with a Strands tool that
mutates its own span via ``update_current_span``.

After the OTel POC migration, ``init_evals_strands(...)`` carries
ONLY trace-level kwargs. Per-call agent / LLM / tool metric collections
and ``BaseMetric`` instances are staged at the call site:

    with next_agent_span(metric_collection="agent_v1", metrics=[...]):
        with next_llm_span(metric_collection="llm_v1"):
            invoke_evals_agent(prompt, invoke_func=invoke_func)

The Strands tool ``special_tool`` uses ``update_current_span`` from
inside its body to set its own ``metric_collection`` — exercising the
placeholder push/pop path that flips Strands from "Bad" to "Good" in
the integrations matrix.
"""

import os
from typing import Dict, List, Optional

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from deepeval.integrations.strands import instrument_strands
from deepeval.tracing import update_current_span


_DEFAULT_MODEL_ID = os.environ.get("STRANDS_TEST_MODEL", "gpt-4o-mini")


def _build_openai_model() -> OpenAIModel:
    return OpenAIModel(
        client_args={"api_key": os.environ.get("OPENAI_API_KEY", "")},
        model_id=_DEFAULT_MODEL_ID,
        params={"temperature": 0.0},
    )


@tool
def special_tool(query: str) -> str:
    """A tool used by feature tests.

    Mutates its own span via ``update_current_span(...)`` so the
    placeholder push/pop pattern is exercised end-to-end. With the
    POC migration this lands on ``confident.span.metric_collection``
    of THIS tool span (no longer a no-op as it was under the old
    ``is_test_mode`` path)."""
    update_current_span(metric_collection="special_tool_v1")
    return f"Processed: {query}"


def init_evals_strands(
    name: str = "strands-evals-test",
    tags: List[str] = None,
    metadata: Dict = None,
    thread_id: str = None,
    user_id: str = None,
    metric_collection: Optional[str] = None,
):
    """Wire deepeval OTel pipeline + a Strands agent with one
    ``update_current_span``-using tool. Trace-only kwargs."""
    instrument_strands(
        name=name,
        tags=tags or ["strands", "evals"],
        metadata=metadata or {"test_type": "evals"},
        thread_id=thread_id,
        user_id=user_id,
        metric_collection=metric_collection,
    )

    agent = Agent(model=_build_openai_model(), tools=[special_tool])

    def invoke(payload: dict):
        user_message = payload.get("prompt", "")
        instruction = "You are a helpful assistant. Be concise. "
        result = agent(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    async def ainvoke(payload: dict):
        user_message = payload.get("prompt", "")
        instruction = "You are a helpful assistant. Be concise. "
        result = await agent.invoke_async(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    invoke.ainvoke = ainvoke
    return invoke


def invoke_evals_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_evals_strands()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_evals_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_evals_strands()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
