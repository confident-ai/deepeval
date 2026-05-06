"""test_agentcore_agent.py — pytest analog of ``test_pydantic_agent.py``
for the AgentCore × Strands integration.

Run with::

    deepeval test run test_agentcore_agent.py

Same shape as ``test_pydantic_agent.py``: pull a dataset by alias,
instrument the agent at import time, wrap the agent invocation in
``next_agent_span(metrics=[...])`` for a span-level metric, and pass
the trace-level metric to ``assert_test``. The deepeval pytest plugin
wraps each test in an eval session so the agent's OTel spans route
through REST (``ContextAwareSpanProcessor`` flips routing because
``trace_manager.is_evaluating`` is True under ``deepeval test run``).

Requirements:
  - ``CONFIDENT_API_KEY`` in env (or ``deepeval login``)
  - ``OPENAI_API_KEY`` in env (the AnswerRelevancy scorer)
  - AWS credentials (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    optionally ``AWS_REGION``) — Strands invokes Bedrock under the hood.
  - ``pip install bedrock-agentcore strands-agents pytest``
"""

import uuid
from pathlib import Path

import pytest
from strands import Agent

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.agentcore import instrument_agentcore
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing.context import next_agent_span


RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


# Wire the deepeval OTel pipeline at import time. Trace-level kwargs
# only — span-level fields belong on per-call ``with next_*_span(...)``
# blocks below.
instrument_agentcore(
    name="agentcore-pytest-agent",
    tags=["agentcore", "pytest"],
    metadata={"run_id": RUN_ID, "script": Path(__file__).stem},
)


# Module-scope agent so spans share the same instrumented TracerProvider.
agent = Agent(
    model="amazon.nova-lite-v1:0",
    system_prompt="Be concise. Reply with one short sentence.",
)


async def run_agent(prompt: str) -> str:
    """Wrap the Strands invocation in ``next_agent_span(metrics=[...])``
    so the AnswerRelevancyMetric attaches to the agent span via the
    ``stash_pending_metrics`` overlay (carried across OTel transport
    into ``ConfidentSpanExporter``). Mirrors the ``run_agent`` in
    ``test_pydantic_agent.py``.
    """
    with next_agent_span(metrics=[AnswerRelevancyMetric(threshold=0.2)]):
        result = await agent.invoke_async(prompt)
        return result.message.get("content", [{}])[0].get("text", "")


dataset = EvaluationDataset()
dataset.pull(alias="Single Turn QA")


@pytest.mark.parametrize("golden", dataset.goldens)
async def test_agentcore_agent(golden: Golden):
    await run_agent(golden.input)
    assert_test(golden=golden, metrics=[AnswerRelevancyMetric(threshold=0.8)])
