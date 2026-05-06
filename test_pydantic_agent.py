"""test_pydantic_agent.py — pytest analog of ``pydantic_after_evals_iterator.py``.

Run with::

    deepeval test run test_pydantic_agent.py

Same 3 goldens, same agent setup, but driven by pytest + ``assert_test``
instead of ``dataset.evals_iterator``. The deepeval pytest plugin
(``deepeval test run``) wraps each test in an eval session so the agent's
OTel spans route through REST and the trace gets evaluated against the
metrics passed to ``assert_test``.

Requirements:
  - ``CONFIDENT_API_KEY`` in env (or ``deepeval login``)
  - ``OPENAI_API_KEY`` in env
  - ``pip install pydantic-ai pytest``
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from pydantic_ai import Agent

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing.context import next_agent_span


RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise. Reply with one short sentence.",
    instrument=DeepEvalInstrumentationSettings(),
)


async def run_agent(prompt: str) -> str:
    # Span-level metric attached to the agent span via next_agent_span;
    # trace-level metric is passed to assert_test below. Mirrors the
    # split used in pydantic_after_evals_iterator.py.
    with next_agent_span(metrics=[AnswerRelevancyMetric(threshold=0.2)]):
        result = await agent.run(prompt)
        return result.output


dataset = EvaluationDataset()
dataset.pull(alias="Single Turn QA")


@pytest.mark.parametrize("golden", dataset.goldens)
async def test_pydantic_agent(golden: Golden):
    # await agent.run(golden.input)
    await run_agent(golden.input)
    # asyncio.run(run_agent(golden.input))
    assert_test(golden=golden, metrics=[AnswerRelevancyMetric(threshold=0.8)])
