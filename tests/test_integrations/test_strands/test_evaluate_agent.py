"""Component-level evals for Strands via ``dataset.evals_iterator``.

Mirrors ``tests/test_integrations/test_agentcore/test_evaluate_agent.py``:
drives a Strands agent through the async iterator path, with a
per-task ``next_agent_span(metrics=[...])`` wrap so the
``AnswerRelevancyMetric`` lands on the agent span via the
``stash_pending_metrics`` overlay (carried across OTel transport into
``ConfidentSpanExporter``). The ``evals_iterator`` itself sets
``trace_manager.is_evaluating=True``, which:

  - flips ``ContextAwareSpanProcessor`` to REST routing so the spans
    flow through ``trace_manager`` (instead of OTLP), and
  - gates ``stash_pending_metrics`` so ``BaseMetric`` instances
    actually make it from the interceptor to the exporter.

This is the canonical end-to-end shape for Strands + component-level
evals after the OTel POC migration.

Skipped without ``OPENAI_API_KEY`` (used both for Strands' OpenAIModel
provider and for the AnswerRelevancyMetric scorer).
"""

import asyncio
import os

import pytest

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import next_agent_span

from tests.test_integrations.test_strands.apps.strands_eval_app import (
    ainvoke_evals_agent,
    init_evals_strands,
)


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason=(
        "OPENAI_API_KEY is required for both Strands' OpenAIModel "
        "provider and the AnswerRelevancyMetric scorer."
    ),
)


answer_relevancy_metric = AnswerRelevancyMetric()


def test_evaluate_agent():
    """End-to-end: 1 golden through a Strands agent, scored by
    AnswerRelevancyMetric attached via ``next_agent_span(metrics=[...])``.
    """
    invoke_func = init_evals_strands(
        name="strands-evaluate-agent",
        tags=["strands", "evaluate", "iterator"],
        metadata={"test_type": "evaluate_agent"},
        thread_id="evaluate-agent-thread-001",
        user_id="evaluate-agent-user-001",
    )

    dataset = EvaluationDataset(
        goldens=[Golden(input="What's 7 multiplied by 8?")]
    )

    async def run_agent(prompt: str):
        # Span-level metric attached to the agent span via
        # next_agent_span; with ``trace_manager.is_evaluating=True`` set
        # by evals_iterator, the interceptor's ``stash_pending_metrics``
        # call carries the metric across OTel transport so the
        # exporter can re-attach it on the rebuilt AgentSpan.
        with next_agent_span(metrics=[answer_relevancy_metric]):
            return await ainvoke_evals_agent(prompt, invoke_func=invoke_func)

    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=True),
        metrics=[answer_relevancy_metric],
    ):
        task = asyncio.create_task(run_agent(golden.input))
        dataset.evaluate(task)

    assert answer_relevancy_metric.score is not None
    assert answer_relevancy_metric.score > 0.0
