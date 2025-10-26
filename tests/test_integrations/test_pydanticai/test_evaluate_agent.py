import asyncio
import os
import pytest
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig


dataset = EvaluationDataset(goldens=[Golden(input="What's 7 * 8?")])

answer_relavancy_metric = AnswerRelevancyMetric()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    instrument=ConfidentInstrumentationSettings(
        agent_metrics=[answer_relavancy_metric],
        is_test_mode=True,
    ),
)


async def run_agent(input: str):
    return await agent.run(input)


def run_eval():
    # use the ASYNC iterator path so it collects and awaits our tasks,
    # then finalizes and serializes traces.
    # don't try / except pass.. or we won't know what went wrong.
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=True)
    ):
        task = asyncio.create_task(run_agent(golden.input))
        dataset.evaluate(task)


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="needs OPENAI_API_KEY",
)
def test_evaluate_agent():
    run_eval()

    assert answer_relavancy_metric.score is not None
    assert answer_relavancy_metric.score > 0.0
