import asyncio
import threading
import pytest
import time
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
    ]
)

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
    try:
        for golden in dataset.evals_iterator():
            task = asyncio.create_task(run_agent(golden.input))
            dataset.evaluate(task)
    except:
        pass

def test_evaluate_agent():
    run_eval()
    assert answer_relavancy_metric.score > 0.0