"""Manual smoke test: async + span-level metric.

    python try_evals_iterator_async_span.py

Metric is declared on @observe(metrics=[...]) and evaluated on the span.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import List

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.tracing import observe


AGENT_DELAY_SEC = 0.6
METRIC_DELAY_SEC = 0.4


class RandomScoreMetric(BaseMetric):
    threshold: float = 0.5
    async_mode: bool = True
    _required_params: List[LLMTestCaseParams] = [LLMTestCaseParams.INPUT]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _finalize(self) -> float:
        self.score = random.random()
        self.success = self.score >= self.threshold
        self.reason = f"random score {self.score:.3f}"
        return self.score

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        time.sleep(METRIC_DELAY_SEC)
        return self._finalize()

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await asyncio.sleep(METRIC_DELAY_SEC)
        return self._finalize()

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "RandomScore"


QUESTIONS = [
    f"[{i}] {q}"
    for i, q in enumerate(
        [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What is 2 + 2?",
            "Define entropy.",
            "What is the speed of light?",
        ]
    )
]


@observe(type="agent", name="span_metric_agent", metrics=[RandomScoreMetric()])
def agent(question: str) -> str:
    time.sleep(AGENT_DELAY_SEC)
    return f"Answer to {question!r} is 42."


if __name__ == "__main__":
    dataset = EvaluationDataset(goldens=[Golden(input=q) for q in QUESTIONS])
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=True),
        display_config=DisplayConfig(show_indicator=True, verbose_mode=False),
    ):
        agent(golden.input)
