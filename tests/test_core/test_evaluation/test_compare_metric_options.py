import json

import pytest

from deepeval import compare
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import ArenaGEval
from deepeval.metrics.arena_g_eval.arena_g_eval import ArenaGEvalTemplate
from deepeval.metrics.arena_g_eval.utils import FAKE_NAMES
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ArenaTestCase,
    Contestant,
    LLMTestCase,
    SingleTurnParams,
)


class RecordingJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []
        super().__init__("recording-judge")

    def load_model(self):
        return None

    def get_model_name(self):
        return "recording-judge"

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        if "rewritten_reason" in prompt:
            return json.dumps({"rewritten_reason": "reason"})
        winner = next(name for name in FAKE_NAMES if f'"{name}"' in prompt)
        return json.dumps({"winner": winner, "reason": "reason"})

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema)


class CustomArenaTemplate(ArenaGEvalTemplate):
    @staticmethod
    def generate_arena_winner(evaluation_steps, test_case_contents, parameters):
        return f"CUSTOM ARENA PROMPT {test_case_contents!r}"


@pytest.mark.parametrize("run_async", [False, True])
def test_compare_uses_the_metric_evaluation_template(run_async):
    judge = RecordingJudge()
    metric = ArenaGEval(
        name="Friendly",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=["Pick the friendlier reply."],
        model=judge,
        evaluation_template=CustomArenaTemplate,
    )
    test_case = ArenaTestCase(
        contestants=[
            Contestant(
                name="A",
                test_case=LLMTestCase(input="Hi", actual_output="Hello!"),
            ),
            Contestant(
                name="B",
                test_case=LLMTestCase(input="Hi", actual_output="Hm."),
            ),
        ]
    )

    compare(
        test_cases=[test_case],
        metric=metric,
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=True),
    )

    assert any(p.startswith("CUSTOM ARENA PROMPT") for p in judge.prompts)
