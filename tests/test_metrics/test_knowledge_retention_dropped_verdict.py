"""KnowledgeRetentionMetric must survive a verdict the judge answered out of
vocabulary. generate_qag_verdict drops such a verdict to None, and the reason
step used to read ``.verdict`` off it.
"""

import json

import pytest

from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.metrics.knowledge_retention.schema import (
    Knowledge,
    KnowledgeRetentionScoreReason,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


class ScriptedJudge(DeepEvalBaseLLM):
    """Extracts one fact, then answers each verdict prompt in turn."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0
        super().__init__(model="scripted-judge")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        if schema is Knowledge:
            return json.dumps({"data": {"diet": "vegetarian"}})
        if schema is KnowledgeRetentionScoreReason:
            return json.dumps({"reason": "scripted reason"})
        verdict = self.verdicts[self.calls % len(self.verdicts)]
        self.calls += 1
        return json.dumps({"verdict": verdict, "reason": f"verdict {verdict}"})

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "scripted-judge"


def _test_case():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="I am vegetarian."),
            Turn(role="assistant", content="Noted."),
            Turn(role="user", content="Any dinner idea?"),
            Turn(role="assistant", content="Try a steak."),
        ]
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_dropped_verdict_does_not_break_the_reason(async_mode):
    metric = KnowledgeRetentionMetric(
        model=ScriptedJudge(["maybe", "yes"]), async_mode=async_mode
    )

    metric.measure(_test_case(), _show_indicator=False)

    assert metric.verdicts[0] is None
    assert metric.verdicts[1].verdict == "yes"
    assert metric.reason == "scripted reason"
