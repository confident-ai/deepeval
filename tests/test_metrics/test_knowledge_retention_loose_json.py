"""Regression tests for loose-JSON judges in KnowledgeRetentionMetric."""

import asyncio
import json

import pytest

from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.metrics.base_metric import Verdict
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


class LooseJSONJudge(DeepEvalBaseLLM):
    def __init__(self, verdict_payload):
        self.verdict_payload = verdict_payload
        super().__init__()

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        schema_name = schema.__name__ if schema is not None else None
        if schema_name == "Knowledge":
            return json.dumps({"data": {"tls": "HTTPS encrypts traffic"}})
        if schema_name == "KnowledgeRetentionVerdict":
            return json.dumps(self.verdict_payload)
        if schema_name == "KnowledgeRetentionScoreReason":
            return json.dumps({"reason": "stub reason"})
        raise AssertionError(f"unexpected schema: {schema_name}")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "loose-json-judge"


def _test_case():
    return ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="How does HTTPS prevent eavesdropping?",
            ),
            Turn(
                role="assistant",
                content="HTTPS encrypts traffic with TLS.",
            ),
        ]
    )


def _measure_metric(metric, async_mode):
    if async_mode:
        asyncio.run(metric.a_measure(_test_case(), _show_indicator=False))
    else:
        metric.measure(_test_case(), _show_indicator=False)


@pytest.mark.parametrize("async_mode", [False, True])
def test_extra_verdict_key_is_ignored(async_mode):
    metric = KnowledgeRetentionMetric(
        model=LooseJSONJudge(
            {
                "verdict": "no",
                "reason": "assistant recalled it",
                "knowledge": "TLS encrypts traffic",
            }
        ),
        async_mode=async_mode,
    )

    _measure_metric(metric, async_mode)

    assert len(metric.verdicts) == 1
    assert metric.verdicts[0].verdict == Verdict.NO
    assert metric.score == 1.0


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("include_reason", [False, True])
def test_invalid_verdict_fails_closed_without_crashing(
    async_mode, include_reason
):
    metric = KnowledgeRetentionMetric(
        model=LooseJSONJudge(
            {
                "verdict": "unsure",
                "reason": "cannot tell",
            }
        ),
        include_reason=include_reason,
        async_mode=async_mode,
    )

    _measure_metric(metric, async_mode)

    assert metric.verdicts == [None]
    assert metric.score == 0.0
    assert metric.reason == ("stub reason" if include_reason else None)
