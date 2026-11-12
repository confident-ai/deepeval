"""Regression tests for #3629: empty judge output in the turn metrics.

When the judge returns no verdicts for retrieval contexts that do exist, the
contextual turn metrics used to report "There were no retrieval contexts",
which hid the failure. The score is unchanged (1); the reason must now say
that the judge returned nothing. TurnFaithfulness keeps its 1.0 for an empty
claim list but says that no claims were extracted.

The stub judge answers every structured call with empty lists, so these tests
run without an LLM provider API key.
"""

import pytest

from deepeval.metrics import (
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


class _EmptyExtractionLLM(DeepEvalBaseLLM):
    """Returns valid schema instances whose list fields are empty."""

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        fields = {}
        for name, field in schema.model_fields.items():
            if "List" in str(field.annotation) or "list" in str(
                field.annotation
            ):
                fields[name] = []
            else:
                fields[name] = "stub reason"
        return schema(**fields)

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        return "empty-extraction-stub"


def _test_case(retrieval_context):
    return ConversationalTestCase(
        expected_outcome="The assistant explains how HTTPS works.",
        turns=[
            Turn(role="user", content="How does HTTPS prevent eavesdropping?"),
            Turn(
                role="assistant",
                content="HTTPS encrypts traffic with TLS session keys.",
                retrieval_context=retrieval_context,
            ),
        ],
    )


CONTEXTUAL_METRICS = [
    (TurnContextualPrecisionMetric, "precision"),
    (TurnContextualRecallMetric, "recall"),
    (TurnContextualRelevancyMetric, "relevancy"),
]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("metric_cls, name", CONTEXTUAL_METRICS)
def test_judge_returning_no_verdicts_is_reported(metric_cls, name, async_mode):
    metric = metric_cls(
        model=_EmptyExtractionLLM(), async_mode=async_mode, verbose_mode=False
    )
    metric.measure(_test_case(["ctx one", "ctx two"]), _show_indicator=False)

    assert metric.score == 1
    assert (
        "Reason: The evaluation model returned no verdicts for the 2 "
        "retrieval context(s)" in metric.verbose_logs
    )
    assert f"the contextual {name} could not be evaluated" in (
        metric.verbose_logs
    )
    assert "There were no retrieval contexts" not in metric.verbose_logs


@pytest.mark.parametrize("metric_cls, name", CONTEXTUAL_METRICS)
def test_empty_verdicts_reason_without_contexts_is_unchanged(metric_cls, name):
    metric = metric_cls(model=_EmptyExtractionLLM())
    assert metric._empty_verdicts_reason(0) == (
        "There were no retrieval contexts in the given turns to evaluate "
        f"the contextual {name}."
    )


@pytest.mark.parametrize("metric_cls, name", CONTEXTUAL_METRICS)
def test_score_and_reason_default_keeps_old_message(metric_cls, name):
    # Callers that do not pass the context count get the previous behavior.
    metric = metric_cls(model=_EmptyExtractionLLM())
    score, reason = metric._get_interaction_score_and_reason(
        "input", [], multimodal=False
    )
    assert score == 1
    assert reason.startswith("There were no retrieval contexts")


@pytest.mark.parametrize("async_mode", [False, True])
def test_faithfulness_reason_says_no_claims_were_extracted(async_mode):
    metric = TurnFaithfulnessMetric(
        model=_EmptyExtractionLLM(), async_mode=async_mode
    )
    metric.measure(_test_case(["ctx one"]), _show_indicator=False)

    assert metric.score == 1.0
    assert (
        "Reason: No claims were extracted from the assistant turns"
        in metric.verbose_logs
    )
