"""
Tests for the ``evaluation_template`` parameter — the documented ability to
override a metric's LLM-as-a-judge prompts by subclassing its template class.

All tests are offline: they inject a stub LLM so no metric constructs a real
provider client, and they assert on the prompt strings produced by
``_get_prompt`` (no model calls / network required).
"""

import pytest

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.templates.resolver import resolve_template
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval.metrics.faithfulness import FaithfulnessTemplate
from deepeval.metrics.g_eval import GEvalTemplate
from deepeval.metrics.contextual_precision import ContextualPrecisionTemplate
from deepeval.metrics.contextual_recall import ContextualRecallTemplate
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
from deepeval.metrics.prompt_template import BasePromptTemplate


class _StubLLM(DeepEvalBaseLLM):
    """Non-native stub so metrics construct without any provider API key."""

    def __init__(self):
        pass

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs) -> str:
        return ""

    async def a_generate(self, *args, **kwargs) -> str:
        return ""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub"


# --------------------------------------------------------------------------- #
# Template classes exist, are exported, and are subclassable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "template_cls",
    [
        AnswerRelevancyTemplate,
        FaithfulnessTemplate,
        GEvalTemplate,
        ContextualPrecisionTemplate,
        ContextualRecallTemplate,
        ContextualRelevancyTemplate,
    ],
)
def test_template_classes_are_subclassable_base_templates(template_cls):
    assert issubclass(template_cls, BasePromptTemplate)


# --------------------------------------------------------------------------- #
# AnswerRelevancy — the documented example
# --------------------------------------------------------------------------- #


def test_answer_relevancy_override_is_used():
    class CustomTemplate(AnswerRelevancyTemplate):
        @staticmethod
        def generate_statements(actual_output):
            return f"CUSTOM::{actual_output}"

    metric = AnswerRelevancyMetric(
        model=_StubLLM(), evaluation_template=CustomTemplate
    )
    prompt = metric._get_prompt(
        "generate_statements", multimodal=False, actual_output="hello"
    )
    assert prompt == "CUSTOM::hello"


def test_non_overridden_method_falls_back_to_default_template():
    class CustomTemplate(AnswerRelevancyTemplate):
        @staticmethod
        def generate_statements(actual_output):
            return "CUSTOM"

    metric = AnswerRelevancyMetric(
        model=_StubLLM(), evaluation_template=CustomTemplate
    )
    # generate_verdicts is NOT overridden -> uses deepeval's default template.
    prompt = metric._get_prompt(
        "generate_verdicts",
        multimodal=False,
        input="q",
        statements=["s1", "s2"],
    )
    assert "CUSTOM" not in prompt
    assert isinstance(prompt, str) and len(prompt) > 50


def test_default_metric_is_byte_identical_to_resolver():
    """With no custom template (default), the prompt must be exactly what the
    resolver produces — i.e. zero behavior change for existing users."""
    metric = AnswerRelevancyMetric(model=_StubLLM())
    got = metric._get_prompt(
        "generate_statements", multimodal=False, actual_output="x"
    )
    expected = resolve_template(
        "metrics",
        "AnswerRelevancyMetric",
        "generate_statements",
        multimodal=False,
        actual_output="x",
    )
    assert got == expected


def test_override_receives_the_metric_kwargs():
    captured = {}

    class CustomTemplate(AnswerRelevancyTemplate):
        @staticmethod
        def generate_reason(irrelevant_statements, input, score):
            captured["irrelevant_statements"] = irrelevant_statements
            captured["input"] = input
            captured["score"] = score
            return "R"

    metric = AnswerRelevancyMetric(
        model=_StubLLM(), evaluation_template=CustomTemplate
    )
    out = metric._get_prompt(
        "generate_reason",
        multimodal=False,
        irrelevant_statements=["bad"],
        input="q",
        score="0.50",
    )
    assert out == "R"
    assert captured == {
        "irrelevant_statements": ["bad"],
        "input": "q",
        "score": "0.50",
    }


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #


def test_faithfulness_override_is_used():
    class CustomTemplate(FaithfulnessTemplate):
        @staticmethod
        def generate_claims(**kwargs):
            return "CUSTOM_CLAIMS"

    metric = FaithfulnessMetric(
        model=_StubLLM(), evaluation_template=CustomTemplate
    )
    prompt = metric._get_prompt(
        "generate_claims",
        multimodal=False,
        actual_output="o",
        multimodal_instruction="",
    )
    assert prompt == "CUSTOM_CLAIMS"
    # a non-overridden method still falls back
    verdicts = metric._get_prompt(
        "generate_verdicts",
        multimodal=False,
        claims=["c"],
        retrieval_context="ctx",
    )
    assert "CUSTOM_CLAIMS" not in verdicts and len(verdicts) > 50


# --------------------------------------------------------------------------- #
# GEval — the metric the maintainers explicitly asked for (#1989 / #1757)
# --------------------------------------------------------------------------- #


def test_geval_override_is_used():
    class CustomTemplate(GEvalTemplate):
        @staticmethod
        def generate_evaluation_steps(**kwargs):
            return "CUSTOM_STEPS"

    metric = GEval(
        name="Custom",
        criteria="Some criteria",
        model=_StubLLM(),
        evaluation_template=CustomTemplate,
    )
    prompt = metric._get_prompt(
        "generate_evaluation_steps",
        multimodal=False,
        criteria="Some criteria",
        parameters="Input, Actual Output",
    )
    assert prompt == "CUSTOM_STEPS"


def test_geval_default_falls_back_to_resolver():
    metric = GEval(name="Custom", criteria="c", model=_StubLLM())
    got = metric._get_prompt(
        "generate_evaluation_steps",
        multimodal=False,
        criteria="c",
        parameters="Input",
    )
    expected = resolve_template(
        "metrics",
        "GEval",
        "generate_evaluation_steps",
        multimodal=False,
        criteria="c",
        parameters="Input",
    )
    assert got == expected


# --------------------------------------------------------------------------- #
# Contextual metrics (RAG retriever suite)
# --------------------------------------------------------------------------- #


_CONTEXTUAL_CASES = [
    (ContextualPrecisionMetric, ContextualPrecisionTemplate),
    (ContextualRecallMetric, ContextualRecallTemplate),
    (ContextualRelevancyMetric, ContextualRelevancyTemplate),
]


@pytest.mark.parametrize("metric_cls,template_cls", _CONTEXTUAL_CASES)
def test_contextual_metric_override_is_used(metric_cls, template_cls):
    class CustomTemplate(template_cls):
        @staticmethod
        def generate_verdicts(**kwargs):
            return "CUSTOM_VERDICTS"

    metric = metric_cls(model=_StubLLM(), evaluation_template=CustomTemplate)
    assert (
        metric._get_prompt("generate_verdicts", multimodal=False)
        == "CUSTOM_VERDICTS"
    )


def test_contextual_precision_default_matches_resolver():
    metric = ContextualPrecisionMetric(model=_StubLLM())
    kwargs = dict(
        input="q",
        expected_output="e",
        document_count_str="1",
        context_to_display="ctx",
        multimodal_note="",
    )
    got = metric._get_prompt("generate_verdicts", multimodal=False, **kwargs)
    expected = resolve_template(
        "metrics",
        "ContextualPrecisionMetric",
        "generate_verdicts",
        multimodal=False,
        **kwargs,
    )
    assert got == expected
