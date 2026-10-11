"""Offline checks that the LLM judge sees the same inputs Jev does, and that
rendered prompts contain no unrendered template syntax."""

import pytest

from deepeval.classifiers import RequiredDisclosureClassifier
from deepeval.metrics import NonAdviceMetric, RoleViolationMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessTemplate
from deepeval.test_case import LLMTestCase
from tests.test_metrics.system_one_fakes import CannedLLM, ScriptedLLM


def test_non_advice_verdict_prompt_lists_configured_advice_types():
    llm = ScriptedLLM(
        [
            '{"advices": ["You should buy NVDA."]}',
            '{"verdicts": [{"verdict": "yes", "reason": "Stock pick."}]}',
            '{"reason": "ok"}',
        ]
    )
    metric = NonAdviceMetric(
        advice_types=["financial", "crypto"],
        model=llm,
        eval_mode="llm",
        async_mode=False,
    )

    metric.measure(
        LLMTestCase(input="Any tips?", actual_output="You should buy NVDA.")
    )

    verdict_prompt = llm.prompts[1]
    assert "Advice types to flag: financial, crypto" in verdict_prompt


def test_role_violation_verdict_prompt_includes_expected_role():
    llm = ScriptedLLM(
        [
            '{"role_violations": ["I am a human."]}',
            '{"verdicts": [{"verdict": "yes", "reason": "Claims humanity."}]}',
            '{"reason": "ok"}',
        ]
    )
    metric = RoleViolationMetric(
        role="pirate-themed support bot",
        model=llm,
        eval_mode="llm",
        async_mode=False,
    )

    metric.measure(LLMTestCase(input="Who are you?", actual_output="Arr."))

    verdict_prompt = llm.prompts[1]
    assert "Expected Role: pirate-themed support bot" in verdict_prompt


@pytest.mark.parametrize("multimodal", [False, True])
def test_faithfulness_verdict_prompt_has_no_template_braces(multimodal):
    prompt = FaithfulnessTemplate.generate_verdicts(
        claims=["Paris is in France."],
        retrieval_context="Paris is the capital of France.",
        multimodal=multimodal,
    )

    assert "{{" not in prompt and "}}" not in prompt
    assert '"verdicts": [' in prompt


def test_required_disclosure_classifier_constructs_with_disclosures():
    classifier = RequiredDisclosureClassifier(
        disclosures=["AI disclosure"], model=CannedLLM(), eval_mode="llm"
    )

    assert classifier.name == "required_disclosure"
    assert [label.name for label in classifier.labels] == [
        "present",
        "missing",
        "partial",
    ]
    assert "AI disclosure" in classifier.labels[0].description
