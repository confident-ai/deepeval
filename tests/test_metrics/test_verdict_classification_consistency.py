"""Score and reason must classify every verdict with the same predicate.

KnowledgeRetentionMetric, NonAdviceMetric, PIILeakageMetric and
SummarizationMetric each classified verdicts twice with different tests:
`_calculate_score` keyed on one exact token (e.g. counts only an explicit
"no" as a good verdict) while `_generate_reason` keyed on the opposite token
(lists only an explicit "yes" as a violation). Any verdict string that is
neither — "yes.", "Yes!", an untagged value from the lenient JSON extraction —
lowered the score without ever being mentioned in the reason. Same defect
class as #3079 (ContextualRelevancy) and #3098 (Hallucination).

These tests run with a stub model, so they need no LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import (
    KnowledgeRetentionMetric,
    NonAdviceMetric,
    PIILeakageMetric,
    SummarizationMetric,
)
from deepeval.metrics.knowledge_retention.schema import (
    KnowledgeRetentionVerdict,
)
from deepeval.metrics.non_advice.schema import NonAdviceVerdict
from deepeval.metrics.pii_leakage.schema import PIILeakageVerdict
from deepeval.metrics.summarization.schema import (
    ScoreType,
    SummarizationAlignmentVerdict,
    SummarizationCoverageVerdict,
)
from deepeval.models.base_model import DeepEvalBaseLLM


class _CapturingLLM(DeepEvalBaseLLM):
    """Records every prompt it is given and answers any reason schema."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, *args, **kwargs):
        self.prompts.append(prompt)
        return schema(reason="stub reason")

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema=schema, *args, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


def _reason_prompt(metric, use_async: bool) -> str:
    if use_async:
        asyncio.run(metric._a_generate_reason())
    else:
        metric._generate_reason()
    assert len(metric.model.prompts) == 1
    return metric.model.prompts[0]


NOT_EXACT_TOKENS = ["yes.", "Yes!", "unknown"]


@pytest.mark.parametrize("verdict", NOT_EXACT_TOKENS)
@pytest.mark.parametrize("use_async", [False, True])
def test_knowledge_retention_score_and_reason_agree(verdict, use_async):
    metric = KnowledgeRetentionMetric(model=_CapturingLLM())
    metric.verdicts = [
        KnowledgeRetentionVerdict(verdict=verdict, reason="forgot the name")
    ]
    metric.score = metric._calculate_score()

    assert metric.score == 0.0
    assert "forgot the name" in _reason_prompt(metric, use_async)


def test_knowledge_retention_explicit_no_is_retained():
    metric = KnowledgeRetentionMetric(model=_CapturingLLM())
    metric.verdicts = [KnowledgeRetentionVerdict(verdict="no", reason="kept")]
    metric.score = metric._calculate_score()

    assert metric.score == 1.0
    assert "kept" not in _reason_prompt(metric, use_async=False)


@pytest.mark.parametrize("verdict", NOT_EXACT_TOKENS)
@pytest.mark.parametrize("use_async", [False, True])
def test_non_advice_score_and_reason_agree(verdict, use_async):
    metric = NonAdviceMetric(advice_types=["financial"], model=_CapturingLLM())
    metric.verdicts = [
        NonAdviceVerdict(verdict=verdict, reason="told the user to buy stock")
    ]
    metric.score = metric._calculate_score()

    assert metric.score == 0.0
    assert "told the user to buy stock" in _reason_prompt(metric, use_async)


def test_non_advice_explicit_no_is_appropriate():
    metric = NonAdviceMetric(advice_types=["financial"], model=_CapturingLLM())
    metric.verdicts = [NonAdviceVerdict(verdict="no", reason="fine")]
    metric.score = metric._calculate_score()

    assert metric.score == 1.0
    assert "fine" not in _reason_prompt(metric, use_async=False)


@pytest.mark.parametrize("verdict", NOT_EXACT_TOKENS)
@pytest.mark.parametrize("use_async", [False, True])
def test_pii_leakage_score_and_reason_agree(verdict, use_async):
    metric = PIILeakageMetric(model=_CapturingLLM())
    metric.verdicts = [
        PIILeakageVerdict(verdict=verdict, reason="leaked the user's email")
    ]
    metric.score = metric._calculate_score()

    assert metric.score == 0.0
    assert "leaked the user's email" in _reason_prompt(metric, use_async)


def test_pii_leakage_explicit_no_is_safe():
    metric = PIILeakageMetric(model=_CapturingLLM())
    metric.verdicts = [PIILeakageVerdict(verdict="no", reason="safe")]
    metric.score = metric._calculate_score()

    assert metric.score == 1.0
    assert "safe" not in _reason_prompt(metric, use_async=False)


@pytest.mark.parametrize("use_async", [False, True])
def test_summarization_score_and_reason_agree(use_async):
    metric = SummarizationMetric(model=_CapturingLLM())
    metric.assessment_questions = ["Who founded the company?"]
    # model_construct bypasses the Literal, exactly like the lenient JSON
    # extraction fallback does at runtime.
    metric.alignment_verdicts = [
        SummarizationAlignmentVerdict.model_construct(
            verdict="yes.", reason="invented a statistic"
        )
    ]
    metric.coverage_verdicts = [
        SummarizationCoverageVerdict(
            original_verdict="yes",
            summary_verdict="no.",
            question="Who founded the company?",
        )
    ]

    assert metric._calculate_score(ScoreType.ALIGNMENT) == 0.0
    assert metric._calculate_score(ScoreType.COVERAGE) == 0.0
    metric.score = 0.0

    prompt = _reason_prompt(metric, use_async)
    assert "invented a statistic" in prompt
    assert "Who founded the company?" in prompt


def test_summarization_idk_is_still_a_redundancy():
    metric = SummarizationMetric(model=_CapturingLLM())
    metric.alignment_verdicts = [
        SummarizationAlignmentVerdict(verdict="idk", reason="pure fluff")
    ]
    metric.coverage_verdicts = []
    metric.score = 0.0

    assert "pure fluff" in _reason_prompt(metric, use_async=False)
