import pytest

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.contextual_relevancy.contextual_relevancy import (
    _split_relevant_irrelevant,
    _verdict_is_relevant,
)
from deepeval.metrics.contextual_relevancy.schema import (
    ContextualRelevancyVerdict,
    ContextualRelevancyVerdicts,
)
from deepeval.models import DeepEvalBaseLLM


class _StubModel(DeepEvalBaseLLM):
    """The behavior under test (verdict classification and scoring) never calls
    the LLM; the model only needs to exist for construction."""

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-model"

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "verdict-classification test must not call the LLM"
        )

    async def a_generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError(
            "verdict-classification test must not call the LLM"
        )


def _verdicts_list(entries) -> list:
    return [
        ContextualRelevancyVerdicts(
            verdicts=[
                ContextualRelevancyVerdict(statement=s, verdict=v, reason=r)
                for (s, v, r) in entries
            ]
        )
    ]


def _score(entries) -> float:
    metric = ContextualRelevancyMetric(
        model=_StubModel(), async_mode=False, include_reason=False
    )
    metric.verdicts_list = _verdicts_list(entries)
    return metric._calculate_score()


class TestVerdictNormalization:
    def test_verdict_is_relevant_matrix(self):
        # The exact corrupted tokens reported in #3079.
        relevant = ["yes", "yes.", "Yes ", "YES!", "  yes  "]
        not_relevant = ["no", "no.", "No", "maybe", "yes\u0578", None, ""]
        for token in relevant:
            assert _verdict_is_relevant(token) is True, repr(token)
        for token in not_relevant:
            assert _verdict_is_relevant(token) is False, repr(token)

    def test_score_counts_normalized_yes_as_relevant(self):
        # Regresses #3079: all four verdicts were "yes." which previously
        # scored 0.00 (the reason called them relevant).
        score = _score(
            [
                ("s1", "yes.", None),
                ("s2", "yes.", None),
                ("s3", "yes", None),
                ("s4", "Yes!", None),
            ]
        )
        assert score == 1.0

    def test_score_mixed_yes_and_no(self):
        # Backward compatibility for well-formed verdicts.
        score = _score(
            [
                ("s1", "yes", None),
                ("s2", "yes.", None),
                ("s3", "no", None),
            ]
        )
        assert score == pytest.approx(2 / 3)

    def test_score_ignores_ambiguous_tokens(self):
        score = _score(
            [
                ("s1", "maybe", None),
                ("s2", "yes", None),
            ]
        )
        assert score == 0.5

    def test_reason_buckets_agree_with_score(self):
        # A verdict must never be scored as irrelevant yet reasoned as relevant
        # (or vice versa). The reason path is built from the same predicate.
        relevant, irrelevant = _split_relevant_irrelevant(
            _verdicts_list(
                [
                    ("s_yes_punct", "yes.", "keep"),
                    ("s_no", "no", "drop"),
                    ("s_ambiguous", "maybe", None),
                    ("s_yes", "YES", "keep2"),
                ]
            )
        )
        assert relevant == ["s_yes_punct", "s_yes"]
        assert irrelevant == ["drop", None]

    def test_no_verdicts_scores_zero(self):
        assert _score([]) == 0.0
