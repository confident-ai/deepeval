import pytest

from deepeval.metrics import HallucinationMetric
from deepeval.metrics.hallucination.hallucination import _is_contradiction
from deepeval.metrics.hallucination.schema import HallucinationVerdict
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


def _verdict(verdict: str) -> HallucinationVerdict:
    # model_construct bypasses the Literal["yes", "no"] validation so we can
    # inject the malformed tokens the lenient JSON extraction can produce.
    return HallucinationVerdict.model_construct(
        verdict=verdict, reason="reason-%s" % verdict
    )


def _score(*verdicts: str) -> float:
    metric = HallucinationMetric(
        model=_StubModel(), async_mode=False, include_reason=False
    )
    metric.verdicts = [_verdict(v) for v in verdicts]
    return metric._calculate_score()


def _reason_buckets(metric: HallucinationMetric) -> tuple:
    # Re-implement the bucket split exactly as the reason generators do, so we
    # can assert structural agreement with _calculate_score.
    factual_alignments = []
    contradictions = []
    for verdict in metric.verdicts:
        if _is_contradiction(verdict.verdict):
            contradictions.append(verdict.reason)
        else:
            factual_alignments.append(verdict.reason)
    return factual_alignments, contradictions


class TestHallucinationVerdictNormalization:
    def test_is_contradiction_matrix(self):
        contradiction = ["no", "no.", "No ", "NO!", "  no  "]
        aligned = ["yes", "yes.", "Yes ", "maybe", "yes\u0578", None, ""]
        for token in contradiction:
            assert _is_contradiction(token) is True, repr(token)
        for token in aligned:
            assert _is_contradiction(token) is False, repr(token)

    def test_malformed_yes_scores_aligned(self):
        # Regresses #3098: "yes." scored as aligned (0 hallucinations) but the
        # reason listed it as a contradiction. Now both agree it is aligned.
        assert _score("yes.") == 1.0

    def test_malformed_no_counts_as_contradiction(self):
        # "no." is semantically a contradiction; it must drag the score down.
        assert _score("no.") == 0.0

    def test_mixed_verdicts_score(self):
        # Backward compatibility for well-formed verdicts.
        assert _score("yes", "yes", "no") == pytest.approx(2 / 3)

    def test_reason_buckets_agree_with_score(self):
        metric = HallucinationMetric(
            model=_StubModel(), async_mode=False, include_reason=False
        )
        metric.verdicts = [
            _verdict("yes."),  # aligned (score counts it, reason must too)
            _verdict("no"),  # contradiction
            _verdict("maybe"),  # aligned (not an explicit "no")
        ]
        factual, contradictions = _reason_buckets(metric)
        assert factual == ["reason-yes.", "reason-maybe"]
        assert contradictions == ["reason-no"]
        # The classification used by the score matches the reason buckets.
        score = metric._calculate_score()
        assert score == pytest.approx(len(factual) / len(metric.verdicts))

    def test_empty_verdicts_score_one(self):
        metric = HallucinationMetric(
            model=_StubModel(), async_mode=False, include_reason=False
        )
        metric.verdicts = []
        assert metric._calculate_score() == 1.0
