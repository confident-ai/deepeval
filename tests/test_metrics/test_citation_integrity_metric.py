import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import CitationIntegrityMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData

PASSAGES = [
    "The Eiffel Tower is 330 metres tall.",
    "The Eiffel Tower was completed in 1889.",
    "Paris is the capital of France.",
]


def _test_case(actual_output, retrieval_context=PASSAGES):
    return LLMTestCase(
        input="Tell me about the Eiffel Tower.",
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )


class TestCitationIntegrityMetric:
    """CitationIntegrityMetric is deterministic, so these run without an API key."""

    def test_every_sentence_cited_passes(self):
        metric = CitationIntegrityMetric()
        metric.measure(
            _test_case(
                "The tower is 330 metres tall [1]. It was completed in 1889 [2]."
            )
        )
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_marker_after_full_stop_is_attached(self):
        metric = CitationIntegrityMetric()
        metric.measure(
            _test_case(
                "The tower is 330 metres tall. [1] It opened in 1889. [2]"
            )
        )
        assert metric.score == 1.0

    def test_multiple_references_in_one_marker(self):
        metric = CitationIntegrityMetric()
        metric.measure(_test_case("It is 330 metres and from 1889 [1, 2]."))
        assert metric.score == 1.0

    def test_uncited_sentence_gives_partial_credit(self):
        metric = CitationIntegrityMetric(threshold=0.5)
        metric.measure(
            _test_case("The tower is 330 metres tall [1]. It is very popular.")
        )
        assert metric.score == 0.5
        assert metric.is_successful() is True
        assert "1 of 2" in metric.reason

    def test_uncited_sentence_fails_default_threshold(self):
        metric = CitationIntegrityMetric()
        metric.measure(
            _test_case("The tower is 330 metres tall [1]. It is very popular.")
        )
        assert metric.score == 0.5
        assert metric.is_successful() is False

    def test_out_of_range_index_forces_zero(self):
        metric = CitationIntegrityMetric(threshold=0.1)
        metric.measure(
            _test_case(
                "The tower is 330 metres tall [1]. It opened in 1889 [7]."
            )
        )
        assert metric.score == 0.0
        assert metric.is_successful() is False
        assert "[7]" in metric.reason

    def test_index_zero_is_broken(self):
        metric = CitationIntegrityMetric()
        metric.measure(_test_case("The tower is 330 metres tall [0]."))
        assert metric.score == 0.0

    def test_source_name_citations_resolve(self):
        context = [
            RetrievedContextData(context=PASSAGES[0], source="tower_facts"),
            RetrievedContextData(context=PASSAGES[1], source="history"),
        ]
        metric = CitationIntegrityMetric()
        metric.measure(
            _test_case(
                "It is 330 metres tall [tower_facts]. It opened in 1889 [2].",
                retrieval_context=context,
            )
        )
        assert metric.score == 1.0

    def test_invented_source_forces_zero(self):
        context = [
            RetrievedContextData(context=PASSAGES[0], source="tower_facts")
        ]
        metric = CitationIntegrityMetric()
        metric.measure(
            _test_case(
                "It is 330 metres tall [tower_facts_v2].",
                retrieval_context=context,
            )
        )
        assert metric.score == 0.0
        assert "[tower_facts_v2]" in metric.reason

    def test_markdown_link_is_not_a_citation(self):
        metric = CitationIntegrityMetric()
        metric.measure(
            _test_case(
                "The tower is 330 metres tall [1]. "
                "See [the official site](https://www.toureiffel.paris)."
            )
        )
        assert metric.score == 0.5
        assert "broken" not in metric.reason

    def test_no_citations_scores_zero(self):
        metric = CitationIntegrityMetric()
        metric.measure(_test_case("The tower is tall. It is in Paris."))
        assert metric.score == 0.0
        assert "0 of 2" in metric.reason

    def test_strict_mode_zeroes_partial_success(self):
        metric = CitationIntegrityMetric(strict_mode=True)
        metric.measure(
            _test_case("The tower is 330 metres tall [1]. It is very popular.")
        )
        assert metric.score == 0
        assert metric.is_successful() is False

    def test_include_reason_false(self):
        metric = CitationIntegrityMetric(include_reason=False)
        metric.measure(_test_case("The tower is 330 metres tall [1]."))
        assert metric.reason is None

    def test_missing_retrieval_context_raises(self):
        metric = CitationIntegrityMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(
                LLMTestCase(
                    input="Tell me about the Eiffel Tower.",
                    actual_output="It is 330 metres tall [1].",
                )
            )

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = CitationIntegrityMetric()
        score = await metric.a_measure(
            _test_case("The tower is 330 metres tall [1]. It is popular.")
        )
        assert score == 0.5
        assert metric.is_successful() is False
