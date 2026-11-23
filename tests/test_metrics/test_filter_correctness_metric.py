import pytest

from deepeval.metrics import FilterCorrectnessMetric
from deepeval.test_case import LLMTestCase, RetrievedContextData


def _chunk(context, source, metadata=None):
    return RetrievedContextData(
        context=context, source=source, metadata=metadata
    )


def _test_case(chunks):
    return LLMTestCase(
        input="show me tenant A documents",
        actual_output="Here are the documents.",
        retrieval_context=chunks,
    )


class TestFilterCorrectnessMetric:
    """FilterCorrectnessMetric is deterministic, so these run without any API key."""

    def test_all_chunks_compliant_passes(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values="A"
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", {"tenant_id": "A"}),
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_one_violation_reduces_score(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values="A"
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", {"tenant_id": "B"}),
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 0.5
        assert metric.is_successful() is False
        assert "tenant_id" in metric.reason
        assert "'B'" in metric.reason

    def test_all_violations_zero_score(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values=["A", "C"]
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "B"}),
                    _chunk("doc2", "s2", {"tenant_id": "D"}),
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 0.0
        assert metric.is_successful() is False

    def test_allowed_values_list_semantics(self):
        metric = FilterCorrectnessMetric(
            filter_field="jurisdiction",
            allowed_values=["US", "CA"],
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"jurisdiction": "US"}),
                    _chunk("doc2", "s2", {"jurisdiction": "CA"}),
                    _chunk("doc3", "s3", {"jurisdiction": "EU"}),
                ]
            ),
            _show_indicator=False,
        )
        assert round(metric.score, 4) == round(2 / 3, 4)
        assert metric.is_successful() is False

    def test_missing_metadata_fails_by_default(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values="A"
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", None),  # no metadata
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 0.5
        assert metric.is_successful() is False

    def test_missing_metadata_passes_when_flag_false(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id",
            allowed_values="A",
            missing_metadata_fails=False,
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", None),  # no metadata, ignored
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_missing_field_in_metadata_fails(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values="A"
        )
        # chunk has metadata but key is absent
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"other_field": "x"}),
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 0.0
        assert metric.is_successful() is False

    def test_empty_retrieval_context_passes(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values="A"
        )
        metric.measure(_test_case([]), _show_indicator=False)
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_plain_string_chunks_fail_when_missing_fails(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id",
            allowed_values="A",
            missing_metadata_fails=True,
        )
        metric.measure(
            _test_case(["plain string chunk"]),
            _show_indicator=False,
        )
        assert metric.score == 0.0
        assert metric.is_successful() is False

    def test_plain_string_chunks_pass_when_missing_ignored(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id",
            allowed_values="A",
            missing_metadata_fails=False,
        )
        metric.measure(
            _test_case(["plain string chunk"]),
            _show_indicator=False,
        )
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_strict_mode_zeroes_partial_score(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id",
            allowed_values="A",
            strict_mode=True,
        )
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", {"tenant_id": "B"}),
                ]
            ),
            _show_indicator=False,
        )
        assert metric.score == 0
        assert metric.is_successful() is False

    def test_threshold_partial_pass(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id",
            allowed_values="A",
            threshold=0.6,
        )
        # 2 of 3 compliant -> 0.67 >= 0.6 -> pass
        metric.measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", {"tenant_id": "A"}),
                    _chunk("doc3", "s3", {"tenant_id": "B"}),
                ]
            ),
            _show_indicator=False,
        )
        assert round(metric.score, 2) == 0.67
        assert metric.is_successful() is True

    def test_requires_filter_field(self):
        with pytest.raises(ValueError):
            FilterCorrectnessMetric(filter_field="", allowed_values="A")

    def test_no_reason_when_include_reason_false(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id",
            allowed_values="A",
            include_reason=False,
        )
        metric.measure(
            _test_case([_chunk("doc1", "s1", {"tenant_id": "B"})]),
            _show_indicator=False,
        )
        assert metric.reason is None

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = FilterCorrectnessMetric(
            filter_field="tenant_id", allowed_values="A"
        )
        score = await metric.a_measure(
            _test_case(
                [
                    _chunk("doc1", "s1", {"tenant_id": "A"}),
                    _chunk("doc2", "s2", {"tenant_id": "B"}),
                ]
            ),
            _show_indicator=False,
        )
        assert score == 0.5
        assert metric.is_successful() is False
