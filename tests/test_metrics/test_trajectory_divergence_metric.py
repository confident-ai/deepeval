import pytest

from deepeval.metrics.community import TrajectoryDivergenceMetric
from deepeval.metrics.community.trace_divergence import align
from deepeval.test_case import LLMTestCase


def step(name, event_id=None, **arguments):
    return {
        "kind": "tool",
        "name": name,
        "args": arguments,
        "id": event_id or name,
    }


SEARCH = step("search", q="revenue 2024")
OPEN = step("open_document", document_id=7)
SUMMARIZE = step("summarize", style="brief")
EMAIL = step("send_email", to="cfo@example.com")


def _make_test_case():
    return LLMTestCase(input="compare the two runs", actual_output="done")


def aligned_traces():
    return [SEARCH, OPEN, SUMMARIZE, EMAIL], [SEARCH, OPEN, SUMMARIZE, EMAIL]


def test_identical_traces_score_perfect_and_pass():
    metric = TrajectoryDivergenceMetric(*aligned_traces())
    score = metric.measure(_make_test_case())
    assert score == 1.0
    assert metric.success is True
    assert "aligned across all 4 steps" in metric.reason


def test_identical_traces_score_perfect_in_async_mode():
    metric = TrajectoryDivergenceMetric(*aligned_traces())
    score = metric.measure(_make_test_case())
    assert score == 1.0


def test_arg_change_is_scored_as_divergence():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN], [step("search", q="2025"), OPEN]
    )
    score = metric.measure(_make_test_case())
    assert score < 1.0
    assert metric.success is False
    assert metric.reason is not None
    assert "step 1" in metric.reason
    assert "different arguments" in metric.reason


def test_tool_change_reason_names_both_tools():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN], [SEARCH, step("ask_human"), step("wait")]
    )
    metric.measure(_make_test_case())
    assert metric.alignment_result.divergence_kind == "tool_change"
    assert "open_document" in metric.reason
    assert "ask_human" in metric.reason


def test_reorder_is_a_divergence():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN, SUMMARIZE], [OPEN, SEARCH, SUMMARIZE]
    )
    metric.measure(_make_test_case())
    assert metric.alignment_result.divergence_kind == "order_change"
    assert "different order" in metric.reason
    assert metric.score < 1.0
    assert metric.success is False


def test_retry_that_rejoins_is_localized_not_divergence_to_end():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN, SUMMARIZE, EMAIL],
        [
            SEARCH,
            step("search", event_id="retry", q="revenue 2024", retry=1),
            OPEN,
            SUMMARIZE,
            EMAIL,
        ],
    )
    metric.measure(_make_test_case())
    result = metric.alignment_result
    assert result.divergence_kind == "extra"
    assert result.resync_at == 2
    assert metric.score == pytest.approx(0.8)
    assert "resynchronize at step 3" in metric.reason
    assert metric.success is False


def test_lower_threshold_tolerates_recovered_divergence():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN, SUMMARIZE, EMAIL],
        [
            SEARCH,
            step("search", event_id="retry", q="revenue 2024", retry=1),
            OPEN,
            SUMMARIZE,
            EMAIL,
        ],
        threshold=0.7,
    )
    metric.measure(_make_test_case())
    assert metric.success is True


def test_unrecovered_path_change_reason_mentions_no_resync():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN, SUMMARIZE],
        [SEARCH, step("ask_human"), step("wait")],
    )
    metric.measure(_make_test_case())
    assert metric.alignment_result.resync_at is None
    assert "do not resynchronize" in metric.reason


def test_absent_trailing_step_is_classified():
    metric = TrajectoryDivergenceMetric([SEARCH, OPEN], [SEARCH])
    metric.measure(_make_test_case())
    assert metric.alignment_result.divergence_kind == "absent"
    assert "absent from the candidate trace" in metric.reason


def test_extra_trailing_step_is_classified():
    metric = TrajectoryDivergenceMetric([SEARCH], [SEARCH, EMAIL])
    metric.measure(_make_test_case())
    assert metric.alignment_result.divergence_kind == "extra"
    assert "inserts an extra step" in metric.reason


def test_score_only_mode_leaves_success_none():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN], [step("search", q="2025"), OPEN], threshold=None
    )
    metric.measure(_make_test_case())
    assert metric.success is None
    assert metric.score < 1.0


def test_strict_mode_is_binary():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN], [step("search", q="2025"), OPEN], strict_mode=True
    )
    metric.measure(_make_test_case())
    assert metric.score == 0
    assert metric.success is False

    strict_aligned = TrajectoryDivergenceMetric(
        *aligned_traces(), strict_mode=True
    )
    strict_aligned.measure(_make_test_case())
    assert strict_aligned.score == 1.0
    assert strict_aligned.success is True


def test_include_reason_false_returns_none_reason():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN], [SEARCH], include_reason=False
    )
    metric.measure(_make_test_case())
    assert metric.reason is None


def test_alignment_result_is_exposed_for_localization():
    metric = TrajectoryDivergenceMetric(
        [SEARCH, OPEN, SUMMARIZE, EMAIL],
        [SEARCH, step("open_document", document_id=99), SUMMARIZE, EMAIL],
    )
    metric.measure(_make_test_case())
    result = metric.alignment_result
    assert result.matched_prefix_len == 1
    assert result.first_divergence == 1
    assert result.divergence_kind == "arg_change"
    assert result.baseline_len == 4
    assert result.candidate_len == 4


def test_verbose_logs_are_populated():
    metric = TrajectoryDivergenceMetric(*aligned_traces())
    metric.measure(_make_test_case())
    assert metric.verbose_logs is not None
    assert "Alignment" in metric.verbose_logs


def test_empty_traces_align():
    metric = TrajectoryDivergenceMetric([], [])
    score = metric.measure(_make_test_case())
    assert score == 1.0
    assert metric.success is True


def test_wrapper_consistency_with_raw_align():
    baseline, candidate = [SEARCH, OPEN], [SEARCH]
    metric = TrajectoryDivergenceMetric(baseline, candidate)
    metric.measure(_make_test_case())
    result = align(baseline, candidate)
    assert metric.alignment_result == result


def test_accepts_span_like_objects():
    class SpanLike:
        def __init__(self, name, args):
            self.name = name
            self.args = args
            self.id = name

    baseline = [SpanLike("search", {"q": "x"})]
    candidate = [SpanLike("search", {"q": "y"})]
    metric = TrajectoryDivergenceMetric(baseline, candidate)
    metric.measure(_make_test_case())
    assert metric.score < 1.0
