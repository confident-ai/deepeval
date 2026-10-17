import json

import pytest

from deepeval.metrics.community.trace_divergence import (
    DIVERGENCE_KINDS,
    TRACE_PROJECTION_VERSION,
    align,
    project,
)


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


def test_identical_traces_are_aligned():
    result = align([SEARCH, OPEN], [SEARCH, OPEN])
    assert result.aligned
    assert result.first_divergence is None
    assert result.divergence_ratio == 0.0


def test_encoding_only_differences_are_aligned():
    baseline = [step("search", q="x", limit=10)]
    candidate = [
        {
            "name": "search",
            "args": {"limit": 10.0, "q": "x"},
            "kind": "tool",
        }
    ]
    assert align(baseline, candidate).aligned


def test_same_tool_with_different_arguments_is_arg_change():
    result = align([SEARCH, OPEN], [step("search", q="2025"), OPEN])
    assert not result.aligned
    assert result.first_divergence == 0
    assert result.divergence_kind == "arg_change"


def test_same_length_substitution_that_rejoins_reports_resync():
    result = align(
        [SEARCH, OPEN, SUMMARIZE],
        [step("lookup", q="revenue 2024"), OPEN, SUMMARIZE],
    )
    assert not result.aligned
    assert result.divergence_kind == "tool_change"
    assert result.resync_at == 1
    assert result.divergence_ratio == pytest.approx(1 / 3)


def test_both_sides_can_advance_before_rejoining():
    result = align(
        [SEARCH, step("baseline_only"), OPEN, SUMMARIZE],
        [step("candidate_a"), step("candidate_b"), OPEN, SUMMARIZE],
    )
    assert not result.aligned
    assert result.divergence_kind == "tool_change"
    assert result.resync_at == 2


def test_reorder_is_localized_without_assuming_independence():
    result = align(
        [SEARCH, OPEN, SUMMARIZE],
        [OPEN, SEARCH, SUMMARIZE],
    )
    assert not result.aligned
    assert result.divergence_kind == "order_change"
    assert result.resync_at == 2
    assert result.reordered


def test_retry_that_rejoins_reports_resync():
    result = align(
        [SEARCH, OPEN, SUMMARIZE, EMAIL],
        [
            SEARCH,
            step("search", event_id="retry", q="revenue 2024", retry=1),
            OPEN,
            SUMMARIZE,
            EMAIL,
        ],
    )
    assert not result.aligned
    assert result.divergence_kind == "extra"
    assert result.resync_at == 2
    assert result.unmatched_candidate == ["retry"]
    assert result.divergence_ratio == 0.2


def test_path_change_does_not_report_recovery():
    candidate = [SEARCH, step("ask_human"), step("wait")]
    result = align([SEARCH, OPEN, SUMMARIZE], candidate)
    assert result.divergence_kind == "tool_change"
    assert result.resync_at is None


@pytest.mark.parametrize(
    ("baseline", "candidate", "kind", "unmatched"),
    [
        ([SEARCH, OPEN], [SEARCH], "absent", ["open_document"]),
        ([SEARCH], [SEARCH, EMAIL], "extra", ["send_email"]),
    ],
)
def test_trailing_steps_are_classified(baseline, candidate, kind, unmatched):
    result = align(baseline, candidate)
    assert result.divergence_kind == kind
    observed = (
        result.unmatched_baseline
        if kind == "absent"
        else result.unmatched_candidate
    )
    assert observed == unmatched


def test_duplicate_steps_are_counted_by_occurrence():
    first = step("poll", event_id="poll-1")
    duplicate = step("poll", event_id="poll-2")
    result = align([first], [first, duplicate])
    assert result.divergence_kind == "extra"
    assert result.unmatched_candidate == ["poll-2"]


def test_result_is_json_serializable_and_versioned():
    payload = align([SEARCH], [SEARCH]).as_dict()
    assert payload["projection_version"] == TRACE_PROJECTION_VERSION
    assert set(DIVERGENCE_KINDS) == {
        "arg_change",
        "tool_change",
        "order_change",
        "absent",
        "extra",
    }
    json.dumps(payload)


def test_projection_supports_deepeval_shaped_objects():
    class ToolCall:
        def __init__(self):
            self.name = "search"
            self.input_parameters = {"q": "x"}
            self.id = "call-1"

    assert project([ToolCall()])[0].event_id == "call-1"


@pytest.mark.parametrize(
    "bad_step",
    [
        {"name": "", "args": {}},
        {"name": "search", "args": {1: "not-json-object"}},
        {"name": "search", "args": {"bad": object()}},
    ],
)
def test_malformed_projection_fails_closed(bad_step):
    with pytest.raises(ValueError):
        project([bad_step])


def test_invalid_alignment_options_fail_closed():
    with pytest.raises(ValueError):
        align([], [], lookahead=0)
