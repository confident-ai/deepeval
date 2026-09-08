"""Regression tests for trimAndLoadJson / trim_and_load_json.

Covers trailing-comma handling (PR #2701) and <think> tag stripping for
reasoning models like DeepSeek-R1, QwQ, and Nemotron (PR #2410 follow-up).
"""

import json

import pytest

from deepeval.dataset.utils import trimAndLoadJson as trim_dataset
from deepeval.metrics.utils import trimAndLoadJson as trim_metrics
from deepeval.models.llms.utils import trim_and_load_json as trim_models

TRIM_FNS = [trim_metrics, trim_dataset, trim_models]


# --- Existing trailing-comma tests (from PR #2701) ---


@pytest.mark.parametrize("trim", TRIM_FNS)
@pytest.mark.parametrize(
    "raw",
    [
        '{"reason": "found items A, B, ] then stopped"}',
        '{"note": "the set is {x, y, } here"}',
    ],
)
def test_valid_json_string_values_are_preserved(trim, raw):
    assert trim(raw) == json.loads(raw)


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_trailing_comma_is_still_stripped(trim):
    assert trim('{"a": [1, 2, ]}') == {"a": [1, 2]}


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_invalid_json_still_raises(trim):
    with pytest.raises((ValueError, Exception)):
        trim("not json at all {[")


# --- <think> tag tests ---


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_think_block_with_braces_parses_correctly(trim):
    """A <think> block containing braces should not confuse JSON extraction."""
    raw = (
        "<think>\nLet me reason about this. The expected format is "
        '{"score": 7} but I need to evaluate carefully.\n</think>\n'
        '{"score": 9, "reason": "correct answer"}'
    )
    result = trim(raw)
    assert result == {"score": 9, "reason": "correct answer"}


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_think_tag_in_json_string_value_is_preserved(trim):
    """Valid JSON whose string value contains literal <think> text must not
    be corrupted — the stripping only runs in the fallback path."""
    raw = '{"note": "the <think>reasoning</think> was good"}'
    assert trim(raw) == json.loads(raw)


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_closing_tag_only_variant(trim):
    """Some chat templates inject the opening <think> so the model only emits
    the closing </think>.  Everything before it should be stripped."""
    raw = (
        "Let me think about this step by step.\n"
        "</think>\n"
        '{"score": 5, "reason": "partially correct"}'
    )
    result = trim(raw)
    assert result == {"score": 5, "reason": "partially correct"}


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_think_block_without_braces_already_works(trim):
    """A brace-free <think> block before JSON already parses without the
    fallback — the existing find('{') extraction handles it."""
    raw = "<think>simple reasoning here</think>\n" '{"score": 10}'
    assert trim(raw) == {"score": 10}


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_nested_json_in_think_block(trim):
    """A <think> block with nested JSON-like structures should still allow
    the actual JSON after the block to be parsed."""
    raw = (
        '<think>\nComparing {"a": 1} vs {"b": 2} to decide.\n</think>\n'
        '```json\n{"score": 8, "reason": "good"}\n```'
    )
    result = trim(raw)
    assert result == {"score": 8, "reason": "good"}
