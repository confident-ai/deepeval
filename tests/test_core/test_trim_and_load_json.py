"""Regression tests for trimAndLoadJson trailing-comma handling.

The metrics and dataset copies stripped ``,\\s*`` before a closing ``]``/``}``
unconditionally, which corrupted valid JSON whose string values happened to
contain ``", ]"`` or ``", }"``. PR #2701 fixed those two copies; this file
also covers the third copy in ``deepeval.models.llms.utils`` (issue #2770).
The cleanup must only run as a fallback when a direct parse fails.
"""

import json

import pytest

from deepeval.dataset.utils import trimAndLoadJson as trim_dataset
from deepeval.metrics.utils import trimAndLoadJson as trim_metrics
from deepeval.models.llms.utils import trim_and_load_json as trim_models

TRIM_FNS = [trim_metrics, trim_dataset]


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
    with pytest.raises(ValueError):
        trim("not json at all {[")


# --- models/llms/utils.py copy (issue #2770) ---


@pytest.mark.parametrize(
    "raw",
    [
        '{"reason": "score is 1,} good"}',
        '{"note": "the set is {x, y, } here"}',
        '{"items": "found A, B, ] then done"}',
    ],
)
def test_models_utils_valid_json_string_values_are_preserved(raw):
    assert trim_models(raw) == json.loads(raw)


def test_models_utils_trailing_comma_is_still_stripped():
    assert trim_models('{"a": [1, 2, ]}') == {"a": [1, 2]}


def test_models_utils_markdown_fence_is_stripped():
    fenced = '```json\n{"verdict": "yes"}\n```'
    assert trim_models(fenced) == {"verdict": "yes"}
