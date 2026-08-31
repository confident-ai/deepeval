"""Regression tests for trimAndLoadJson trailing-comma handling.

The metrics, dataset, and models/llms copies stripped ``,\\s*`` before a closing
``]``/``}`` unconditionally, which corrupted valid JSON whose string values
happened to contain ``", ]"`` or ``", }"``. The cleanup must only run as a
fallback when a direct parse fails.
"""

import json

import pytest

from deepeval.dataset.utils import trimAndLoadJson as trim_dataset
from deepeval.errors import DeepEvalError
from deepeval.metrics.utils import trimAndLoadJson as trim_metrics
from deepeval.models.llms.utils import trim_and_load_json as trim_models

# Each copy raises a different error type on unparseable input.
TRIM_FNS = [
    (trim_metrics, ValueError),
    (trim_dataset, ValueError),
    (trim_models, DeepEvalError),
]


@pytest.mark.parametrize("trim", [fn for fn, _ in TRIM_FNS])
@pytest.mark.parametrize(
    "raw",
    [
        '{"reason": "found items A, B, ] then stopped"}',
        '{"note": "the set is {x, y, } here"}',
        '{"a": 1}',
    ],
)
def test_valid_json_string_values_are_preserved(trim, raw):
    assert trim(raw) == json.loads(raw)


@pytest.mark.parametrize("trim", [fn for fn, _ in TRIM_FNS])
def test_trailing_comma_is_still_stripped(trim):
    assert trim('{"a": [1, 2, ]}') == {"a": [1, 2]}


@pytest.mark.parametrize("trim,exc", TRIM_FNS)
def test_invalid_json_still_raises(trim, exc):
    with pytest.raises(exc):
        trim("not json at all {[")
