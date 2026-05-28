"""Regression tests for trimAndLoadJson trailing-comma handling.

Both the metrics and dataset copies stripped ``,\\s*`` before a closing ``]``/``}``
unconditionally, which corrupted valid JSON whose string values happened to
contain ``", ]"`` or ``", }"``. The cleanup must only run as a fallback when a
direct parse fails.
"""

import json

import pytest

from deepeval.dataset.utils import trimAndLoadJson as trim_dataset
from deepeval.metrics.utils import trimAndLoadJson as trim_metrics

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
