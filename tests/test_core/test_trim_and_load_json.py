"""Regression tests for trimAndLoadJson trailing-comma handling.

The metrics, dataset, and models copies stripped ``,\\s*`` before a closing
``]``/``}`` unconditionally, which corrupted valid JSON whose string values
happened to contain ``", ]"`` or ``", }"``. The cleanup must only run as a
fallback when a direct parse fails.
"""

import json

import pytest

from deepeval.errors import DeepEvalError
from deepeval.dataset.utils import trimAndLoadJson as trim_dataset
from deepeval.metrics.utils import trimAndLoadJson as trim_metrics
from deepeval.models.llms.utils import trim_and_load_json as trim_models

# The metrics and dataset copies raise ``ValueError`` on invalid JSON; the
# models copy raises ``DeepEvalError``. Group them where the distinction matters.
TRIM_FNS = [trim_metrics, trim_dataset]
ALL_TRIM_FNS = [trim_metrics, trim_dataset, trim_models]


@pytest.mark.parametrize("trim", ALL_TRIM_FNS)
@pytest.mark.parametrize(
    "raw",
    [
        '{"reason": "found items A, B, ] then stopped"}',
        '{"note": "the set is {x, y, } here"}',
    ],
)
def test_valid_json_string_values_are_preserved(trim, raw):
    assert trim(raw) == json.loads(raw)


@pytest.mark.parametrize("trim", ALL_TRIM_FNS)
def test_trailing_comma_is_still_stripped(trim):
    assert trim('{"a": [1, 2, ]}') == {"a": [1, 2]}


@pytest.mark.parametrize("trim", TRIM_FNS)
def test_invalid_json_still_raises(trim):
    with pytest.raises(ValueError):
        trim("not json at all {[")


def test_models_copy_invalid_json_raises():
    with pytest.raises(DeepEvalError):
        trim_models("not json at all {[")
