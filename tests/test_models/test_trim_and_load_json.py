"""
Regression tests for trim_and_load_json corrupting valid JSON string values
(GitHub issue #2770).

The function unconditionally ran ``re.sub(r",\\s*([\\]}])", r"\\1", jsonStr)``
on the entire JSON string *before* parsing, so it could not distinguish a
structural trailing comma from a comma that legitimately appears inside a
string value.  PR #2701 fixed this in ``deepeval/metrics/utils.py`` and
``deepeval/dataset/utils.py``, but the third copy in
``deepeval/models/llms/utils.py`` was missed.

The fix: try ``json.loads`` first; only fall back to the trailing-comma
regex when the direct parse raises ``JSONDecodeError``.
"""

import pytest

from deepeval.models.llms.utils import trim_and_load_json
from deepeval.errors import DeepEvalError


class TestTrimAndLoadJsonPreservesStringValues:
    """Valid JSON must parse unchanged — commas inside strings are safe."""

    def test_comma_before_brace_in_string_value(self):
        """The exact reproduction from issue #2770."""
        result = trim_and_load_json('{"reason": "score is 1,} good"}')
        assert result == {"reason": "score is 1,} good"}

    def test_comma_before_bracket_in_string_value(self):
        result = trim_and_load_json('{"reason": "items: a,] done"}')
        assert result == {"reason": "items: a,] done"}

    def test_multiple_commas_in_string(self):
        result = trim_and_load_json('{"verdict": "x, y, z]", "score": 0.8}')
        assert result == {"verdict": "x, y, z]", "score": 0.8}

    def test_clean_json_still_works(self):
        result = trim_and_load_json('{"score": 1, "reason": "good"}')
        assert result == {"score": 1, "reason": "good"}


class TestTrimAndLoadJsonTrailingCommaFallback:
    """The trailing-comma regex must still work as a fallback."""

    def test_trailing_comma_before_brace(self):
        result = trim_and_load_json('{"score": 1, "reason": "ok",}')
        assert result == {"score": 1, "reason": "ok"}

    def test_trailing_comma_before_bracket(self):
        result = trim_and_load_json('{"items": [1, 2, 3,]}')
        assert result == {"items": [1, 2, 3]}


class TestTrimAndLoadJsonInvalidInput:
    """Truly invalid JSON must raise DeepEvalError."""

    def test_garbage_input(self):
        with pytest.raises(DeepEvalError):
            trim_and_load_json("not json at all")

    def test_empty_string(self):
        with pytest.raises(DeepEvalError):
            trim_and_load_json("")
