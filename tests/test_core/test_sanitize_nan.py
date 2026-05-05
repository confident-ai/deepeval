"""Tests for NaN / Infinity / -Infinity sanitization.

Validates that non-finite floats are replaced with None before JSON
serialization so payloads sent to the backend are always valid JSON.
"""

import json
import math
import pytest

from deepeval.tracing.utils import (
    make_json_serializable,
    make_json_serializable_for_metadata,
)
from deepeval.confident.api import _sanitize_body

# ---------------------------------------------------------------------------
# make_json_serializable
# ---------------------------------------------------------------------------


class TestMakeJsonSerializable:
    """make_json_serializable must neutralise non-finite floats."""

    def test_nan_replaced_with_none(self):
        assert make_json_serializable(float("nan")) is None

    def test_inf_replaced_with_none(self):
        assert make_json_serializable(float("inf")) is None

    def test_neg_inf_replaced_with_none(self):
        assert make_json_serializable(float("-inf")) is None

    def test_normal_float_preserved(self):
        assert make_json_serializable(3.14) == 3.14

    def test_zero_float_preserved(self):
        assert make_json_serializable(0.0) == 0.0

    def test_negative_float_preserved(self):
        assert make_json_serializable(-1.5) == -1.5

    def test_nan_inside_dict(self):
        result = make_json_serializable({"score": float("nan"), "ok": 1.0})
        assert result["score"] is None
        assert result["ok"] == 1.0

    def test_nan_inside_list(self):
        result = make_json_serializable([1.0, float("nan"), float("inf")])
        assert result == [1.0, None, None]

    def test_deeply_nested(self):
        obj = {"level1": {"level2": [{"value": float("nan")}, {"value": 42.0}]}}
        result = make_json_serializable(obj)
        assert result["level1"]["level2"][0]["value"] is None
        assert result["level1"]["level2"][1]["value"] == 42.0

    def test_result_is_valid_json(self):
        """The whole point: the output must survive json.dumps / json.loads."""
        payload = {
            "score": float("nan"),
            "threshold": 0.5,
            "cost": float("inf"),
            "neg": float("-inf"),
            "nested": {"v": float("nan")},
            "items": [float("inf"), 1.0],
        }
        sanitized = make_json_serializable(payload)
        roundtripped = json.loads(json.dumps(sanitized))
        assert roundtripped["score"] is None
        assert roundtripped["threshold"] == 0.5
        assert roundtripped["cost"] is None
        assert roundtripped["neg"] is None
        assert roundtripped["nested"]["v"] is None
        assert roundtripped["items"] == [None, 1.0]

    def test_other_types_unaffected(self):
        result = make_json_serializable(
            {"s": "hello", "i": 42, "b": True, "n": None}
        )
        assert result == {"s": "hello", "i": 42, "b": True, "n": None}


# ---------------------------------------------------------------------------
# make_json_serializable_for_metadata
# ---------------------------------------------------------------------------


class TestMakeJsonSerializableForMetadata:
    """metadata variant preserves finite primitives, replaces non-finite with None.

    Previously this helper stringified every primitive (``True`` → ``"True"``,
    ``3.14`` → ``"3.14"``), which destroyed type fidelity for user metadata.
    The contract is now: primitives pass through, non-finite floats become
    None, everything else gets serialized recursively.
    """

    def test_nan_replaced_with_none(self):
        assert make_json_serializable_for_metadata(float("nan")) is None

    def test_inf_replaced_with_none(self):
        assert make_json_serializable_for_metadata(float("inf")) is None

    def test_neg_inf_replaced_with_none(self):
        assert make_json_serializable_for_metadata(float("-inf")) is None

    def test_finite_float_preserved(self):
        assert make_json_serializable_for_metadata(3.14) == 3.14

    def test_int_preserved(self):
        assert make_json_serializable_for_metadata(42) == 42

    def test_bool_preserved(self):
        assert make_json_serializable_for_metadata(True) is True
        assert make_json_serializable_for_metadata(False) is False

    def test_none_preserved(self):
        assert make_json_serializable_for_metadata(None) is None

    def test_nan_inside_dict(self):
        result = make_json_serializable_for_metadata(
            {"cost": float("nan"), "ok": 2.0}
        )
        assert result["cost"] is None
        assert result["ok"] == 2.0

    def test_mixed_primitives_inside_dict(self):
        """Regression guard: every primitive type must round-trip with its
        native JSON type intact."""
        result = make_json_serializable_for_metadata(
            {
                "flag": True,
                "count": 7,
                "ratio": 0.25,
                "missing": None,
                "label": "ok",
            }
        )
        assert result == {
            "flag": True,
            "count": 7,
            "ratio": 0.25,
            "missing": None,
            "label": "ok",
        }


# ---------------------------------------------------------------------------
# _sanitize_body  (API-layer catch-all)
# ---------------------------------------------------------------------------


class TestSanitizeBody:
    """_sanitize_body is the last line of defence before HTTP serialization."""

    def test_nan(self):
        assert _sanitize_body(float("nan")) is None

    def test_inf(self):
        assert _sanitize_body(float("inf")) is None

    def test_neg_inf(self):
        assert _sanitize_body(float("-inf")) is None

    def test_normal_float(self):
        assert _sanitize_body(3.14) == 3.14

    def test_flat_dict(self):
        result = _sanitize_body({"a": float("nan"), "b": 1.0, "c": "hi"})
        assert result == {"a": None, "b": 1.0, "c": "hi"}

    def test_nested_dict(self):
        result = _sanitize_body({"outer": {"inner": float("inf")}})
        assert result == {"outer": {"inner": None}}

    def test_list(self):
        result = _sanitize_body([float("nan"), 1, "x", float("-inf")])
        assert result == [None, 1, "x", None]

    def test_tuple_becomes_list(self):
        result = _sanitize_body((float("nan"), 2.0))
        assert result == [None, 2.0]

    def test_non_numeric_passthrough(self):
        assert _sanitize_body("hello") == "hello"
        assert _sanitize_body(42) == 42
        assert _sanitize_body(True) is True
        assert _sanitize_body(None) is None

    def test_full_trace_shaped_payload(self):
        """Simulate a realistic trace payload with problematic values."""
        payload = {
            "uuid": "abc-123",
            "baseSpans": [],
            "llmSpans": [
                {
                    "uuid": "span-1",
                    "inputTokenCount": float("nan"),
                    "outputTokenCount": float("inf"),
                    "costPerInputToken": float("-inf"),
                    "costPerOutputToken": 0.00003,
                    "metricsData": [
                        {
                            "name": "faithfulness",
                            "score": float("nan"),
                            "threshold": 0.7,
                            "evaluationCost": float("inf"),
                        }
                    ],
                }
            ],
            "startTime": "2025-01-01T00:00:00Z",
            "endTime": "2025-01-01T00:00:01Z",
        }
        sanitized = _sanitize_body(payload)
        span = sanitized["llmSpans"][0]
        assert span["inputTokenCount"] is None
        assert span["outputTokenCount"] is None
        assert span["costPerInputToken"] is None
        assert span["costPerOutputToken"] == 0.00003
        metric = span["metricsData"][0]
        assert metric["score"] is None
        assert metric["threshold"] == 0.7
        assert metric["evaluationCost"] is None

        roundtripped = json.loads(json.dumps(sanitized))
        assert roundtripped is not None
