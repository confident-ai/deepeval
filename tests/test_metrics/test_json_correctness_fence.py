"""Offline tests for JsonCorrectnessMetric's code-fence handling.

JsonCorrectnessMetric previously fed ``actual_output`` straight into
``model_validate_json``. LLM eval outputs are very commonly wrapped in a
markdown code fence (````` ```json ... ``` `````); in that case a *valid* JSON
was scored as invalid — a false negative. We now strip an enclosing fence
before validation.

These tests are fully offline: ``async_mode=False`` + ``include_reason=False``
so no LLM interaction is required, and a tiny local pydantic schema is used.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deepeval.metrics import JsonCorrectnessMetric
from deepeval.metrics.json_correctness.json_correctness import (
    _strip_code_fence,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


def _stub_model() -> DeepEvalBaseLLM:
    m = MagicMock(spec=DeepEvalBaseLLM)
    m.get_model_name.return_value = "mock-llm"
    m.supports_multimodal.return_value = False
    return m


def _make_metric() -> JsonCorrectnessMetric:
    return JsonCorrectnessMetric(
        expected_schema=User,
        model=_stub_model(),
        async_mode=False,
        include_reason=False,
        strict_mode=False,
        threshold=0.5,
    )


def _case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual_output)


# ---------------------------------------------------------------------------
# 1. The stripping helper itself.
# ---------------------------------------------------------------------------


class TestStripCodeFence:
    def test_strips_json_fence(self):
        out = _strip_code_fence('```json\n{"name":"A","age":1}\n```')
        assert out == '{"name":"A","age":1}'

    def test_strips_bare_fence(self):
        out = _strip_code_fence('```\n{"name":"A","age":1}\n```')
        assert out == '{"name":"A","age":1}'

    def test_strips_fence_with_leading_blank_lines(self):
        out = _strip_code_fence('\n\n```json\n{"name":"A","age":1}\n```\n')
        assert out == '{"name":"A","age":1}'

    def test_plain_json_passthrough(self):
        s = '{"name":"A","age":1}'
        assert _strip_code_fence(s) == s

    def test_non_json_text_passthrough(self):
        s = "the answer is no"
        assert _strip_code_fence(s) == s

    def test_fence_not_at_boundaries_passthrough(self):
        # ````` ``` ``` ``^ inside a sentence should not be treated as a fence.
        s = 'prefix\n```json\n{"age":1}\n```'
        assert _strip_code_fence(s) == s


# ---------------------------------------------------------------------------
# 2. End-to-end score through the metric (offline).
# ---------------------------------------------------------------------------


class TestJsonCorrectnessFence:
    def test_fenced_valid_json_now_passes(self):
        m = _make_metric()
        m.measure(_case('```json\n{"name":"Alice","age":30}\n```'))
        assert m.score == 1.0

    def test_plain_valid_json_still_passes(self):
        m = _make_metric()
        m.measure(_case('{"name":"Alice","age":30}'))
        assert m.score == 1.0

    def test_plain_invalid_json_still_fails(self):
        m = _make_metric()
        m.measure(_case('{"name":"Alice","age":"not-an-int"}'))
        assert m.score == 0.0

    def test_fenced_invalid_json_still_fails(self):
        # The strip must not turn an invalid body into a pass.
        m = _make_metric()
        m.measure(_case('```json\n{"name":"Alice","age":"x"}\n```'))
        assert m.score == 0.0

    def test_non_json_text_unchanged_fails(self):
        m = _make_metric()
        m.measure(_case("sorry, I can't do that"))
        assert m.score == 0.0
