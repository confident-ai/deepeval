"""Regression tests for ToolUseMetric's final-reason generation.

Two related defects:

- `include_reason` was accepted by the constructor but never consulted, so
  both final-reason LLM calls (tool selection and argument correctness) were
  made even when the user explicitly disabled them.
- The argument-correctness reason helpers rendered the tool-selection
  template (`get_tool_selection_final_reason`) instead of the dedicated
  `get_tool_argument_final_reason` template, so the argument half of every
  reason was written by a prompt about tool choice.

These tests run with stub models, so they need no LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import ToolUseMetric
from deepeval.metrics.tool_use.schema import (
    ArgumentCorrectnessScore,
    Reason,
    ToolSelectionScore,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn


class _StubLLM(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM that never calls out to a provider."""

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("generate should not be called in this test")

    async def a_generate(self, *args, **kwargs) -> str:
        raise AssertionError("a_generate should not be called in this test")

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


class _CapturingLLM(_StubLLM):
    """Records the prompt it was given and answers the reason schema."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, prompt, schema=None, *args, **kwargs):
        self.prompts.append(prompt)
        assert schema is Reason
        return Reason(reason="stub reason")

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema=schema, *args, **kwargs)


class _SchemaStubLLM(_StubLLM):
    """Answers the score prompts; the reason prompts must never come."""

    def generate(self, prompt, schema=None, *args, **kwargs):
        if schema is ToolSelectionScore:
            return ToolSelectionScore(score=1.0, reason="right tool")
        if schema is ArgumentCorrectnessScore:
            return ArgumentCorrectnessScore(score=1.0, reason="right args")
        raise AssertionError(
            f"unexpected schema requested: {getattr(schema, '__name__', schema)} "
            "(the reason prompts must be skipped when include_reason=False)"
        )

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema=schema, *args, **kwargs)


_AVAILABLE_TOOLS = [ToolCall(name="CheckDiscount")]
_SELECTION_SCORES = [ToolSelectionScore(score=1.0, reason="right tool")]
_ARGUMENT_SCORES = [ArgumentCorrectnessScore(score=1.0, reason="right args")]


def _metric(model=None, **kwargs) -> ToolUseMetric:
    metric = ToolUseMetric(
        available_tools=_AVAILABLE_TOOLS, model=model or _StubLLM(), **kwargs
    )
    metric.score = 1.0
    return metric


def test_sync_reason_helpers_respect_include_reason_false():
    metric = _metric(include_reason=False)
    assert (
        metric._generate_reason_for_tool_selection(
            _SELECTION_SCORES, multimodal=False
        )
        is None
    )
    assert (
        metric._generate_reason_for_argument_correctness(
            _ARGUMENT_SCORES, multimodal=False
        )
        is None
    )


def test_async_reason_helpers_respect_include_reason_false():
    metric = _metric(include_reason=False)
    assert (
        asyncio.run(
            metric._a_generate_reason_for_tool_selection(
                _SELECTION_SCORES, multimodal=False
            )
        )
        is None
    )
    assert (
        asyncio.run(
            metric._a_generate_reason_for_argument_correctness(
                _ARGUMENT_SCORES, multimodal=False
            )
        )
        is None
    )


def test_default_async_measure_skips_reasons_when_disabled():
    metric = ToolUseMetric(
        available_tools=_AVAILABLE_TOOLS,
        model=_SchemaStubLLM(),
        include_reason=False,
    )
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Do these shoes have a discount?"),
            Turn(
                role="assistant",
                content="Let me check that for you.",
                tools_called=[ToolCall(name="CheckDiscount")],
            ),
        ]
    )
    metric.measure(test_case, _show_indicator=False)

    assert metric.score is not None
    assert metric.reason is None


@pytest.mark.parametrize("use_async", [False, True])
def test_argument_correctness_reason_uses_argument_template(use_async):
    model = _CapturingLLM()
    metric = _metric(model=model)

    if use_async:
        asyncio.run(
            metric._a_generate_reason_for_argument_correctness(
                _ARGUMENT_SCORES, multimodal=False
            )
        )
    else:
        metric._generate_reason_for_argument_correctness(
            _ARGUMENT_SCORES, multimodal=False
        )

    assert len(model.prompts) == 1
    prompt = model.prompts[0]
    assert "Tool Argument Quality" in prompt
    assert "**Tool Selection** evaluation" not in prompt
