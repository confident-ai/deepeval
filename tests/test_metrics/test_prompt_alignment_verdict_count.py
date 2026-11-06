"""Regression tests for PromptAlignmentMetric's verdict-count validation.

The verdicts template asks the judge for exactly one verdict per prompt
instruction, but the schema accepts a list of any length. Previously an empty
list scored 1, and a partial list was scored only over the verdicts returned,
so a malformed judge response could pass assert_test() even in strict_mode.

These tests use a stub model that returns fixed verdicts, so they run without
an LLM provider API key. They cover sync and async execution, and both schema
(structured) and raw JSON judge responses.
"""

import json

import pytest

from deepeval import assert_test, evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import PromptAlignmentMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

INSTRUCTIONS = [
    "Use uppercase.",
    "Output valid JSON.",
    "Include a citations array.",
]
TEST_CASE = LLMTestCase(
    input="Provide the report.", actual_output="hello there"
)


class _FixedVerdictsLLM(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM returning fixed verdicts, no provider calls."""

    def __init__(self, verdicts, structured: bool):
        self.verdicts = verdicts
        self.structured = structured
        super().__init__("fixed-verdicts-llm")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None):
        if schema is not None and "verdicts" in schema.model_fields:
            data = {"verdicts": self.verdicts}
        else:
            data = {"reason": "Fixed reason."}
        if self.structured and schema is not None:
            return schema(**data)
        return json.dumps(data)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self, *args, **kwargs) -> str:
        return "fixed-verdicts-llm"


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("count", [0, 1, 4])
def test_incomplete_or_excess_verdicts_must_not_pass(
    count, structured, run_async
):
    model = _FixedVerdictsLLM([{"verdict": "yes"}] * count, structured)
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS, model=model, strict_mode=True
    )
    with pytest.raises(ValueError, match="verdict"):
        assert_test(TEST_CASE, [metric], run_async=run_async)


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("verdict", ["yes", "no"])
def test_complete_verdicts_retain_expected_gate(verdict, structured, run_async):
    model = _FixedVerdictsLLM(
        [{"verdict": verdict, "reason": "Fixed control."}] * len(INSTRUCTIONS),
        structured,
    )
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS, model=model, strict_mode=True
    )
    if verdict == "no":
        with pytest.raises(AssertionError):
            assert_test(TEST_CASE, [metric], run_async=run_async)
    else:
        assert_test(TEST_CASE, [metric], run_async=run_async)


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("count", [0, 1])
def test_mismatched_count_is_recorded_as_errored_when_errors_ignored(
    count, run_async
):
    """With ignore_errors=True the run continues, but the metric must be
    recorded as errored rather than as a pass."""
    model = _FixedVerdictsLLM([{"verdict": "yes"}] * count, structured=True)
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS, model=model, strict_mode=True
    )
    result = evaluate(
        test_cases=[TEST_CASE],
        metrics=[metric],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=True),
    )
    metric_data = result.test_results[0].metrics_data[0]
    assert metric_data.error is not None
    assert "verdict" in metric_data.error
    assert metric_data.success is False
    assert result.test_results[0].success is False
