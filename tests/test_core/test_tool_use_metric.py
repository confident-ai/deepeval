import asyncio
import json

from deepeval.metrics.tool_use.schema import (
    ArgumentCorrectnessScore,
    Reason,
    ToolSelectionScore,
)
from deepeval.metrics.tool_use.tool_use import ToolUseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn


class _StubJudgeLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        tool_selection_score: float = 1.0,
        argument_correctness_score: float = 1.0,
    ):
        self.tool_selection_score = tool_selection_score
        self.argument_correctness_score = argument_correctness_score
        self.calls: list[str] = []

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return "stub-judge-llm"

    def _build_response(self, prompt: str, schema=None):
        schema_name = schema.__name__ if schema is not None else "raw"
        self.calls.append(schema_name)

        if schema is ToolSelectionScore:
            return ToolSelectionScore(
                score=self.tool_selection_score,
                reason="Appropriate tool selection decision.",
            )
        if schema is ArgumentCorrectnessScore:
            return ArgumentCorrectnessScore(
                score=self.argument_correctness_score,
                reason="Arguments evaluated.",
            )
        if schema is Reason:
            return Reason(reason="Summary reason.")

        if "Argument Correctness" in prompt:
            return json.dumps(
                {
                    "score": self.argument_correctness_score,
                    "reason": "Arguments evaluated.",
                }
            )
        if "Tool Selection" in prompt:
            return json.dumps(
                {
                    "score": self.tool_selection_score,
                    "reason": "Appropriate tool selection decision.",
                }
            )
        return json.dumps({"reason": "Summary reason."})

    def generate(self, prompt: str, schema=None):
        return self._build_response(prompt, schema=schema)

    async def a_generate(self, prompt: str, schema=None):
        return self._build_response(prompt, schema=schema)


def _available_tools():
    return [ToolCall(name="search", description="Search documentation")]


def _no_tool_call_conversation():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hi, how are you?"),
            Turn(role="assistant", content="I am doing well, thanks!"),
        ]
    )


def _tool_call_conversation():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Find the release notes."),
            Turn(
                role="assistant",
                content="Let me search for the release notes.",
                tools_called=[
                    ToolCall(
                        name="search",
                        input_parameters={"query": "release notes"},
                    )
                ],
            ),
        ]
    )


def test_tool_use_metric_no_tool_calls_sync():
    judge = _StubJudgeLLM(tool_selection_score=1.0)
    metric = ToolUseMetric(
        available_tools=_available_tools(),
        model=judge,
        async_mode=False,
    )

    score = metric.measure(_no_tool_call_conversation(), _show_indicator=False)

    assert score == 1.0
    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert metric.reason == "Summary reason."
    assert "ArgumentCorrectnessScore" not in judge.calls
    assert judge.calls.count("Reason") == 1


def test_tool_use_metric_no_tool_calls_async():
    judge = _StubJudgeLLM(tool_selection_score=1.0)
    metric = ToolUseMetric(
        available_tools=_available_tools(),
        model=judge,
        async_mode=True,
    )

    score = asyncio.run(
        metric.a_measure(_no_tool_call_conversation(), _show_indicator=False)
    )

    assert score == 1.0
    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert metric.reason == "Summary reason."
    assert "ArgumentCorrectnessScore" not in judge.calls
    assert judge.calls.count("Reason") == 1


def test_tool_use_metric_with_tool_calls_uses_min_score_sync_and_async():
    sync_judge = _StubJudgeLLM(
        tool_selection_score=1.0,
        argument_correctness_score=0.6,
    )
    sync_metric = ToolUseMetric(
        available_tools=_available_tools(),
        model=sync_judge,
        async_mode=False,
    )
    sync_score = sync_metric.measure(
        _tool_call_conversation(), _show_indicator=False
    )

    assert sync_score == 0.6
    assert sync_metric.score == 0.6
    assert sync_metric.reason == "Summary reason.\nSummary reason."
    assert "ArgumentCorrectnessScore" in sync_judge.calls
    assert sync_judge.calls.count("Reason") == 2

    async_judge = _StubJudgeLLM(
        tool_selection_score=1.0,
        argument_correctness_score=0.6,
    )
    async_metric = ToolUseMetric(
        available_tools=_available_tools(),
        model=async_judge,
        async_mode=True,
    )
    async_score = asyncio.run(
        async_metric.a_measure(
            _tool_call_conversation(), _show_indicator=False
        )
    )

    assert async_score == 0.6
    assert async_metric.score == 0.6
    assert async_metric.reason == "Summary reason.\nSummary reason."
    assert "ArgumentCorrectnessScore" in async_judge.calls
    assert async_judge.calls.count("Reason") == 2


def test_calculate_score_edge_cases():
    judge = _StubJudgeLLM()
    metric = ToolUseMetric(
        available_tools=_available_tools(),
        model=judge,
        threshold=0.5,
        strict_mode=False,
    )

    assert (
        metric._calculate_score(
            [ToolSelectionScore(score=0.8, reason="good")], []
        )
        == 0.8
    )
    assert metric._calculate_score([], []) == 0.0
    assert (
        metric._calculate_score(
            [
                ToolSelectionScore(score=1.0, reason="perfect"),
                ToolSelectionScore(score=0.5, reason="partial"),
            ],
            [ArgumentCorrectnessScore(score=0.6, reason="ok")],
        )
        == 0.6
    )

    strict_metric = ToolUseMetric(
        available_tools=_available_tools(),
        model=judge,
        strict_mode=True,
    )
    assert (
        strict_metric._calculate_score(
            [ToolSelectionScore(score=0.9, reason="almost")], []
        )
        == 0
    )
    assert (
        strict_metric._calculate_score(
            [ToolSelectionScore(score=1.0, reason="perfect")], []
        )
        == 1.0
    )
