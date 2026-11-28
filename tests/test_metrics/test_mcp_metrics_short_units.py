"""Regression tests for #3647: MCP multi-turn metrics and short units.

`MCPTaskCompletionMetric` and `MultiTurnMCPUseMetric` built their task lists
with an `if len(unit_interaction) <= 2: continue` filter, so a task completed
in a single `[user, assistant]` exchange was never scored. When no task was
built, an `else 1` divisor turned the empty score list into a silent 0.0 and
the reason was an LLM summary of an empty list.

These tests run with a stub model, so they need no LLM provider API key.
"""

import pytest

from deepeval.metrics import MCPTaskCompletionMetric, MultiTurnMCPUseMetric
from deepeval.metrics.mcp.schema import ArgsScore, Reason, TaskScore, ToolScore
from deepeval.metrics.mcp.utils import NO_TASKS_REASON
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, MCPServer, Turn


class _StubLLM(DeepEvalBaseLLM):
    """Scores every task 1.0 and records which schemas were requested."""

    def __init__(self):
        self.requested = []
        super().__init__(model="stub-llm")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, *args, **kwargs):
        if schema is None:
            raise AssertionError("schema-less generate call not expected")
        self.requested.append(schema.__name__)
        if schema in (TaskScore, ToolScore, ArgsScore):
            return schema(score=1.0, reason="Done.")
        if schema is Reason:
            return Reason(reason="LLM reason.")
        raise AssertionError(f"unexpected schema: {schema.__name__}")

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema=schema, *args, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


def _test_case(turns):
    return ConversationalTestCase(
        turns=turns,
        mcp_servers=[MCPServer(server_name="Test Server")],
    )


# Two tasks, each answered directly in one [user, assistant] exchange.
SINGLE_EXCHANGE_TURNS = [
    Turn(role="user", content="What is your refund window?"),
    Turn(role="assistant", content="You can get a full refund for 30 days."),
    Turn(role="user", content="Do you ship to Canada?"),
    Turn(role="assistant", content="Yes, we ship to Canada."),
]

# No complete user -> assistant interaction, so no unit and no task.
NO_UNIT_TURNS = [
    Turn(role="user", content="Hello?"),
]

METRIC_CLASSES = [MCPTaskCompletionMetric, MultiTurnMCPUseMetric]


@pytest.mark.parametrize("metric_cls", METRIC_CLASSES)
def test_single_exchange_units_become_tasks(metric_cls):
    metric = metric_cls(model=_StubLLM())
    units = [
        SINGLE_EXCHANGE_TURNS[0:2],
        SINGLE_EXCHANGE_TURNS[2:4],
    ]

    tasks = metric._get_tasks(units)

    assert [task.task for task in tasks] == [
        "What is your refund window?\n",
        "Do you ship to Canada?\n",
    ]
    assert tasks[0].steps_taken == [
        "Agent's response to user: \nYou can get a full refund for 30 days."
    ]


@pytest.mark.parametrize("metric_cls", METRIC_CLASSES)
@pytest.mark.parametrize("async_mode", [True, False])
def test_single_exchange_conversation_is_scored(metric_cls, async_mode):
    model = _StubLLM()
    metric = metric_cls(model=model, async_mode=async_mode)

    metric.measure(_test_case(SINGLE_EXCHANGE_TURNS), _show_indicator=False)

    assert len(metric.tasks) == 2
    assert metric.score == 1.0
    assert metric.reason == "LLM reason."


@pytest.mark.parametrize("metric_cls", METRIC_CLASSES)
@pytest.mark.parametrize("async_mode", [True, False])
def test_no_tasks_scores_one_with_explicit_reason(metric_cls, async_mode):
    model = _StubLLM()
    metric = metric_cls(model=model, async_mode=async_mode)

    metric.measure(_test_case(NO_UNIT_TURNS), _show_indicator=False)

    assert metric.tasks == []
    assert metric.score == 1
    assert metric.reason == NO_TASKS_REASON
    assert metric.success is True
    # Nothing to judge, so the model is never called (not even for a reason).
    assert model.requested == []


@pytest.mark.parametrize("metric_cls", METRIC_CLASSES)
def test_no_tasks_respects_include_reason_false(metric_cls):
    metric = metric_cls(model=_StubLLM(), include_reason=False)

    metric.measure(_test_case(NO_UNIT_TURNS), _show_indicator=False)

    assert metric.score == 1
    assert metric.reason is None


def test_task_completion_calculate_score_empty_is_one():
    metric = MCPTaskCompletionMetric(model=_StubLLM())
    assert metric._calculate_score([]) == 1


def test_mcp_use_calculate_score_empty_is_one():
    metric = MultiTurnMCPUseMetric(model=_StubLLM())
    assert metric._calculate_score([], []) == 1
