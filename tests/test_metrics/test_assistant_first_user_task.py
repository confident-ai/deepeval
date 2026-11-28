"""Regression tests for conversations where the assistant speaks first.

`get_unit_interactions` keeps a leading assistant turn (e.g. a greeting) in the
same unit as the user request that follows it. GoalAccuracyMetric,
ToolUseMetric, MCPTaskCompletionMetric and MultiTurnMCPUseMetric collected
user messages from the start of the unit and stopped at the first non-user
turn, so the greeting left the user task empty, and the MCP metrics listed the
user request as "Agent's response to user".

These tests run with a stub model, so they need no LLM provider API key.
"""

import asyncio

import pytest

from deepeval.metrics import (
    GoalAccuracyMetric,
    MCPTaskCompletionMetric,
    MultiTurnMCPUseMetric,
    ToolUseMetric,
)
from deepeval.metrics.utils import get_unit_interactions, split_user_turns
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


GREETING = Turn(role="assistant", content="Hi! How can I help?")
REQUEST = Turn(role="user", content="Book me a flight to Delhi on Friday.")
REPLY = Turn(role="assistant", content="Done, booked for Friday 9:40.")


def _units(turns):
    return get_unit_interactions(turns)


def test_split_user_turns_skips_leading_assistant_turns():
    user_turns, later_turns = split_user_turns([GREETING, REQUEST, REPLY])
    assert user_turns == [REQUEST]
    assert later_turns == [REPLY]


def test_split_user_turns_keeps_user_first_units_unchanged():
    second = Turn(role="user", content="Window seat please.")
    user_turns, later_turns = split_user_turns([REQUEST, second, REPLY])
    assert user_turns == [REQUEST, second]
    assert later_turns == [REPLY]


def test_split_user_turns_without_user_turn():
    assert split_user_turns([GREETING, REPLY]) == ([], [])


def test_goal_accuracy_keeps_user_goal_after_greeting():
    metric = GoalAccuracyMetric(model=_StubLLM(), eval_mode="llm")
    for turns in ([REQUEST, REPLY], [GREETING, REQUEST, REPLY]):
        (goal,) = metric._goal_and_steps_taken(_units(turns))
        assert goal.user_goal == f"User messages: \n{REQUEST.content}\n"
        assert len(goal.steps_taken) == 1
        assert REPLY.content in goal.steps_taken[0]
        assert GREETING.content not in goal.steps_taken[0]


def test_tool_use_keeps_user_messages_after_greeting():
    metric = ToolUseMetric(
        available_tools=[ToolCall(name="book_flight")],
        model=_StubLLM(),
        eval_mode="llm",
    )
    for turns in ([REQUEST, REPLY], [GREETING, REQUEST, REPLY]):
        (task,) = metric._get_user_input_and_turns(_units(turns))
        assert task.user_messages == f"{REQUEST.content} \n"
        assert task.assistant_messages == f"{REPLY.content} \n"


@pytest.mark.parametrize(
    "metric_class", [MCPTaskCompletionMetric, MultiTurnMCPUseMetric]
)
def test_mcp_metrics_keep_task_and_do_not_label_user_as_agent(metric_class):
    follow_up = Turn(role="assistant", content="Anything else?")
    metric = metric_class(model=_StubLLM(), eval_mode="llm")
    (task,) = metric._get_tasks(_units([GREETING, REQUEST, REPLY, follow_up]))
    assert task.task == f"{REQUEST.content}\n"
    assert task.steps_taken == [
        f"Agent's response to user: \n{REPLY.content}",
        f"Agent's response to user: \n{follow_up.content}",
    ]


@pytest.mark.parametrize(
    "metric_class", [MCPTaskCompletionMetric, MultiTurnMCPUseMetric]
)
def test_mcp_metrics_do_not_list_consecutive_user_turns_as_agent(
    metric_class,
):
    second = Turn(role="user", content="Window seat please.")
    follow_up = Turn(role="assistant", content="Anything else?")
    metric = metric_class(model=_StubLLM(), eval_mode="llm")
    (task,) = metric._get_tasks(_units([REQUEST, second, REPLY, follow_up]))
    assert task.task == f"{REQUEST.content}\n{second.content}\n"
    assert task.steps_taken == [
        f"Agent's response to user: \n{REPLY.content}",
        f"Agent's response to user: \n{follow_up.content}",
    ]


def test_goal_accuracy_several_leading_assistants_and_units():
    metric = GoalAccuracyMetric(model=_StubLLM(), eval_mode="llm")
    second_request = Turn(role="user", content="And a hotel near the airport.")
    second_reply = Turn(role="assistant", content="Booked the Airport Inn.")
    turns = [
        GREETING,
        Turn(role="assistant", content="I can book flights and hotels."),
        REQUEST,
        REPLY,
        second_request,
        second_reply,
    ]
    goals = metric._goal_and_steps_taken(_units(turns))
    assert [goal.user_goal for goal in goals] == [
        f"User messages: \n{REQUEST.content}\n",
        f"User messages: \n{second_request.content}\n",
    ]


class _RecordingLLM(_StubLLM):
    """Stub judge that records prompts and returns a fixed score."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        return schema(score=1.0, reason="ok") if schema else "ok"

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
def test_goal_accuracy_measure_sends_user_request_to_judge(async_mode):
    judge = _RecordingLLM()
    metric = GoalAccuracyMetric(
        model=judge, eval_mode="llm", async_mode=async_mode
    )
    metric.measure(
        ConversationalTestCase(turns=[GREETING, REQUEST, REPLY]),
        _show_indicator=False,
    )
    assert judge.prompts
    assert any(REQUEST.content in prompt for prompt in judge.prompts)


def test_goal_accuracy_a_measure_sends_user_request_to_judge():
    judge = _RecordingLLM()
    metric = GoalAccuracyMetric(model=judge, eval_mode="llm")
    asyncio.run(
        metric.a_measure(
            ConversationalTestCase(turns=[GREETING, REQUEST, REPLY]),
            _show_indicator=False,
        )
    )
    assert any(REQUEST.content in prompt for prompt in judge.prompts)
