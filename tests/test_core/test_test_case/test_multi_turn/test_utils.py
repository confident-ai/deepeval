from deepeval.metrics import (
    GoalAccuracyMetric,
    MCPTaskCompletionMetric,
    MultiTurnMCPUseMetric,
    ToolUseMetric,
)
from deepeval.metrics.utils import get_unit_interactions, split_unit_interaction
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn


def make_turns(seq):
    """Helper to create Turn objects from list of (role, content)."""
    return [Turn(role=role, content=content) for role, content in seq]


class TestGetUnitInteractions:
    def test_consecutive_users(self):
        seq = [("user", "u1"), ("user", "u2"), ("assistant", "a1")]
        expected = [[("user", "u1"), ("user", "u2"), ("assistant", "a1")]]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_consecutive_assistants(self):
        seq = [("user", "u1"), ("assistant", "a1"), ("assistant", "a2")]
        expected = [[("user", "u1"), ("assistant", "a1"), ("assistant", "a2")]]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_user_assistant_user(self):
        seq = [("user", "u1"), ("assistant", "a1"), ("user", "u2")]
        expected = [[("user", "u1"), ("assistant", "a1")]]  # last user ignored
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_starts_with_assistant(self):
        seq = [("assistant", "a1"), ("user", "u1"), ("assistant", "a2")]
        expected = [[("assistant", "a1"), ("user", "u1"), ("assistant", "a2")]]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_assistant_only_start_then_end_with_user(self):
        seq = [("assistant", "a1"), ("assistant", "a2"), ("user", "u1")]
        expected = []  # ends with user -> ignored
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_multiple_units(self):
        seq = [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ]
        expected = [
            [("user", "u1"), ("assistant", "a1")],
            [("user", "u2"), ("assistant", "a2")],
        ]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_empty_input(self):
        seq = []
        expected = []
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected


class _RecordingJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None, **kwargs):
        self.prompts.append(prompt)
        if schema is None:
            return "ok"
        fields = getattr(schema, "model_fields", {})
        values = {}
        for name in fields:
            if name == "score":
                values[name] = 1.0
            elif name == "reason":
                values[name] = "ok"
            elif name == "verdict":
                values[name] = "yes"
        return schema(**values)

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema, **kwargs)

    def get_model_name(self):
        return "recording-judge"


class TestAssistantFirstUserTaskExtraction:
    def test_split_unit_interaction_leading_assistant(self):
        unit = make_turns(
            [
                ("assistant", "Hi!"),
                ("user", "First request."),
                ("user", "More detail."),
                ("assistant", "Done."),
            ]
        )
        user_turns, response_turns = split_unit_interaction(unit)
        assert [(t.role, t.content) for t in user_turns] == [
            ("user", "First request."),
            ("user", "More detail."),
        ]
        assert [(t.role, t.content) for t in response_turns] == [
            ("assistant", "Done."),
        ]

    def test_split_unit_interaction_without_user_turns(self):
        unit = make_turns([("assistant", "Hi!")])
        user_turns, response_turns = split_unit_interaction(unit)
        assert user_turns == []
        assert response_turns == []

    def test_goal_accuracy_includes_user_task_after_greeting(self):
        judge = _RecordingJudge()
        metric = GoalAccuracyMetric(
            model=judge, eval_mode="llm", async_mode=False
        )
        turns = [
            Turn(role="assistant", content="Hi! How can I help?"),
            Turn(role="user", content="Book me a flight to Delhi on Friday."),
            Turn(
                role="assistant",
                content="Done - booked BA 143 to Delhi, Friday 9:40.",
            ),
        ]
        metric.measure(
            ConversationalTestCase(turns=turns), _show_indicator=False
        )
        assert any(
            "Book me a flight to Delhi on Friday." in prompt
            for prompt in judge.prompts
        )

    def test_tool_use_includes_user_task_after_greeting(self):
        judge = _RecordingJudge()
        metric = ToolUseMetric(
            available_tools=[ToolCall(name="book_flight")],
            model=judge,
            eval_mode="llm",
            async_mode=False,
        )
        turns = [
            Turn(role="assistant", content="Hi! How can I help?"),
            Turn(role="user", content="Book me a flight to Delhi on Friday."),
            Turn(
                role="assistant",
                content="Booked.",
                tools_called=[ToolCall(name="book_flight")],
            ),
        ]
        units = get_unit_interactions(turns)
        extracted = metric._get_user_input_and_turns(units)
        assert len(extracted) == 1
        assert (
            "Book me a flight to Delhi on Friday." in extracted[0].user_messages
        )
        assert "Hi! How can I help?" not in extracted[0].assistant_messages
        assert extracted[0].tools_used is True

        metric.measure(
            ConversationalTestCase(turns=turns), _show_indicator=False
        )
        assert any(
            "Book me a flight to Delhi on Friday." in prompt
            for prompt in judge.prompts
        )

    def test_mcp_metrics_extract_user_task_without_mislabeling_steps(self):
        judge = _RecordingJudge()
        turns = [
            Turn(role="assistant", content="Welcome!"),
            Turn(role="user", content="Check status."),
            Turn(role="user", content="Include details."),
            Turn(role="assistant", content="All systems green."),
        ]
        units = get_unit_interactions(turns)

        for metric_cls in (MCPTaskCompletionMetric, MultiTurnMCPUseMetric):
            metric = metric_cls(model=judge, eval_mode="llm", async_mode=False)
            tasks = metric._get_tasks(units)
            assert len(tasks) == 1
            assert tasks[0].task == "Check status.\nInclude details.\n"
            assert tasks[0].steps_taken == [
                "Agent's response to user: \nAll systems green."
            ]
