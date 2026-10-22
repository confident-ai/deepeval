"""GoalAccuracyMetric must not duplicate earlier turns inside later steps.

`_goal_and_steps_taken` accumulated one running transcript string and
appended it as a "step" on every assistant turn, so steps_taken was
["A", "A B", "A B C", ...]. The judge prompt joins the steps, repeating
message A n times, B n-1 times ... — O(n^2) prompt growth on long
conversations and duplicated evidence shown to the goal/plan judges.

Deterministic helper-level tests: no LLM, no API keys.
"""

from deepeval.metrics.goal_accuracy.goal_accuracy import GoalAccuracyMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn


class _NeverCalledModel(DeepEvalBaseLLM):
    """Constructor-friendly stand-in; the helper under test never calls it."""

    def load_model(self):
        return None

    def generate(self, prompt: str, schema=None, **kwargs):  # pragma: no cover
        raise AssertionError("the helper must not call the model")

    async def a_generate(
        self, prompt: str, schema=None, **kwargs
    ):  # pragma: no cover
        raise AssertionError("the helper must not call the model")

    def get_model_name(self):
        return "stub-goal-accuracy"


def _metric() -> GoalAccuracyMetric:
    # Only the pure helper is exercised; the model is never called.
    return GoalAccuracyMetric(model=_NeverCalledModel())


def _interaction():
    return [
        Turn(role="user", content="MSG-U"),
        Turn(role="assistant", content="MSG-A"),
        Turn(role="assistant", content="MSG-B"),
        Turn(role="assistant", content="MSG-C"),
    ]


def test_each_step_contains_exactly_one_turn():
    metric = _metric()

    result = metric._goal_and_steps_taken([_interaction()])

    steps = result[0].steps_taken
    assert len(steps) == 3
    assert steps[0] == "Assistant messages: \nMSG-A \n"
    assert steps[1] == "Assistant messages: \nMSG-B \n"
    assert steps[2] == "Assistant messages: \nMSG-C \n"


def test_joined_steps_mention_each_message_once():
    metric = _metric()

    result = metric._goal_and_steps_taken([_interaction()])
    joined = "\n".join(result[0].steps_taken)

    for message in ("MSG-A", "MSG-B", "MSG-C"):
        assert joined.count(message) == 1, (
            f"{message} must appear exactly once in the joined judge prompt, "
            f"found {joined.count(message)}"
        )


def test_tools_called_stay_attached_to_their_own_step():
    metric = _metric()
    interaction = [
        Turn(role="user", content="MSG-U"),
        Turn(
            role="assistant",
            content="MSG-A",
            tools_called=[ToolCall(name="search", input_parameters={"q": "x"})],
        ),
        Turn(role="assistant", content="MSG-B"),
    ]

    result = metric._goal_and_steps_taken([interaction])

    steps = result[0].steps_taken
    assert "search" in steps[0]
    assert (
        "search" not in steps[1]
    ), "a later step must not repeat an earlier turn's tool calls"


def test_user_goal_collects_leading_user_turns():
    metric = _metric()

    result = metric._goal_and_steps_taken([_interaction()])

    assert result[0].user_goal == "User messages: \nMSG-U\n"
