from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ToolCall


class _StubModel(DeepEvalBaseLLM):
    """Offline stub so the metric can be built without a provider/API key."""

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs):
        return "", 0.0

    async def a_generate(self, *args, **kwargs):
        return "", 0.0

    def get_model_name(self, *args, **kwargs):
        return "stub"


def _ordering_reason(expected, called):
    metric = ToolCorrectnessMetric(
        model=_StubModel(),
        should_consider_ordering=True,
    )
    metric.expected_tools = [ToolCall(name=name) for name in expected]
    metric.tools_called = [ToolCall(name=name) for name in called]
    return metric._generate_reason()


def test_reordered_repeated_tool_reason_is_not_empty():
    # Regression for #3014: a repeated tool called out of order lowers the
    # weighted-LCS score below 1, but the set-based missing / out-of-order
    # checks see every name on both sides, so `issues` used to stay empty and
    # the reason rendered as a dangling "Incorrect tool usage: ;".
    reason = _ordering_reason(
        ["WebSearch", "ToolQuery", "WebSearch"],
        ["WebSearch", "WebSearch", "ToolQuery"],
    )
    assert "Incorrect tool usage: ;" not in reason
    assert "different order" in reason


def test_out_of_order_tool_reason_still_reported():
    reason = _ordering_reason(["A", "B"], ["B", "A"])
    assert "out-of-order tools" in reason


def test_correct_ordering_reason():
    reason = _ordering_reason(["A", "B", "A"], ["A", "B", "A"])
    assert "Correct ordering" in reason
