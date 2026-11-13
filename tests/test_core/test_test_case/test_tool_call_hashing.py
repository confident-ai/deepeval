import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams


class NeverGenerate(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def get_model_name(self):
        return "offline-test-model"

    def generate(self, *args, **kwargs):
        raise AssertionError("Tool correctness must not generate model output")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("Tool correctness must not generate model output")

    batch_generate = generate
    generate_samples = generate
    generate_raw_response = generate
    generate_with_schema = generate
    a_generate_raw_response = a_generate
    a_generate_with_schema = a_generate


def _tool_call(field, mapping):
    # Top-level parameter names are strings; their values may be arbitrary dicts.
    value = {"lookup": mapping} if field == "input_parameters" else mapping
    return ToolCall(name="lookup", **{field: value})


@pytest.mark.parametrize("field", ["input_parameters", "output"])
def test_mixed_mapping_keys_are_hashable_regardless_of_order(field):
    left = _tool_call(field, {1: ["one"], "mode": {"fast": True}})
    right = _tool_call(field, {"mode": {"fast": True}, 1: ["one"]})

    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


@pytest.mark.parametrize("field", ["input_parameters", "output"])
@pytest.mark.parametrize("number", [True, 1, 1.0], ids=["bool", "int", "float"])
def test_equal_numeric_keys_and_values_have_equal_hashes(field, number):
    left = _tool_call(field, {True: [True], "values": {True, "ok"}})
    right = _tool_call(
        field, {"values": frozenset({number, "ok"}), number: [number]}
    )

    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


@pytest.mark.parametrize("field", ["input_parameters", "output"])
def test_integer_and_string_keys_keep_distinct_values(field):
    left = _tool_call(field, {1: "integer", "1": "string"})
    right = _tool_call(field, {"1": "integer", 1: "string"})

    assert left != right
    assert len({left, right}) == 2


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("field", ["input_parameters", "output"])
@pytest.mark.parametrize(
    "matches", [True, False], ids=["matching", "mismatched"]
)
def test_tool_correctness_measures_mixed_mapping_keys(
    async_mode, field, matches
):
    expected = _tool_call(field, {1: "one", "mode": "fast"})
    called = _tool_call(
        field,
        {"mode": "fast", 1: "one"} if matches else {"mode": "slow", 1: "two"},
    )
    test_case = LLMTestCase(
        input="Look up the value",
        actual_output="done",
        tools_called=[called],
        expected_tools=[expected],
    )
    metric = ToolCorrectnessMetric(
        model=NeverGenerate(),
        async_mode=async_mode,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
            ToolCallParams.OUTPUT,
        ],
    )

    metric.measure(test_case, _show_indicator=False)

    assert metric.score == (1.0 if matches else 0.0)
    assert metric.success is matches
    assert metric.reason
