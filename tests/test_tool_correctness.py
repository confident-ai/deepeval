from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams, MLLMTestCase
from deepeval.metrics import ToolCorrectnessMetric
from deepeval import evaluate

tool_correctness_metric = ToolCorrectnessMetric(
    threshold=0.5,
    evaluation_params=[
        ToolCallParams.TOOL,
        ToolCallParams.INPUT_PARAMETERS,
        ToolCallParams.OUTPUT,
    ],
)

exact_match = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(
            name="tool1", input_parameters={"a": 1, "b": 2, "c": 2}, output="x"
        )
    ],
    expected_tools=[
        ToolCall(
            name="tool1", input_parameters={"a": 1, "b": 4, "c": 2}, output="x"
        )
    ],
)

tool_name_mismatch = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool2", input_parameters={"a": 1, "b": 2}, output="x")
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x")
    ],
)

input_parameter_mismatch = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool1", input_parameters={"a": 2, "b": 2}, output="x")
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x")
    ],
)

output_mismatch = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="y")
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x")
    ],
)

extra_tools_called = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x"),
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y"),
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x")
    ],
)

missing_tools = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x")
    ],
)

correct_ordering = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x"),
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y"),
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x"),
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y"),
    ],
)

out_of_order_tools = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y"),
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x"),
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x"),
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y"),
    ],
)

missing_and_out_of_order = LLMTestCase or MLLMTestCase(
    input="",
    actual_output="",
    tools_called=[
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y")
    ],
    expected_tools=[
        ToolCall(name="tool1", input_parameters={"a": 1, "b": 2}, output="x"),
        ToolCall(name="tool2", input_parameters={"c": 3}, output="y"),
    ],
)


# List of test case names
test_cases = [
    exact_match,
    tool_name_mismatch,
    input_parameter_mismatch,
    output_mismatch,
    extra_tools_called,
    missing_tools,
    correct_ordering,
    out_of_order_tools,
    missing_and_out_of_order,
]

evaluate(
    test_cases=test_cases, metrics=[tool_correctness_metric], verbose_mode=True
)
