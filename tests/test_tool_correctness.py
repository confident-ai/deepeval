from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from deepeval.metrics import ToolCorrectnessMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate, assert_test
import pytest

input_parameters_correctness = ToolCorrectnessMetric(
    threshold=0.5,
    tool_call_param=ToolCallParams.INPUT_PARAMETERS,
    should_exact_match=False,
)
tool_correctness_metric = ToolCorrectnessMetric(
    threshold=0.5,
    tool_call_param=ToolCallParams.TOOL,
    should_exact_match=True,
    should_consider_ordering=True,
)
output_correctness_metric = ToolCorrectnessMetric(
    tool_call_param=ToolCallParams.OUTPUT,
)
answer_relevancy_metric = AnswerRelevancyMetric()

# Test Case 1: Simple Calculation with String Output
test_case_simple_string_output = LLMTestCase(
    input="Calculate the total for 2 items at $20 each, including 5% tax.",
    actual_output="The total cost is $42.00.",
    tools_called=[
        ToolCall(
            name="Calculator Tool",
            description="Performs calculations.",
            input_parameters={"expression": "(20 * 2) * 1.05"},
            output="42.00",
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Calculator Tool",
            description="Performs calculations.",
            input_parameters={"expression": "(20 * 2) * 1.05"},
            output="42.00",
        ),
    ],
)

# Test Case 2: Nested Dictionary Input and Boolean Output
test_case_nested_boolean_output = LLMTestCase(
    input="Check if the total cost exceeds $50 for 3 items at $17 each, including 10% tax.",
    actual_output="Yes, it exceeds $50.",
    tools_called=[
        ToolCall(
            name="Calculator Tool",
            description="Performs calculations.",
            input_parameters={
                "calculation": {
                    "unit_price": 17,
                    "quantity": 3,
                    "tax_rate": 0.10,
                },
                "check": {"threshold": 50},
            },
            output=True,
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Calculator Tool",
            description="Performs calculations.",
            input_parameters={
                "calculation": {
                    "unit_price": 17,
                    "quantity": 3,
                    "tax_rate": 0.10,
                },
                "check": {"threshold": 50},
            },
            output=True,
        ),
    ],
)

# Test Case 3: Array Input and Numeric Output
test_case_array_input = LLMTestCase(
    input="Find the sum of the list [5, 10, 15].",
    actual_output="The sum is 30.",
    tools_called=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": [5, 10, 15]},
            output=30,
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": [5, 10, 15]},
            output=30,
        ),
    ],
)

# Test Case 4: Nested and Mixed Types
test_case_nested_mixed = LLMTestCase(
    input="Calculate the average of two groups: [5, 10] and [15, 20].",
    actual_output="The average is 12.5.",
    tools_called=[
        ToolCall(
            name="Statistics Tool",
            description="Performs statistical calculations.",
            input_parameters={
                "group1": [5, 10],
                "group2": [15, 20],
                "operation": "average",
            },
            output={"average": 12.5},
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Statistics Tool",
            description="Performs statistical calculations.",
            input_parameters={
                "group1": [5, 10],
                "group2": [15, 20],
                "operation": "average",
            },
            output={"average": 12.5},
        ),
    ],
)

# Test Case 5: String Input and Dictionary Output
test_case_string_to_dict = LLMTestCase(
    input="Break down the phrase 'Hello, World!' into character counts.",
    actual_output="{'H': 1, 'e': 1, 'l': 3, 'o': 2, ',': 1, ' ': 1, 'W': 1, 'r': 1, 'd': 1, '!': 1}",
    tools_called=[
        ToolCall(
            name="Character Count Tool",
            description="Counts occurrences of characters in a string.",
            input_parameters={"text": "Hello, World!"},
            output={
                "H": 1,
                "e": 1,
                "l": 3,
                "o": 2,
                ",": 1,
                " ": 1,
                "W": 1,
                "r": 1,
                "d": 1,
                "!": 1,
            },
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Character Count Tool",
            description="Counts occurrences of characters in a string.",
            input_parameters={"text": "Hello, World!"},
            output={
                "H": 1,
                "e": 1,
                "l": 3,
                "o": 2,
                ",": 1,
                " ": 1,
                "W": 1,
                "r": 1,
                "d": 1,
                "!": 1,
            },
        ),
    ],
)

# Test Case 6: Edge Case - Empty Input
test_case_empty_input = LLMTestCase(
    input="Sum an empty list.",
    actual_output="The sum is 0.",
    tools_called=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": []},
            output=0,
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": []},
            output=0,
        ),
    ],
)

# Test Case 7: Edge Case - Large Input
test_case_large_input = LLMTestCase(
    input="Find the sum of numbers from 1 to 1,000,000.",
    actual_output="The sum is 500000500000.",
    tools_called=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": list(range(1, 1000001))},
            output=500000500000,
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": list(range(1, 1000001))},
            output=500000500000,
        ),
    ],
)

# Test Case 8: Non-Numeric Nested Input
test_case_non_numeric_nested = LLMTestCase(
    input="Extract the domain from the emails in ['user@example.com', 'admin@domain.com'].",
    actual_output="{'example.com': 1, 'domain.com': 1}",
    tools_called=[
        ToolCall(
            name="Domain Extractor",
            description="Extracts and counts domains from email addresses.",
            input_parameters={
                "emails": ["user@example.com", "admin@domain.com"]
            },
            output={"example.com": 1, "domain.com": 1},
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Domain Extractor",
            description="Extracts and counts domains from email addresses.",
            input_parameters={
                "emails": ["user@example.com", "admin@domain.com"]
            },
            output={"example.com": 1, "domain.com": 1},
        ),
    ],
)

# Test Case 9: Mixed Output Types
test_case_mixed_output = LLMTestCase(
    input="Perform operations on 10: square it, divide by 2, and return the results.",
    actual_output="{'square': 100, 'half': 5.0}",
    tools_called=[
        ToolCall(
            name="Math Tool",
            description="Performs various mathematical operations.",
            input_parameters={
                "number": 10,
                "operations": ["square", "divide_by_2"],
            },
            output={"square": 100, "half": 5.0},
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Math Tool",
            description="Performs various mathematical operations.",
            input_parameters={
                "number": 10,
                "operations": ["rectangle", "divide_by_2"],
            },
            output={"square": 100, "half": 5.0},
        ),
    ],
)

# Test Case 10: Invalid Input Handling
test_case_invalid_input = LLMTestCase(
    input="Sum the numbers in ['a', 2, 3].",
    actual_output="Invalid input detected.",
    tools_called=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": ["a", 2, 3]},
            output="Error: Non-numeric input detected.",
        ),
    ],
    expected_tools=[
        ToolCall(
            name="Summation Tool",
            description="Calculates the sum of a list of numbers.",
            input_parameters={"numbers": ["a", 2, 3]},
            output="Error: Non-numeric input detected.",
        ),
    ],
)

dataset = EvaluationDataset(
    test_cases=[
        test_case_simple_string_output,
        test_case_nested_boolean_output,
        test_case_array_input,
        test_case_nested_mixed,
        test_case_string_to_dict,
        test_case_empty_input,
        # test_case_large_input,
        test_case_non_numeric_nested,
        test_case_mixed_output,
        test_case_invalid_input,
    ]
)


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    assert_test(
        test_case,
        [
            # answer_relevancy_metric,
            tool_correctness_metric,
            input_parameters_correctness,
            output_correctness_metric,
        ],
    )
