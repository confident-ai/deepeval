from deepeval.test_case import ToolCall, LLMTestCase

################################################
# Initialization ###############################
################################################

search_tool = ToolCall(
    name="search tool",
    description="tool that searches the web for the latest information",
    reasoning="User asked for information outside my knowledge...searching web for information",
    output={"num_results": 5, "top_results": "Today is Jan 19, 2025"},
    input_parameters={"user_string": "what is today's date?"},
)

calculator_tool = ToolCall(
    name="calculator tool",
    description="tool that calculates anything",
    reasoning="User asked for a solution to math equation... using calculator tool",
    output=5,
    input_parameters={"user_string": "what is 2+3?"},
)

tool_test_case = LLMTestCase(
    input="What is today's date and what is 2+3?",
    actual_output="Today's date is Jan 19, 2025 and 2+3 = 5",
    tools_called=[search_tool, calculator_tool],
    expected_tools=[search_tool, calculator_tool],
)

################################################
# Metrics ######################################
################################################

from deepeval.metrics import ToolCorrectnessMetric, GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate

tool_correctness_metric = ToolCorrectnessMetric(verbose_mode=True)
# tool_correctness_geval_metric = GEval(
#     name="Tool Correctness",
#     criteria="Is the expected tools same as tools called",
#     evaluation_params=[
#         LLMTestCaseParams.TOOLS_CALLED,
#         LLMTestCaseParams.EXPECTED_TOOLS,
#     ],
#     verbose_mode=True,
# )
# tool_correctness_geval_metric.measure(tool_test_case)
tool_correctness_metric.measure(tool_test_case)

################################################
# Dataset ######################################
################################################

from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(test_cases=[tool_test_case])
# dataset.push("test", auto_convert_test_cases_to_goldens=True)
# dataset.pull('test')

# print(dataset)

################################################
# Evalutae #####################################
################################################

# evaluate([tool_test_case], metrics=[tool_correctness_metric, tool_correctness_geval_metric])
