from deepeval.metrics.dag import (
    VerdictNode,
    TaskNode,
    NonBinaryJudgementNode,
    BinaryJudgementNode,
)
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.metrics import DAGMetric


non_binary = NonBinaryJudgementNode(
    criteria="What is the output language?",
    children=[
        VerdictNode(verdict="english", score=10),
        VerdictNode(verdict="French", score=0),
    ],
)

verdict_node_yes = VerdictNode(verdict=True, child=non_binary)
verdict_node_no = VerdictNode(verdict=False, score=0)


binary = BinaryJudgementNode(
    criteria="does the list of extracted words contain the same number of words in the `actual_output`, ignore formatting?",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[verdict_node_yes, verdict_node_no],
)


task_node = TaskNode(
    instructions="Extract all words from the `actual output`",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="list of extracted words",
    children=[binary, non_binary],
)


dag_metric = DAGMetric(name="dag", root_node=[task_node], async_mode=False)

# dag_metric.measure(
#     test_case=LLMTestCase(input="..", actual_output="Les miserable")
# )
# print(dag_metric.score)
# print(dag_metric.reason)


async def main():
    # Perform the measure asynchronously
    await dag_metric.a_measure(
        test_case=LLMTestCase(input="..", actual_output="Les miserable")
    )

    # Print results after the measure is complete
    print(dag_metric.score)
    print(dag_metric.reason)


import asyncio

asyncio.run(main())


from deepeval.metrics.dag import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)

correct_order_node = NonBinaryJudgementNode(
    criteria="Are the headings in the correct order: 'intro' => 'body' => 'conclusion'?",
    children=[
        VerdictNode(verdict="Yes", score=10),
        VerdictNode(verdict="Two are out of order", score=4),
        VerdictNode(verdict="All out of order", score=2),
    ],
)

correct_headings_node = BinaryJudgementNode(
    criteria="Does the heading contain all three: 'intro', 'body', and 'conclusion'?",
    children=[VerdictNode(verdict=False, score=0), correct_order_node],
)

extract_headings_node = TaskNode(
    instructions="Extract all headings in `actual_output`",
    LLMTestCaseParams=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Summary headings",
    children=[correct_headings_node, correct_order_node],
)
