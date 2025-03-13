from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="""
Alice: "Today's agenda: product update, blockers, and marketing timeline. Bob, updates?"
Bob: "Core features are done, but we're optimizing performance for large datasets. Fixes by Friday, testing next week."
Alice: "Charlie, does this timeline work for marketing?"
Charlie: "We need finalized messaging by Monday."
Alice: "Bob, can we provide a stable version by then?"
Bob: "Yes, we’ll share an early build."
Charlie: "Great, we'll start preparing assets."
Alice: "Plan: fixes by Friday, marketing prep Monday, sync next Wednesday. Thanks, everyone!"
""",
    actual_output="""
Intro:
Alice outlined the agenda: product updates, blockers, and marketing alignment.

Body:
Bob reported performance issues being optimized, with fixes expected by Friday. Charlie requested finalized messaging by Monday for marketing preparation. Bob confirmed an early stable build would be ready.

Conclusion:
The team aligned on next steps: engineering finalizing fixes, marketing preparing content, and a follow-up sync scheduled for Wednesday.
""",
)

from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)
from deepeval.test_case import LLMTestCaseParams

correct_order_node = NonBinaryJudgementNode(
    criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
    children=[
        VerdictNode(verdict="Yes", score=10),
        VerdictNode(verdict="Two are out of order", score=4),
        VerdictNode(verdict="All out of order", score=2),
    ],
)

correct_headings_node = BinaryJudgementNode(
    label="Correct Heading Node",
    criteria="Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=correct_order_node),
    ],
)

extract_headings_node = TaskNode(
    instructions="Extract all headings in `actual_output`",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Summary headings",
    children=[correct_headings_node, correct_order_node],
)

from deepeval.metrics.dag.utils import copy_graph

# create the DAG
dag = DeepAcyclicGraph(root_nodes=[correct_headings_node])


# copy_graph(dag)

from deepeval.metrics import DAGMetric
from deepeval import evaluate


format_correctness = DAGMetric(
    name="Format Correctness", dag=dag, verbose_mode=True, async_mode=False
)
# format_correctness.measure(test_case)
# print(format_correctness.score, format_correctness.reason)

evaluate([test_case, test_case], [format_correctness])
