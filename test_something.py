from dotenv import load_dotenv
from deepeval.test_case import ConversationalTestCase, TurnParams, Turn
from deepeval.metrics.conversational_dag import (
    DeepAcyclicGraph,
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalVerdictNode,
)
from deepeval.metrics import ConversationalDAGMetric

# Load environment variables
load_dotenv()

test_case = ConversationalTestCase(
    turns=[
        Turn(
            role="user", content="what's the weather like"
        ),
        Turn(
            role="assistant",
            content="Screw weather",
        ),
        Turn(
            role="user", content="Hi, what's the weather like in Paris today?"
        ),
        Turn(
            role="assistant",
            content="The weather in Paris today is sunny and 24°C.",
        ),
        Turn(
            role="user", content="what's the weather like?"
        ),
        Turn(
            role="assistant",
            content="Screw weather",
        ),
    ],
    scenario="Ask about weather",
    expected_outcome="Assistant provides weather info",
)

# 2. Binary Judgement: Does mention 'sunny'?
binary_node = ConversationalBinaryJudgementNode(
    criteria="Does the assistant's reply mention that it is sunny?",
    children=[
        ConversationalVerdictNode(verdict=False, score=0),
        ConversationalVerdictNode(verdict=True, score=10),
    ],
)

# Connect nodes
task_node = ConversationalTaskNode(
    instructions="Summarize the assistant's reply in one sentence.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    turn_window=(0,1),
    children=[binary_node],
)

# 3. Build DAG
dag = DeepAcyclicGraph(root_nodes=[task_node])

if __name__ == "__main__":
    # Create DAG metric
    format_correctness = ConversationalDAGMetric(
        name="Weather Mention Check",
        dag=dag,
        threshold=0.5,
        include_reason=True,
        async_mode=False,
        verbose_mode=True,
    )

    print("Testing DAG...")

    format_correctness.measure(test_case)
    print(f"Score: {format_correctness.score}")
    print(f"Success: {format_correctness.success}")
    print(f"Reason: {format_correctness.reason}")


# from dotenv import load_dotenv
# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# from deepeval.metrics.dag import (
#     DeepAcyclicGraph,
#     TaskNode,
#     BinaryJudgementNode,
#     VerdictNode,
# )
# from deepeval.metrics import DAGMetric

# # Load environment variables
# load_dotenv()

# # Create the DAG structure
# correct_letters_node = BinaryJudgementNode(
#     criteria="Does first heading has i as first letter?",
#     children=[
#         VerdictNode(verdict=False, score=0),
#         VerdictNode(verdict=True,score=1),
#     ],
# )
# extract_letter_node= TaskNode(
#     instructions="Extract first heading",
#     output_label="first heading",
#     children=[correct_letters_node],
#     evaluation_params=[]
# )

# correct_order_node = BinaryJudgementNode(
#     criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
#     children=[
#         VerdictNode(verdict=False, score=0),
#         VerdictNode(verdict=True, child=correct_letters_node),
#     ],
# )

# correct_headings_node = BinaryJudgementNode(
#     criteria="Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
#     children=[
#         VerdictNode(verdict=False, score=0),
#         VerdictNode(verdict=True, child=correct_order_node),
#     ],
# )

# extract_headings_node = TaskNode(
#     instructions="Extract all headings in `actual_output`",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
#     output_label="Summary headings",
#     children=[correct_headings_node, correct_order_node, extract_letter_node],
# )

# # Create the DAG
# dag = DeepAcyclicGraph(root_nodes=[extract_headings_node])

# # Test case
# test_case = LLMTestCase(
#     input="Summarize this team meeting with proper structure using intro, body, and conclusion headings:",
#     actual_output="""
# Intro:
# Alice outlined the agenda: product updates, blockers, and marketing alignment.

# Body:
# Bob reported performance issues being optimized, with fixes expected by Friday. Charlie requested finalized messaging by Monday for marketing preparation. Bob confirmed an early stable build would be ready.

# Conclusion:
# The team aligned on next steps: engineering finalizing fixes, marketing preparing content, and a follow-up sync scheduled for Wednesday.
# """
# )

# if __name__ == "__main__":
#     # Create DAG metric
#     format_correctness = DAGMetric( name="Format Correctness", dag=dag)

#     print("Testing DAG...")

#     try:
#         format_correctness.measure(test_case)
#         print(f"Score: {format_correctness.score}")
#         print(f"Success: {format_correctness.success}")
#         print(f"Reason: {format_correctness.reason}")
#     except Exception as e:
#         print(f"Error: {e}")