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
            role="user", content="Hi, what's the weather like in Paris today?"
        ),
        Turn(
            role="assistant",
            content="The weather in Paris today is sunny and 24°C.",
        ),
    ],
    scenario="Ask about weather",
    expected_outcome="Assistant provides weather info",
)

task_node = ConversationalTaskNode(
    instructions="Summarize the assistant's reply in one sentence.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    children=[],
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
task_node.children.append(binary_node)

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
