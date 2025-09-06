import pytest
from deepeval.metrics.dag import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    DeepAcyclicGraph,
)
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalVerdictNode,
)
from deepeval.test_case import LLMTestCaseParams, TurnParams
from deepeval.metrics.dag.utils import (
    is_valid_dag_from_roots,
    extract_required_params,
    copy_graph,
    is_valid_dag,
)


class TestDeepAcyclicGraph:
    """Tests for DAG validation, copying, and parameter extraction."""

    def test_is_valid_dag_true(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        leaf_true = VerdictNode(verdict=True, score=10)
        judgement_node = BinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )
        root = TaskNode(
            instructions="Extract",
            output_label="X",
            children=[judgement_node],
            evaluation_params=[LLMTestCaseParams.INPUT],
        )
        assert is_valid_dag_from_roots([root], multiturn=False) is True

    def test_is_acyclic_dag(self):
        node_a = TaskNode(
            "Task A", output_label="A", evaluation_params=[], children=[]
        )
        node_b = TaskNode(
            "Task B", output_label="B", evaluation_params=[], children=[node_a]
        )
        node_a.children.append(node_b)
        assert is_valid_dag_from_roots([node_a], multiturn=False) is False

    def test_is_valid_dag_deep_nested_mixed_nodes(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        leaf_true = VerdictNode(verdict=True, score=10)
        inner_judge = BinaryJudgementNode(
            criteria="Inner?", children=[leaf_false, leaf_true]
        )
        verdict_node = VerdictNode(verdict="Yes", child=inner_judge)
        outer_judge = NonBinaryJudgementNode(
            criteria="Outer?", children=[verdict_node]
        )
        task = TaskNode(
            instructions="Top Task",
            output_label="deep",
            evaluation_params=[],
            children=[outer_judge],
        )
        assert is_valid_dag(task, multiturn=False) is True

    def test_binary_judge_2_values(self):
        verdict1 = VerdictNode(verdict=True, score=10)
        verdict2 = VerdictNode(verdict=False, score=5)
        verdict3 = VerdictNode(verdict=True, score=0)
        with pytest.raises(ValueError):
            BinaryJudgementNode(
                criteria="Should have strings in verdics",
                children=[verdict1, verdict2, verdict3],
            )

    def test_valid_non_binary(self):
        verdict1 = VerdictNode(verdict="True", score=10)
        verdict2 = VerdictNode(verdict="Idk", score=5)
        verdict3 = VerdictNode(verdict="False", score=0)
        judge_node = NonBinaryJudgementNode(
            criteria="Should have strings in verdics",
            children=[verdict1, verdict2, verdict3],
        )

        assert is_valid_dag(judge_node, multiturn=False) is True

    def test_invalid_non_binary(self):
        verdict1 = VerdictNode(verdict=True, score=10)
        verdict2 = VerdictNode(verdict=False, score=0)
        with pytest.raises(ValueError):
            NonBinaryJudgementNode(
                criteria="Should have strings in verdics",
                children=[verdict1, verdict2],
            )

    def test_invalid_verdicts(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        leaf_true = VerdictNode(verdict=False, score=10)
        with pytest.raises(ValueError):
            BinaryJudgementNode(criteria="?", children=[leaf_false, leaf_true])

    def test_extract_required_params(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        leaf_true = VerdictNode(verdict=True, score=10)
        judgement_node = BinaryJudgementNode(
            criteria="?",
            children=[leaf_false, leaf_true],
            evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT],
        )
        task = TaskNode(
            instructions="Extract something",
            output_label="abc",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            children=[judgement_node],
        )
        params = extract_required_params([task], multiturn=False)
        assert LLMTestCaseParams.INPUT in params
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert len(params) == 3

    def test_invalid_child_type(self):
        invalid_child = "string_instead_of_node"  # Invalid child type
        with pytest.raises(AttributeError):
            TaskNode(
                instructions="Invalid task",
                output_label="X",
                evaluation_params=[],
                children=[invalid_child],
            )

    def test_extract_required_params_non_binary(self):
        leaf1 = VerdictNode(verdict="A", score=0.1)
        leaf2 = VerdictNode(verdict="B", score=0.2)
        non_binary = NonBinaryJudgementNode(
            criteria="Evaluate this",
            children=[leaf1, leaf2],
            evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT],
        )
        task = TaskNode(
            instructions="Analyze",
            output_label="xyz",
            evaluation_params=[LLMTestCaseParams.INPUT],
            children=[non_binary],
        )
        params = extract_required_params([task], multiturn=False)
        assert LLMTestCaseParams.INPUT in params
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert len(params) == 2

    def test_disallow_multiple_judgement_roots(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        leaf_true = VerdictNode(verdict=True, score=10)
        judgement_node1 = BinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )
        judgement_node2 = BinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )
        with pytest.raises(ValueError):
            DeepAcyclicGraph(
                root_nodes=[judgement_node1, judgement_node2],
            )

    def test_only_score_or_child(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        with pytest.raises(ValueError):
            VerdictNode(verdict=True, score=10, child=[leaf_false])

    def test_allow_multiple_tasknode_roots(self):
        node1 = TaskNode("Task 1", "Label1", [], [])
        node2 = TaskNode("Task 2", "Label2", [], [])
        dag = DeepAcyclicGraph(root_nodes=[node1, node2])
        assert is_valid_dag(dag, multiturn=False) is True

    def test_copy_graph_isolated_and_deep(self):
        INSTRUCTIONS = "Instruction 1:"
        OUTPUT_LABEL = "Output label"
        CRITERIA = "Criteria: "

        leaf_false = VerdictNode(verdict=False, score=0)
        leaf_true = VerdictNode(verdict=True, score=10)
        judgement_node = BinaryJudgementNode(
            criteria=CRITERIA, children=[leaf_false, leaf_true]
        )
        task = TaskNode(
            instructions=INSTRUCTIONS,
            output_label=OUTPUT_LABEL,
            evaluation_params=[],
            children=[judgement_node],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])

        copied = copy_graph(dag)
        copied_task = copied.root_nodes[0]
        copied_judge = copied_task.children[0]
        copied_leaf_false = copied_judge.children[0]
        copied_leaf_true = copied_judge.children[1]

        ids_set = {
            hash(dag),
            hash(leaf_false),
            hash(leaf_true),
            hash(judgement_node),
            hash(task),
            hash(copied),
            hash(copied_leaf_false),
            hash(copied_leaf_true),
            hash(copied_judge),
            hash(copied_task),
        }

        assert len(ids_set) == 10
        assert copied is not dag
        assert isinstance(copied, DeepAcyclicGraph)
        assert isinstance(copied_leaf_false, VerdictNode)
        assert isinstance(copied_leaf_true, VerdictNode)
        assert isinstance(copied_judge, BinaryJudgementNode)
        assert isinstance(copied_task, TaskNode)
        assert copied_task is not task
        assert copied_judge is not judgement_node
        assert copied_leaf_false is not leaf_false
        assert copied_leaf_true is not leaf_true
        assert copied_task.output_label == OUTPUT_LABEL
        assert copied_task.instructions == INSTRUCTIONS
        assert len(copied_task.children) == 1
        assert len(copied_judge.children) == 2
        assert copied_judge.criteria == CRITERIA
        assert copied_leaf_false.verdict is False
        assert copied_leaf_false.score == 0
        assert copied_leaf_true.verdict is True
        assert copied_leaf_true.score == 10

    def test_non_binary_node_in_dag(self):
        leaf1 = VerdictNode(verdict="One", score=0.1)
        leaf2 = VerdictNode(verdict="Two", score=0.3)
        leaf3 = VerdictNode(verdict="Three", score=0.5)
        leaf4 = VerdictNode(verdict="Four", score=0.7)
        non_binary = NonBinaryJudgementNode(
            criteria="Evaluate based on: ",
            children=[leaf1, leaf2, leaf3, leaf4],
        )
        task = TaskNode(
            instructions="Do task",
            output_label="test",
            evaluation_params=[],
            children=[non_binary],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag(dag, multiturn=False)

    def test_task_node_leaf(self):
        task = TaskNode(
            instructions="Standalone task",
            output_label="standalone",
            evaluation_params=[LLMTestCaseParams.INPUT],
            children=[],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag_from_roots(dag.root_nodes, multiturn=False)

    def test_verdict_node_with_child(self):
        leaf = VerdictNode(verdict=False, score=0.0)
        verdict = VerdictNode(verdict=True, child=leaf)
        judge = BinaryJudgementNode(
            "Pass?",
            children=[VerdictNode(verdict=False, score=0), verdict],
        )
        task = TaskNode("Check", "result", [], [judge])
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag_from_roots(dag.root_nodes, multiturn=False)


class TestConversationalDeepAcyclicGraph:
    def test_is_valid_dag_true(self):
        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        leaf_true = ConversationalVerdictNode(verdict=True, score=10)
        judgement_node = ConversationalBinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )
        root = ConversationalTaskNode(
            instructions="Extract",
            output_label="X",
            children=[judgement_node],
            evaluation_params=[TurnParams.ROLE],
        )
        assert is_valid_dag_from_roots([root], multiturn=True) is True

    def test_is_acyclic_dag(self):
        node_a = ConversationalTaskNode(
            "Task A", output_label="A", evaluation_params=[], children=[]
        )
        node_b = ConversationalTaskNode(
            "Task B", output_label="B", evaluation_params=[], children=[node_a]
        )
        node_a.children.append(node_b)
        assert is_valid_dag_from_roots([node_a], multiturn=True) is False

    def test_is_valid_dag_deep_nested_mixed_nodes(self):
        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        leaf_true = ConversationalVerdictNode(verdict=True, score=10)
        inner_judge = ConversationalBinaryJudgementNode(
            criteria="Inner?", children=[leaf_false, leaf_true]
        )
        verdict_node = ConversationalVerdictNode(
            verdict="Yes", child=inner_judge
        )
        outer_judge = ConversationalNonBinaryJudgementNode(
            criteria="Outer?", children=[verdict_node]
        )
        task = ConversationalTaskNode(
            instructions="Top Task",
            output_label="deep",
            evaluation_params=[],
            children=[outer_judge],
        )
        assert is_valid_dag(task, multiturn=True) is True

    def test_binary_judge_2_values(self):
        verdict1 = ConversationalVerdictNode(verdict=True, score=10)
        verdict2 = ConversationalVerdictNode(verdict=False, score=5)
        verdict3 = ConversationalVerdictNode(verdict=True, score=0)
        with pytest.raises(ValueError):
            ConversationalBinaryJudgementNode(
                criteria="Should have strings in verdics",
                children=[verdict1, verdict2, verdict3],
            )

    def test_valid_non_binary(self):
        verdict1 = ConversationalVerdictNode(verdict="True", score=10)
        verdict2 = ConversationalVerdictNode(verdict="Idk", score=5)
        verdict3 = ConversationalVerdictNode(verdict="False", score=0)
        judge_node = ConversationalNonBinaryJudgementNode(
            criteria="Should have strings in verdics",
            children=[verdict1, verdict2, verdict3],
        )
        assert is_valid_dag(judge_node, multiturn=True) is True

    def test_invalid_non_binary(self):
        verdict1 = ConversationalVerdictNode(verdict=True, score=10)
        verdict2 = ConversationalVerdictNode(verdict=False, score=0)
        with pytest.raises(ValueError):
            ConversationalNonBinaryJudgementNode(
                criteria="Should have strings in verdics",
                children=[verdict1, verdict2],
            )

    def test_invalid_verdicts(self):
        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        leaf_true = ConversationalVerdictNode(verdict=False, score=10)
        with pytest.raises(ValueError):
            ConversationalBinaryJudgementNode(
                criteria="?", children=[leaf_false, leaf_true]
            )

    def test_extract_required_params(self):
        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        leaf_true = ConversationalVerdictNode(verdict=True, score=10)
        judgement_node = ConversationalBinaryJudgementNode(
            criteria="?",
            children=[leaf_false, leaf_true],
            evaluation_params=[TurnParams.CONTENT],
        )
        task = ConversationalTaskNode(
            instructions="Extract something",
            output_label="abc",
            evaluation_params=[TurnParams.ROLE],
            children=[judgement_node],
        )
        params = extract_required_params([task], multiturn=True)
        assert TurnParams.ROLE in params
        assert TurnParams.CONTENT in params
        assert len(params) == 2

    def test_invalid_child_type(self):
        invalid_child = "string_instead_of_node"
        with pytest.raises(AttributeError):
            ConversationalTaskNode(
                instructions="Invalid task",
                output_label="X",
                evaluation_params=[],
                children=[invalid_child],
            )

    def test_extract_required_params_non_binary(self):
        leaf1 = ConversationalVerdictNode(verdict="A", score=0.1)
        leaf2 = ConversationalVerdictNode(verdict="B", score=0.2)
        non_binary = ConversationalNonBinaryJudgementNode(
            criteria="Evaluate this",
            children=[leaf1, leaf2],
            evaluation_params=[TurnParams.CONTENT],
        )
        task = ConversationalTaskNode(
            instructions="Analyze",
            output_label="xyz",
            evaluation_params=[TurnParams.ROLE],
            children=[non_binary],
        )
        params = extract_required_params([task], multiturn=True)
        assert TurnParams.ROLE in params
        assert TurnParams.CONTENT in params
        assert len(params) == 2

    def test_disallow_multiple_judgement_roots(self):
        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        leaf_true = ConversationalVerdictNode(verdict=True, score=10)
        judgement_node1 = ConversationalBinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )
        judgement_node2 = ConversationalBinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )
        with pytest.raises(ValueError):
            DeepAcyclicGraph(
                root_nodes=[judgement_node1, judgement_node2],
            )

    def test_only_score_or_child(self):
        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        with pytest.raises(ValueError):
            ConversationalVerdictNode(
                verdict=True, score=10, child=[leaf_false]
            )

    def test_allow_multiple_tasknode_roots(self):
        node1 = ConversationalTaskNode("Task 1", "Label1", [], [])
        node2 = ConversationalTaskNode("Task 2", "Label2", [], [])
        dag = DeepAcyclicGraph(root_nodes=[node1, node2])
        assert is_valid_dag(dag, multiturn=True) is True

    def test_copy_graph_isolated_and_deep(self):
        INSTRUCTIONS = "Instruction 1:"
        OUTPUT_LABEL = "Output label"
        CRITERIA = "Criteria: "

        leaf_false = ConversationalVerdictNode(verdict=False, score=0)
        leaf_true = ConversationalVerdictNode(verdict=True, score=10)
        judgement_node = ConversationalBinaryJudgementNode(
            criteria=CRITERIA, children=[leaf_false, leaf_true]
        )
        task = ConversationalTaskNode(
            instructions=INSTRUCTIONS,
            output_label=OUTPUT_LABEL,
            evaluation_params=[],
            children=[judgement_node],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])

        copied = copy_graph(dag)
        copied_task = copied.root_nodes[0]
        copied_judge = copied_task.children[0]
        copied_leaf_false = copied_judge.children[0]
        copied_leaf_true = copied_judge.children[1]

        ids_set = {
            hash(dag),
            hash(leaf_false),
            hash(leaf_true),
            hash(judgement_node),
            hash(task),
            hash(copied),
            hash(copied_leaf_false),
            hash(copied_leaf_true),
            hash(copied_judge),
            hash(copied_task),
        }

        assert len(ids_set) == 10
        assert copied is not dag
        assert isinstance(copied_task, ConversationalTaskNode)
        assert isinstance(copied_judge, ConversationalBinaryJudgementNode)
        assert isinstance(copied_leaf_false, ConversationalVerdictNode)
        assert isinstance(copied_leaf_true, ConversationalVerdictNode)
        assert copied_task is not task
        assert copied_judge is not judgement_node
        assert copied_leaf_false is not leaf_false
        assert copied_leaf_true is not leaf_true
        assert copied_task.output_label == OUTPUT_LABEL
        assert copied_task.instructions == INSTRUCTIONS
        assert len(copied_task.children) == 1
        assert len(copied_judge.children) == 2
        assert copied_judge.criteria == CRITERIA
        assert copied_leaf_false.verdict is False
        assert copied_leaf_false.score == 0
        assert copied_leaf_true.verdict is True
        assert copied_leaf_true.score == 10

    def test_non_binary_node_in_dag(self):
        leaf1 = ConversationalVerdictNode(verdict="One", score=0.1)
        leaf2 = ConversationalVerdictNode(verdict="Two", score=0.3)
        leaf3 = ConversationalVerdictNode(verdict="Three", score=0.5)
        leaf4 = ConversationalVerdictNode(verdict="Four", score=0.7)
        non_binary = ConversationalNonBinaryJudgementNode(
            criteria="Evaluate based on: ",
            children=[leaf1, leaf2, leaf3, leaf4],
        )
        task = ConversationalTaskNode(
            instructions="Do task",
            output_label="test",
            evaluation_params=[],
            children=[non_binary],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag(dag, multiturn=True)

    def test_task_node_leaf(self):
        task = ConversationalTaskNode(
            instructions="Standalone task",
            output_label="standalone",
            evaluation_params=[TurnParams.ROLE],
            children=[],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag_from_roots(dag.root_nodes, multiturn=True)

    def test_verdict_node_with_child(self):
        leaf = ConversationalVerdictNode(verdict=False, score=0.0)
        verdict = ConversationalVerdictNode(verdict=True, child=leaf)
        judge = ConversationalBinaryJudgementNode(
            "Pass?",
            children=[
                ConversationalVerdictNode(verdict=False, score=0),
                verdict,
            ],
        )
        task = ConversationalTaskNode("Check", "result", [], [judge])
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag_from_roots(dag.root_nodes, multiturn=True)
