import pytest
from deepeval.metrics.dag import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.dag.utils import (
    is_valid_dag_from_roots,
    extract_required_params,
    copy_graph,
    is_valid_dag,
)


class TestDeepAcyclicGraph:
    """Tests for DAG validation, copying, and parameter extraction."""

    @staticmethod
    def make_verdict_nodes():
        return VerdictNode(verdict=False, score=0), VerdictNode(
            verdict=True, score=1
        )

    @staticmethod
    def make_binary_judgement_node():
        leaf_false, leaf_true = TestDeepAcyclicGraph.make_verdict_nodes()
        return BinaryJudgementNode(
            criteria="?", children=[leaf_false, leaf_true]
        )

    def test_is_valid_dag_true(self):
        judgement_node = self.make_binary_judgement_node()
        root = TaskNode(
            instructions="Extract",
            output_label="X",
            children=[judgement_node],
            evaluation_params=[LLMTestCaseParams.INPUT],
        )
        assert is_valid_dag_from_roots([root]) is True

    def test_is_valid_dag_false_cycle(self):
        node_a = TaskNode(
            "Task A", output_label="A", evaluation_params=[], children=[]
        )
        node_b = TaskNode(
            "Task B", output_label="B", evaluation_params=[], children=[node_a]
        )
        node_a.children.append(node_b)
        assert is_valid_dag_from_roots([node_a]) is False

    def test_is_valid_dag_deep_nested_mixed_nodes(self):
        leaf_false, leaf_true = self.make_verdict_nodes()
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
        assert is_valid_dag(task) is True

    def test_extract_required_params(self):
        judgement_node = self.make_binary_judgement_node()
        judgement_node.evaluation_params = [LLMTestCaseParams.EXPECTED_OUTPUT]
        task = TaskNode(
            instructions="Extract something",
            output_label="abc",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            children=[judgement_node],
        )
        params = extract_required_params([task])
        assert LLMTestCaseParams.INPUT in params
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert len(params) == 3

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
        params = extract_required_params([task])
        assert LLMTestCaseParams.INPUT in params
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert len(params) == 2

    def test_disallow_multiple_judgement_roots(self):
        node1 = self.make_binary_judgement_node()
        node2 = self.make_binary_judgement_node()
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[node1, node2])

    def test_only_score_or_child(self):
        leaf_false = VerdictNode(verdict=False, score=0)
        with pytest.raises(ValueError):
            VerdictNode(verdict=True, score=10, child=[leaf_false])

    def test_allow_multiple_tasknode_roots(self):
        node1 = TaskNode("Task 1", "Label1", [], [])
        node2 = TaskNode("Task 2", "Label2", [], [])
        dag = DeepAcyclicGraph(root_nodes=[node1, node2])
        assert isinstance(dag, DeepAcyclicGraph)

    def test_copy_graph_isolated_and_deep(self):
        leaf_false, leaf_true = self.make_verdict_nodes()
        judge = BinaryJudgementNode(
            criteria="Check?", children=[leaf_false, leaf_true]
        )
        task = TaskNode(
            instructions="Do X",
            output_label="outX",
            evaluation_params=[],
            children=[judge],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        copied = copy_graph(dag)
        copied_task = copied.root_nodes[0]
        copied_judge = copied_task.children[0]
        copied_leaf_false = copied_judge.children[0]
        copied_leaf_true = copied_judge.children[1]
        assert copied is not dag
        assert copied_task is not task
        assert isinstance(copied_task, TaskNode)
        assert copied_task.output_label == "outX"
        assert copied_task.instructions == "Do X"
        assert len(copied_task.children) == 1
        assert copied_judge is not judge
        assert copied_judge.criteria == "Check?"
        assert len(copied_judge.children) == 2
        assert copied_leaf_false is not leaf_false
        assert copied_leaf_false.verdict is False
        assert copied_leaf_false.score == 0
        assert copied_leaf_true is not leaf_true
        assert copied_leaf_true.verdict is True
        assert copied_leaf_true.score == 1

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
        assert is_valid_dag_from_roots(dag.root_nodes)

    def test_task_node_leaf(self):
        task = TaskNode(
            instructions="Standalone task",
            output_label="standalone",
            evaluation_params=[LLMTestCaseParams.INPUT],
            children=[],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag_from_roots(dag.root_nodes)

    def test_verdict_node_with_child(self):
        leaf = VerdictNode(verdict=False, score=0.0)
        verdict = VerdictNode(verdict=True, child=leaf)
        judge = BinaryJudgementNode(
            "Pass?",
            children=[VerdictNode(verdict=False, score=0), verdict],
        )
        task = TaskNode("Check", "result", [], [judge])
        dag = DeepAcyclicGraph(root_nodes=[task])
        assert is_valid_dag_from_roots(dag.root_nodes)

    def test_is_valid_dag_multiple_roots_valid(self):
        task1 = TaskNode("Task 1", "A", [], [])
        task2 = TaskNode("Task 2", "B", [], [])
        assert is_valid_dag_from_roots([task1, task2]) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
