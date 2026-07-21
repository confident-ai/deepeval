import os
import warnings

import pytest
from deepeval import evaluate
from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics.dag.utils import (
    is_valid_dag_from_roots,
    extract_required_params,
    copy_graph,
    is_valid_dag,
)

requires_openai = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
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
            evaluation_params=[SingleTurnParams.INPUT],
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
            evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT],
        )
        task = TaskNode(
            instructions="Extract something",
            output_label="abc",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            children=[judgement_node],
        )
        params = extract_required_params([task], multiturn=False)
        assert SingleTurnParams.INPUT in params
        assert SingleTurnParams.ACTUAL_OUTPUT in params
        assert SingleTurnParams.EXPECTED_OUTPUT in params
        assert len(params) == 3

    def test_invalid_child_type(self):
        invalid_child = "string_instead_of_node"  # Invalid child type
        with pytest.raises(TypeError):
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
            evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT],
        )
        task = TaskNode(
            instructions="Analyze",
            output_label="xyz",
            evaluation_params=[SingleTurnParams.INPUT],
            children=[non_binary],
        )
        params = extract_required_params([task], multiturn=False)
        assert SingleTurnParams.INPUT in params
        assert SingleTurnParams.EXPECTED_OUTPUT in params
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
            evaluation_params=[SingleTurnParams.INPUT],
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

    def test_add_node_appends_and_returns(self):
        task = TaskNode(
            instructions="Extract",
            output_label="X",
            evaluation_params=[SingleTurnParams.INPUT],
        )
        judge = BinaryJudgementNode(criteria="?")
        returned = task.add_node(judge)
        assert returned is judge
        assert task.children == [judge]

    def test_add_verdict_score_leaf(self):
        judge = BinaryJudgementNode(criteria="?")
        verdict = judge.add_verdict(True, score=10)
        assert isinstance(verdict, VerdictNode)
        assert verdict.verdict is True
        assert verdict.score == 10
        assert verdict.child is None
        assert judge.children == [verdict]

    def test_add_verdict_then_sets_child(self):
        order = NonBinaryJudgementNode(criteria="order?")
        judge = BinaryJudgementNode(criteria="?")
        verdict = judge.add_verdict(True, then=order)
        assert verdict.child is order
        assert verdict.score is None

    def test_add_verdict_rejects_score_and_then(self):
        order = NonBinaryJudgementNode(criteria="order?")
        judge = BinaryJudgementNode(criteria="?")
        with pytest.raises(ValueError):
            judge.add_verdict(True, score=10, then=order)

    def test_top_down_builds_valid_diamond(self):
        extract = TaskNode(
            instructions="Extract",
            output_label="X",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        )
        headings = BinaryJudgementNode(criteria="all three?")
        order = NonBinaryJudgementNode(criteria="order?")
        extract.add_node(headings)
        extract.add_node(order)  # diamond: shared by extract and the True verdict
        headings.add_verdict(False, score=0)
        headings.add_verdict(True, then=order)
        order.add_verdict("Yes", score=10)
        order.add_verdict("No", score=0)

        dag = DeepAcyclicGraph(root_nodes=[extract])
        assert is_valid_dag_from_roots(dag.root_nodes, multiturn=False)
        assert dag.indegree[extract] == 0
        assert dag.indegree[order] == 2

    def test_build_time_validation_incomplete_binary(self):
        extract = TaskNode(
            instructions="Extract",
            output_label="X",
            evaluation_params=[SingleTurnParams.INPUT],
        )
        judge = BinaryJudgementNode(criteria="?")
        extract.add_node(judge)
        judge.add_verdict(True, score=10)  # only one verdict
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[extract])

    def test_nonbinary_schema_deferred_to_build(self):
        order = NonBinaryJudgementNode(criteria="order?")
        order.add_verdict("A", score=10)
        order.add_verdict("B", score=0)
        assert not hasattr(order, "_verdict_schema")
        DeepAcyclicGraph(root_nodes=[order])
        assert hasattr(order, "_verdict_schema")
        assert sorted(order._verdict_options) == ["A", "B"]


class TestTopDownBuilder:

    def test_top_down_build_emits_no_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            extract = TaskNode(
                instructions="Extract",
                output_label="X",
                evaluation_params=[SingleTurnParams.INPUT],
            )
            judge = BinaryJudgementNode(criteria="?")
            extract.add_node(judge)
            judge.add_verdict(True, score=10)
            judge.add_verdict(False, score=0)
            DeepAcyclicGraph(root_nodes=[extract])
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]

    def test_bottom_up_children_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning):
            BinaryJudgementNode(
                criteria="?",
                children=[
                    VerdictNode(verdict=False, score=0),
                    VerdictNode(verdict=True, score=10),
                ],
            )

    def test_add_verdict_requires_score_or_then(self):
        judge = BinaryJudgementNode(criteria="?")
        with pytest.raises(ValueError):
            judge.add_verdict(True)  # neither score nor then

    def test_binary_two_true_verdicts_raises_at_build(self):
        judge = BinaryJudgementNode(criteria="?")
        judge.add_verdict(True, score=10)
        judge.add_verdict(True, score=0)
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[judge])

    def test_binary_non_bool_verdict_raises_at_build(self):
        judge = BinaryJudgementNode(criteria="?")
        judge.add_verdict("yes", score=10)
        judge.add_verdict("no", score=0)
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[judge])

    def test_nonbinary_empty_raises_at_build(self):
        order = NonBinaryJudgementNode(criteria="order?")
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[order])

    def test_nonbinary_duplicate_verdicts_raises_at_build(self):
        order = NonBinaryJudgementNode(criteria="order?")
        order.add_verdict("A", score=10)
        order.add_verdict("A", score=0)
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[order])

    def test_task_node_with_verdict_child_raises_at_build(self):
        task = TaskNode(
            instructions="Extract",
            output_label="X",
            evaluation_params=[SingleTurnParams.INPUT],
        )
        task.add_node(VerdictNode(verdict=True, score=10))
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[task])

    def test_cycle_via_add_node_raises_at_build(self):
        a = TaskNode(
            instructions="A",
            output_label="A",
            evaluation_params=[SingleTurnParams.INPUT],
        )
        b = TaskNode(
            instructions="B",
            output_label="B",
            evaluation_params=[SingleTurnParams.INPUT],
        )
        a.add_node(b)
        b.add_node(a)
        with pytest.raises(ValueError):
            DeepAcyclicGraph(root_nodes=[a])

    def test_copy_graph_top_down_is_warning_free_and_isolated(self):
        task = TaskNode(
            instructions="Extract",
            output_label="X",
            evaluation_params=[SingleTurnParams.INPUT],
        )
        judge = BinaryJudgementNode(criteria="criteria")
        task.add_node(judge)
        judge.add_verdict(True, score=10)
        judge.add_verdict(False, score=0)
        dag = DeepAcyclicGraph(root_nodes=[task])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copied = copy_graph(dag)
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]

        copied_task = copied.root_nodes[0]
        copied_judge = copied_task.children[0]
        assert copied is not dag
        assert isinstance(copied_task, TaskNode)
        assert isinstance(copied_judge, BinaryJudgementNode)
        assert copied_task is not task
        assert copied_judge is not judge
        assert copied_judge.criteria == "criteria"
        assert len(copied_judge.children) == 2
        assert {c.verdict for c in copied_judge.children} == {True, False}
        assert {c.score for c in copied_judge.children} == {10, 0}


@requires_openai
class TestDAGMetric:

    @staticmethod
    def _build_dag() -> DeepAcyclicGraph:
        extract = TaskNode(
            instructions="Extract the refund period, in days, from the output.",
            output_label="Refund period",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        )
        judge = BinaryJudgementNode(
            criteria="Is the extracted refund period exactly 30 days?"
        )
        extract.add_node(judge)
        judge.add_verdict(True, score=10)
        judge.add_verdict(False, score=0)
        return DeepAcyclicGraph(root_nodes=[extract])

    @staticmethod
    def _test_case() -> LLMTestCase:
        return LLMTestCase(
            input="What if these shoes don't fit?",
            actual_output="We offer a 30-day full refund at no extra cost.",
        )

    def test_dag_metric_sync_measure(self):
        metric = DAGMetric(
            name="Refund Period", dag=self._build_dag(), async_mode=False
        )
        metric.measure(self._test_case())
        assert metric.score is not None
        assert 0 <= metric.score <= 1
        assert metric.reason is not None

    def test_dag_metric_async_measure(self):
        metric = DAGMetric(
            name="Refund Period", dag=self._build_dag(), async_mode=True
        )
        metric.measure(self._test_case())
        assert metric.score is not None
        assert 0 <= metric.score <= 1
        assert metric.reason is not None

    def test_dag_metric_via_evaluate(self):
        metric = DAGMetric(name="Refund Period", dag=self._build_dag())
        evaluate([self._test_case()], [metric])
