import pytest
from deepeval.metrics.dag import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import SingleTurnParams
from deepeval.metrics.dag.utils import (
    is_valid_dag_from_roots,
    extract_required_params,
    copy_graph,
    is_valid_dag,
)
from deepeval.metrics.dag.nodes import propagate_skip


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


class TestPropagateSkip:
    """Tests for ``propagate_skip``."""

    def test_decrements_self_indegree(self):
        leaf = VerdictNode(verdict=True, score=10)
        BinaryJudgementNode(
            criteria="?",
            children=[VerdictNode(verdict=False, score=0), leaf],
        )
        assert leaf._indegree == 1
        propagate_skip(leaf)
        assert leaf._indegree == 0

    def test_recurses_through_verdict_child_into_subtree(self):
        inner_yes = VerdictNode(verdict=True, score=10)
        inner_no = VerdictNode(verdict=False, score=0)
        inner_judge = BinaryJudgementNode(
            criteria="inner?", children=[inner_yes, inner_no]
        )
        outer_a = VerdictNode(verdict="a", child=inner_judge)
        outer_b = VerdictNode(verdict="b", score=0)
        NonBinaryJudgementNode(criteria="outer?", children=[outer_a, outer_b])

        propagate_skip(outer_a)

        assert outer_a._indegree == 0
        assert inner_judge._indegree == 0
        assert inner_yes._indegree == 0
        assert inner_no._indegree == 0

    def test_shared_downstream_node_decrements_once_per_dead_path(self):
        """Each construction-time +1 must be matched by exactly one skip."""
        shared_leaf_t = VerdictNode(verdict=True, score=10)
        shared_leaf_f = VerdictNode(verdict=False, score=0)
        shared_judge = BinaryJudgementNode(
            criteria="shared?", children=[shared_leaf_t, shared_leaf_f]
        )
        a = VerdictNode(verdict="a", child=shared_judge)
        b = VerdictNode(verdict="b", child=shared_judge)
        NonBinaryJudgementNode(criteria="?", children=[a, b])
        assert shared_judge._indegree == 2

        propagate_skip(a)
        propagate_skip(b)

        assert shared_judge._indegree == 0
        assert shared_leaf_t._indegree == 0
        assert shared_leaf_f._indegree == 0

    def test_no_op_on_non_basenode(self):
        propagate_skip(None)
        propagate_skip("not a node")


class TestDAGMetricReuse:
    """End-to-end: a ``DAGMetric`` instance must produce correct scores and
    clean verbose logs across repeated ``.measure()`` calls on the same dag."""

    def _build_metric_with_mocked_judge(self, outer_verdicts, inner_verdicts):
        from unittest.mock import patch
        from deepeval.metrics import DAGMetric
        from deepeval.metrics.dag.schema import BinaryJudgementVerdict

        inner_true = VerdictNode(verdict=True, score=10)
        inner_false = VerdictNode(verdict=False, score=5)
        inner_judge = BinaryJudgementNode(
            criteria="inner?",
            children=[inner_true, inner_false],
            evaluation_params=[SingleTurnParams.INPUT],
        )
        outer_true = VerdictNode(verdict=True, child=inner_judge)
        outer_false = VerdictNode(verdict=False, score=0)
        outer_judge = BinaryJudgementNode(
            criteria="outer?",
            children=[outer_true, outer_false],
            evaluation_params=[SingleTurnParams.INPUT],
        )
        task = TaskNode(
            instructions="dummy",
            output_label="result",
            evaluation_params=[SingleTurnParams.INPUT],
            children=[outer_judge],
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        metric = DAGMetric(
            name="test", dag=dag, async_mode=False, _include_dag_suffix=False
        )

        outer_iter = iter(outer_verdicts)
        inner_iter = iter(inner_verdicts)

        def fake_extract(
            metric, prompt, schema_cls, extract_schema, extract_json
        ):
            if schema_cls is BinaryJudgementVerdict:
                # The two judges share the same schema; distinguish by which
                # iter still has elements (outer is consumed first per turn).
                if "outer?" in prompt:
                    return BinaryJudgementVerdict(
                        verdict=next(outer_iter), reason="mocked"
                    )
                return BinaryJudgementVerdict(
                    verdict=next(inner_iter), reason="mocked"
                )
            from deepeval.metrics.dag.schema import TaskNodeOutput

            return TaskNodeOutput(output="mocked task output")

        patcher = patch(
            "deepeval.metrics.dag.nodes.generate_with_schema_and_extract",
            side_effect=fake_extract,
        )
        return metric, patcher

    def test_score_resets_across_measure_calls(self):
        """Consecutive measurements with different verdicts produce different scores."""
        from deepeval.test_case import LLMTestCase

        metric, patcher = self._build_metric_with_mocked_judge(
            outer_verdicts=[True, False],
            inner_verdicts=[True],
        )
        patcher.start()
        try:
            metric.measure(LLMTestCase(input="first", actual_output="x"))
            first_score = metric.score  # outer=True → inner=True → 10/10
            metric.measure(LLMTestCase(input="second", actual_output="x"))
            second_score = metric.score  # outer=False → 0
            assert first_score == 1.0
            assert second_score == 0
        finally:
            patcher.stop()

    def test_shared_downstream_judgement_node_fires_via_either_path(self):
        """JudgementNode shared between two sibling verdict branches must fire
        on whichever branch the verdict picks.
        """
        from unittest.mock import patch
        from deepeval.metrics import DAGMetric
        from deepeval.metrics.dag.schema import BinaryJudgementVerdict
        from deepeval.test_case import LLMTestCase

        shared_judge = BinaryJudgementNode(
            criteria="shared?",
            children=[
                VerdictNode(verdict=True, score=10),
                VerdictNode(verdict=False, score=5),
            ],
            evaluation_params=[SingleTurnParams.INPUT],
        )
        root = BinaryJudgementNode(
            criteria="root?",
            children=[
                VerdictNode(verdict=True, child=shared_judge),
                VerdictNode(verdict=False, child=shared_judge),
            ],
            evaluation_params=[SingleTurnParams.INPUT],
        )
        metric = DAGMetric(
            name="t",
            dag=DeepAcyclicGraph(root_nodes=[root]),
            async_mode=False,
            _include_dag_suffix=False,
        )

        def fake(metric, prompt, schema_cls, **_):
            return BinaryJudgementVerdict(verdict=True, reason="m")

        with patch(
            "deepeval.metrics.dag.nodes.generate_with_schema_and_extract",
            side_effect=fake,
        ):
            metric.measure(LLMTestCase(input="x", actual_output="y"))

        assert metric.score == 1.0

    def test_verbose_steps_does_not_accumulate(self):
        from deepeval.test_case import LLMTestCase

        metric, patcher = self._build_metric_with_mocked_judge(
            outer_verdicts=[True, True, True],
            inner_verdicts=[True, False, True],
        )
        patcher.start()
        try:
            metric.measure(LLMTestCase(input="A", actual_output="x"))
            first = len(metric._verbose_steps)
            metric.measure(LLMTestCase(input="B", actual_output="x"))
            second = len(metric._verbose_steps)
            metric.measure(LLMTestCase(input="C", actual_output="x"))
            third = len(metric._verbose_steps)
            assert first == second == third
        finally:
            patcher.stop()
