"""Tests for DAG -> JSON serialization and deserialization."""

import json
from typing import Optional

import pytest

from deepeval.metrics.dag import (
    BinaryJudgementNode,
    ChildType,
    DeepAcyclicGraph,
    NodeType,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
    dag_from_dict,
    dag_from_json,
    dag_to_dict,
    dag_to_json,
)
from deepeval.metrics.conversational_dag import (
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalTaskNode,
    ConversationalVerdictNode,
)
from deepeval.metrics.dag.utils import is_valid_dag_from_roots
from deepeval.test_case import SingleTurnParams, MultiTurnParams


# ----------------------------------------------------------------------------
# Single-turn structural round-trips (no LLM dependency)
# ----------------------------------------------------------------------------


def _build_simple_single_turn_dag() -> DeepAcyclicGraph:
    leaf_false = VerdictNode(verdict=False, score=0)
    leaf_true = VerdictNode(verdict=True, score=10)
    judgement = BinaryJudgementNode(
        criteria="Is the output a summary?",
        children=[leaf_false, leaf_true],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
    )
    root = TaskNode(
        instructions="Extract the summary.",
        output_label="Summary",
        children=[judgement],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        label="extract",
    )
    return DeepAcyclicGraph(root_nodes=[root])


class TestSingleTurnRoundTrip:
    def test_dag_to_dict_shape(self):
        dag = _build_simple_single_turn_dag()
        data = dag_to_dict(dag)
        assert set(data.keys()) == {"nodes"}
        assert isinstance(data["nodes"], dict)
        # 1 task + 1 binary judgement + 2 verdict = 4 nodes
        assert len(data["nodes"]) == 4

    def test_dag_to_dict_ids_are_unique_uuids(self):
        from uuid import UUID

        dag = _build_simple_single_turn_dag()
        data = dag_to_dict(dag)
        ids = list(data["nodes"].keys())
        assert len(set(ids)) == len(ids)
        for node_id in ids:
            UUID(node_id, version=4)

    def test_dag_to_dict_node_types_use_enum_values(self):
        dag = _build_simple_single_turn_dag()
        data = dag_to_dict(dag)
        types = {spec["type"] for spec in data["nodes"].values()}
        assert NodeType.TASK.value in types
        assert NodeType.BINARY_JUDGEMENT.value in types
        assert NodeType.VERDICT.value in types

    def test_dag_to_dict_evaluation_params_serialized_as_strings(self):
        dag = _build_simple_single_turn_dag()
        data = dag_to_dict(dag)
        task_specs = [
            s
            for s in data["nodes"].values()
            if s["type"] == NodeType.TASK.value
        ]
        assert len(task_specs) == 1
        assert task_specs[0]["evaluation_params"] == [
            SingleTurnParams.ACTUAL_OUTPUT.value
        ]

    def test_dag_to_dict_verdict_with_score_only(self):
        dag = _build_simple_single_turn_dag()
        data = dag_to_dict(dag)
        verdict_specs = [
            s
            for s in data["nodes"].values()
            if s["type"] == NodeType.VERDICT.value
        ]
        assert len(verdict_specs) == 2
        for vs in verdict_specs:
            assert "score" in vs
            assert "child" not in vs

    def test_round_trip_via_dict_preserves_structure(self):
        dag = _build_simple_single_turn_dag()
        data = dag_to_dict(dag)
        rebuilt = dag_from_dict(data)
        assert rebuilt.multiturn is False
        assert len(rebuilt.root_nodes) == 1
        root = rebuilt.root_nodes[0]
        assert isinstance(root, TaskNode)
        assert root.instructions == "Extract the summary."
        assert root.output_label == "Summary"
        assert root.label == "extract"
        assert root.evaluation_params == [SingleTurnParams.ACTUAL_OUTPUT]
        assert len(root.children) == 1
        judge = root.children[0]
        assert isinstance(judge, BinaryJudgementNode)
        assert judge.criteria == "Is the output a summary?"
        assert judge.evaluation_params == [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ]
        assert {c.verdict for c in judge.children} == {True, False}
        assert {c.score for c in judge.children} == {0, 10}

    def test_round_trip_via_json_string(self):
        dag = _build_simple_single_turn_dag()
        s = dag_to_json(dag)
        # must be valid JSON
        json.loads(s)
        rebuilt = dag_from_json(s)
        assert is_valid_dag_from_roots(rebuilt.root_nodes, multiturn=False)

    def test_round_trip_via_graph_methods(self):
        dag = _build_simple_single_turn_dag()
        s = dag.to_json()
        rebuilt = DeepAcyclicGraph.from_json(s)
        assert isinstance(rebuilt, DeepAcyclicGraph)
        assert len(rebuilt.root_nodes) == 1


class TestNonBinaryJudgement:
    def test_non_binary_round_trip(self):
        v_a = VerdictNode(verdict="bullets", score=8)
        v_b = VerdictNode(verdict="paragraph", score=5)
        v_c = VerdictNode(verdict="none", score=0)
        judge = NonBinaryJudgementNode(
            criteria="Classify the format.",
            children=[v_a, v_b, v_c],
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        )
        dag = DeepAcyclicGraph(root_nodes=[judge])

        rebuilt = DeepAcyclicGraph.from_dict(dag.to_dict())
        assert isinstance(rebuilt.root_nodes[0], NonBinaryJudgementNode)
        rebuilt_verdicts = {
            c.verdict: c.score for c in rebuilt.root_nodes[0].children
        }
        assert rebuilt_verdicts == {"bullets": 8, "paragraph": 5, "none": 0}


class TestSharedChildDAG:
    """Shared children must remain a single Python object after deserialize."""

    def test_shared_judgement_node_is_one_object(self):
        # Two verdict branches both point at the same downstream judgement.
        leaf_no = VerdictNode(verdict=False, score=0)
        leaf_yes = VerdictNode(verdict=True, score=10)
        shared_judge = BinaryJudgementNode(
            criteria="Inner check?",
            children=[leaf_no, leaf_yes],
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            label="shared_judge",
        )
        wrap_a = VerdictNode(verdict="left", child=shared_judge)
        wrap_b = VerdictNode(verdict="right", child=shared_judge)
        wrap_c = VerdictNode(verdict="none", score=0)
        outer = NonBinaryJudgementNode(
            criteria="Pick a side",
            children=[wrap_a, wrap_b, wrap_c],
        )
        dag = DeepAcyclicGraph(root_nodes=[outer])

        data = dag_to_dict(dag)
        # The shared inner judge should appear ONCE (as a single node entry).
        shared_specs = [
            (nid, spec)
            for nid, spec in data["nodes"].items()
            if spec["type"] == NodeType.BINARY_JUDGEMENT.value
        ]
        assert len(shared_specs) == 1
        shared_id, _ = shared_specs[0]

        # Both verdict wrappers must reference the shared judge by its id.
        verdict_with_child_specs = [
            spec
            for spec in data["nodes"].values()
            if spec["type"] == NodeType.VERDICT.value and "child" in spec
        ]
        refs = [
            spec["child"]["ref"]
            for spec in verdict_with_child_specs
            if spec["child"]["type"] == ChildType.NODE.value
        ]
        assert refs.count(shared_id) == 2

        rebuilt = dag_from_dict(data)
        rebuilt_outer = rebuilt.root_nodes[0]
        wraps_with_child = [
            c for c in rebuilt_outer.children if c.child is not None
        ]
        assert len(wraps_with_child) == 2
        # The shared judge must be the SAME Python object via both wrappers.
        assert wraps_with_child[0].child is wraps_with_child[1].child


# ----------------------------------------------------------------------------
# Multiturn round-trip
# ----------------------------------------------------------------------------


def _build_simple_multiturn_dag() -> DeepAcyclicGraph:
    v_no = ConversationalVerdictNode(verdict=False, score=0)
    v_yes = ConversationalVerdictNode(verdict=True, score=10)
    judge = ConversationalBinaryJudgementNode(
        criteria="Did the assistant respond appropriately?",
        children=[v_no, v_yes],
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
    )
    return DeepAcyclicGraph(root_nodes=[judge])


class TestMultiturnRoundTrip:
    def test_multiturn_round_trip(self):
        dag = _build_simple_multiturn_dag()
        assert dag.multiturn is True

        s = dag.to_json()
        rebuilt = DeepAcyclicGraph.from_json(s, multiturn=True)
        assert rebuilt.multiturn is True
        root = rebuilt.root_nodes[0]
        assert isinstance(root, ConversationalBinaryJudgementNode)
        assert root.evaluation_params == [
            MultiTurnParams.CONTENT,
            MultiTurnParams.ROLE,
        ]
        assert {c.verdict for c in root.children} == {True, False}

    def test_multiturn_node_type_strings_are_mode_agnostic(self):
        """The JSON type strings do NOT include 'Conversational' prefix."""
        dag = _build_simple_multiturn_dag()
        data = dag_to_dict(dag)
        for spec in data["nodes"].values():
            assert not spec["type"].startswith("Conversational")
            # Must be a valid NodeType
            NodeType(spec["type"])

    def test_multiturn_task_node_turn_window_round_trip(self):
        v_no = ConversationalVerdictNode(verdict=False, score=0)
        v_yes = ConversationalVerdictNode(verdict=True, score=10)
        judge = ConversationalBinaryJudgementNode(
            criteria="?",
            children=[v_no, v_yes],
            evaluation_params=[MultiTurnParams.CONTENT],
        )
        task = ConversationalTaskNode(
            instructions="Look at first 2 turns",
            output_label="X",
            children=[judge],
            evaluation_params=[MultiTurnParams.CONTENT],
            turn_window=(0, 1),
        )
        dag = DeepAcyclicGraph(root_nodes=[task])
        rebuilt = DeepAcyclicGraph.from_dict(dag.to_dict(), multiturn=True)
        rebuilt_root = rebuilt.root_nodes[0]
        assert isinstance(rebuilt_root, ConversationalTaskNode)
        assert rebuilt_root.turn_window == (0, 1)


# ----------------------------------------------------------------------------
# Negative tests (no runtime LLM needed)
# ----------------------------------------------------------------------------


class TestNegative:
    def test_missing_nodes_key(self):
        with pytest.raises(ValueError, match="nodes"):
            dag_from_dict({})

    def test_empty_nodes(self):
        with pytest.raises(ValueError, match="non-empty"):
            dag_from_dict({"nodes": {}})

    def test_unknown_node_type(self):
        data = {
            "nodes": {
                "n0": {"type": "ImaginaryNode", "verdict": True, "score": 1},
            }
        }
        with pytest.raises(ValueError, match="unknown type"):
            dag_from_dict(data)

    def test_unknown_child_type_on_verdict(self):
        data = {
            "nodes": {
                "v": {
                    "type": NodeType.VERDICT.value,
                    "verdict": True,
                    "child": {"type": "made_up"},
                },
            }
        }
        with pytest.raises(ValueError, match="unknown type"):
            dag_from_dict(data)

    def test_unknown_metric_class(self):
        data = {
            "nodes": {
                "n0": {
                    "type": NodeType.BINARY_JUDGEMENT.value,
                    "criteria": "?",
                    "children": ["v_t", "v_f"],
                },
                "v_t": {
                    "type": NodeType.VERDICT.value,
                    "verdict": True,
                    "child": {
                        "type": ChildType.METRIC.value,
                        "metric_class": "DefinitelyNotARealMetric",
                        "kwargs": {},
                    },
                },
                "v_f": {
                    "type": NodeType.VERDICT.value,
                    "verdict": False,
                    "score": 0,
                },
            }
        }
        with pytest.raises(ValueError, match="Unknown metric_class"):
            dag_from_dict(data)

    def test_cycle_in_json_refs(self):
        """A node that references itself as a verdict child."""
        data = {
            "nodes": {
                "j1": {
                    "type": NodeType.BINARY_JUDGEMENT.value,
                    "criteria": "?",
                    "children": ["v_t", "v_f"],
                },
                "v_t": {
                    "type": NodeType.VERDICT.value,
                    "verdict": True,
                    "child": {"type": ChildType.NODE.value, "ref": "j1"},
                },
                "v_f": {
                    "type": NodeType.VERDICT.value,
                    "verdict": False,
                    "score": 0,
                },
            }
        }
        # Every node is referenced (j1 referenced by v_t.child) -> no roots.
        with pytest.raises(ValueError, match="root"):
            dag_from_dict(data)

    def test_invalid_evaluation_param_value(self):
        data = {
            "nodes": {
                "n0": {
                    "type": NodeType.TASK.value,
                    "instructions": "i",
                    "output_label": "o",
                    "evaluation_params": ["not_a_real_param"],
                    "children": ["v"],
                },
                "v": {
                    "type": NodeType.VERDICT.value,
                    "verdict": True,
                    "score": 5,
                },
            }
        }
        with pytest.raises(ValueError, match="evaluation_param"):
            dag_from_dict(data)


# ----------------------------------------------------------------------------
# Smoke: deserialized DAG plays nice with DAGMetric / validation utils
# ----------------------------------------------------------------------------


class TestSmoke:
    def test_deserialized_dag_passes_validation(self):
        dag = _build_simple_single_turn_dag()
        rebuilt = dag_from_json(dag_to_json(dag))
        assert is_valid_dag_from_roots(rebuilt.root_nodes, multiturn=False)
