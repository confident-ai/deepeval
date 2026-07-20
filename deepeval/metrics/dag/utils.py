from typing import Set, Dict, Optional, Union
import inspect

from deepeval.metrics.dag import (
    BaseNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
    TaskNode,
    DeepAcyclicGraph,
)
from deepeval.metrics.conversational_dag import (
    ConversationalBaseNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalTaskNode,
    ConversationalVerdictNode,
)
from deepeval.test_case import SingleTurnParams, MultiTurnParams


def is_valid_dag_from_roots(
    root_nodes: Union[list[BaseNode], list[ConversationalBaseNode]],
    multiturn: bool,
) -> bool:
    visited = set()
    for root in root_nodes:
        if not is_valid_dag(root, multiturn, visited, set()):
            return False
    return True


def is_valid_dag(
    node: Union[BaseNode, ConversationalBaseNode],
    multiturn: bool,
    visited=None,
    stack=None,
) -> bool:
    if visited is None:
        visited = set()
    if stack is None:
        stack = set()

    if node in stack:
        return False
    if node in visited:
        return True

    visited.add(node)
    stack.add(node)
    if not multiturn:
        if (
            isinstance(node, TaskNode)
            or isinstance(node, BinaryJudgementNode)
            or isinstance(node, NonBinaryJudgementNode)
        ):
            for child in node.children:
                if not is_valid_dag(child, multiturn, visited, stack):
                    return False
    else:
        if (
            isinstance(node, ConversationalTaskNode)
            or isinstance(node, ConversationalBinaryJudgementNode)
            or isinstance(node, ConversationalNonBinaryJudgementNode)
        ):
            for child in node.children:
                if not is_valid_dag(child, multiturn, visited, stack):
                    return False

    stack.remove(node)
    return True


def extract_required_params(
    nodes: list[BaseNode],
    multiturn: bool,
    required_params: Optional[
        Union[Set[SingleTurnParams], Set[MultiTurnParams]]
    ] = None,
) -> Union[Set[SingleTurnParams], Set[MultiTurnParams]]:
    if required_params is None:
        required_params = set()

    for node in nodes:
        if not multiturn:
            if (
                isinstance(node, TaskNode)
                or isinstance(node, BinaryJudgementNode)
                or isinstance(node, NonBinaryJudgementNode)
            ):
                if node.evaluation_params is not None:
                    required_params.update(node.evaluation_params)
                extract_required_params(
                    node.children, multiturn, required_params
                )
        else:
            if (
                isinstance(node, ConversationalTaskNode)
                or isinstance(node, ConversationalBinaryJudgementNode)
                or isinstance(node, ConversationalNonBinaryJudgementNode)
            ):
                if node.evaluation_params is not None:
                    required_params.update(node.evaluation_params)
                extract_required_params(
                    node.children, multiturn, required_params
                )

    return required_params


def copy_graph(original_dag: DeepAcyclicGraph) -> DeepAcyclicGraph:
    # This mapping avoids re-copying nodes that appear in multiple places.
    visited: Union[
        Dict[BaseNode, BaseNode],
        Dict[ConversationalBaseNode, ConversationalBaseNode],
    ] = {}

    def copy_node(
        node: Union[BaseNode, ConversationalBaseNode],
    ) -> Union[BaseNode, ConversationalBaseNode]:
        if node in visited:
            return visited[node]

        node_class = type(node)
        args = vars(node)
        superclasses = node_class.__mro__
        valid_params = []
        for superclass in superclasses:
            signature = inspect.signature(superclass.__init__)
            superclass_params = signature.parameters.keys()
            valid_params.extend(superclass_params)
        valid_params = set(valid_params)
        valid_args = {
            key: args[key]
            for key in valid_params
            if key in args
            and key
            not in [
                "children",
                "child",
                "_parents",
                "_parent",
                "_verdict",
                "_indegree",
                "_depth",
            ]
        }
        if not original_dag.multiturn:
            if (
                isinstance(node, TaskNode)
                or isinstance(node, BinaryJudgementNode)
                or isinstance(node, NonBinaryJudgementNode)
            ):
                copied_node = node_class(
                    **valid_args,
                    children=[copy_node(child) for child in node.children]
                )
            else:
                if isinstance(node, VerdictNode) and node.child:
                    copied_node = node_class(
                        **valid_args, child=copy_node(node.child)
                    )
                else:
                    copied_node = node_class(**valid_args)
        else:
            if (
                isinstance(node, ConversationalTaskNode)
                or isinstance(node, ConversationalBinaryJudgementNode)
                or isinstance(node, ConversationalNonBinaryJudgementNode)
            ):
                copied_node = node_class(
                    **valid_args,
                    children=[copy_node(child) for child in node.children]
                )
            else:
                if isinstance(node, ConversationalVerdictNode) and node.child:
                    copied_node = node_class(
                        **valid_args, child=copy_node(node.child)
                    )
                else:
                    copied_node = node_class(**valid_args)

        visited[node] = copied_node
        return copied_node

    # Copy all root nodes (the recursion handles the rest).
    new_root_nodes = [copy_node(root) for root in original_dag.root_nodes]
    return DeepAcyclicGraph(new_root_nodes)


def serialize_dag_to_payload(dag) -> dict:
    from deepeval.metrics.base_metric import (
        BaseConversationalMetric,
        BaseMetric,
    )
    from deepeval.metrics.g_eval.g_eval import GEval
    from deepeval.metrics.conversational_g_eval.conversational_g_eval import (
        ConversationalGEval,
    )
    from deepeval.metrics.dag.serialization.registry import CLASS_TO_NODE_TYPE
    from deepeval.metrics.dag.serialization.serialization import (
        _assign_ids,
        _is_node,
        _serialize_node,
        _walk_nodes,
    )
    from deepeval.metrics.dag.serialization.types import ChildType, NodeType

    if not is_valid_dag_from_roots(
        root_nodes=dag.root_nodes, multiturn=dag.multiturn
    ):
        raise ValueError("Cycle detected in DAG graph; cannot serialize.")

    def serialize_child(child, id_map):
        if _is_node(child):
            return {"type": ChildType.NODE.value, "ref": id_map[id(child)]}
        if isinstance(child, (GEval, ConversationalGEval)):
            if getattr(child, "metric_id", None) is None:
                child.upload()
            return {"type": ChildType.METRIC.value, "metric_name": child.name}
        if isinstance(child, (BaseMetric, BaseConversationalMetric)):
            name = getattr(child, "name", None) or child.__name__
            return {"type": ChildType.METRIC.value, "metric_name": name}
        raise ValueError(
            f"VerdictNode.child has unsupported type '{type(child).__name__}'. "
            "Expected a node, GEval/ConversationalGEval, or a BaseMetric/"
            "BaseConversationalMetric subclass."
        )

    def serialize_node(node, id_map):
        nt = CLASS_TO_NODE_TYPE[type(node)]
        if nt != NodeType.VERDICT:
            return _serialize_node(node, id_map)
        out = {"type": nt.value, "verdict": node.verdict}
        if node.score is not None:
            out["score"] = node.score
        if node.child is not None:
            out["child"] = serialize_child(node.child, id_map)
        return out

    ordered_nodes = _walk_nodes(dag.root_nodes)
    id_map = _assign_ids(ordered_nodes)
    return {
        "nodes": {
            id_map[id(node)]: serialize_node(node, id_map)
            for node in ordered_nodes
        }
    }


def construct_dag_upload_payload(name: str, dag, multi_turn: bool = False) -> dict:
    return {
        "name": name,
        "algorithm": "DAG",
        "multiTurn": multi_turn,
        "dag": serialize_dag_to_payload(dag),
    }


def build_dag_from_payload(payload: dict, *, multiturn: bool):
    from deepeval.metrics.dag.graph import DeepAcyclicGraph
    from deepeval.metrics.dag.serialization.registry import NODE_CLASSES
    from deepeval.metrics.dag.serialization.serialization import (
        _collect_referenced_ids,
        _judgement_kwargs,
        _task_kwargs,
    )
    from deepeval.metrics.dag.serialization.types import ChildType, NodeType

    if not isinstance(payload, dict) or "nodes" not in payload:
        raise ValueError(
            "Invalid DAG document: expected an object with a 'nodes' key."
        )
    nodes_spec = payload["nodes"]
    if not isinstance(nodes_spec, dict) or len(nodes_spec) == 0:
        raise ValueError(
            "Invalid DAG document: 'nodes' must be a non-empty object."
        )

    for node_id, spec in nodes_spec.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(
                f"Node '{node_id}' is missing required 'type' field."
            )
        try:
            NodeType(spec["type"])
        except ValueError:
            valid = ", ".join(nt.value for nt in NodeType)
            raise ValueError(
                f"Node '{node_id}' has unknown type '{spec['type']}'. "
                f"Expected one of: {valid}."
            )

    referenced = _collect_referenced_ids(nodes_spec)
    root_ids = [nid for nid in nodes_spec.keys() if nid not in referenced]
    if not root_ids:
        raise ValueError(
            "No root nodes detected (every node is referenced as a child); "
            "graph would be empty or contain a cycle."
        )

    class_map = NODE_CLASSES[bool(multiturn)]
    built = {}

    def build_verdict(spec, cls, stack):
        verdict = spec["verdict"]
        if spec.get("score") is not None:
            return cls(verdict=verdict, score=spec["score"])

        child_spec = spec.get("child")
        if not isinstance(child_spec, dict) or "type" not in child_spec:
            raise ValueError(
                "VerdictNode spec must have either 'score' or a 'child' object."
            )
        try:
            ctype = ChildType(child_spec["type"])
        except ValueError:
            valid = ", ".join(c.value for c in ChildType)
            raise ValueError(
                f"VerdictNode child has unknown type '{child_spec['type']}'. "
                f"Expected one of: {valid}."
            )

        if ctype == ChildType.NODE:
            ref = child_spec.get("ref")
            if not isinstance(ref, str):
                raise ValueError(
                    "VerdictNode child of type 'node' requires 'ref'."
                )
            child_obj = build(ref, stack)
        elif ctype == ChildType.METRIC:
            name = child_spec.get("metric_name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "VerdictNode metric child requires a 'metric_name'."
                )
            child_obj = pull_child_metric_by_name(name, multiturn=multiturn)
        else:
            raise ValueError(
                f"VerdictNode child type '{ctype.value}' is not supported "
                "when pulling from the platform."
            )
        return cls(verdict=verdict, child=child_obj)

    def build(node_id, stack):
        if node_id in built:
            return built[node_id]
        if node_id in stack:
            raise ValueError(
                f"Cycle detected in DAG refs involving node '{node_id}'."
            )
        if node_id not in nodes_spec:
            raise ValueError(f"Reference to unknown node id '{node_id}'.")

        stack.add(node_id)
        spec = nodes_spec[node_id]
        nt = NodeType(spec["type"])
        cls = class_map[nt]

        if nt == NodeType.VERDICT:
            node = build_verdict(spec, cls, stack)
        elif nt == NodeType.TASK:
            children = [build(cid, stack) for cid in spec.get("children", [])]
            node = cls(children=children, **_task_kwargs(spec, multiturn))
        else:
            children = [build(cid, stack) for cid in spec.get("children", [])]
            node = cls(children=children, **_judgement_kwargs(spec, multiturn))

        stack.discard(node_id)
        built[node_id] = node
        return node

    root_nodes = [build(rid, set()) for rid in root_ids]
    return DeepAcyclicGraph(root_nodes=root_nodes)


def pull_child_metric_by_name(name: str, *, multiturn: bool):
    from deepeval.metrics.g_eval.g_eval import GEval
    from deepeval.metrics.conversational_g_eval.conversational_g_eval import (
        ConversationalGEval,
    )

    metric_cls = ConversationalGEval if multiturn else GEval
    child = metric_cls(name=name)
    child.pull()
    return child
