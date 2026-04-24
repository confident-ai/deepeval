"""JSON serialization for ``DeepAcyclicGraph``.

The JSON document describes ONLY graph structure. It does NOT encode mode
(single-turn vs multiturn), version, or root list - those are inferred or
supplied by the caller.

JSON shape::

    {
      "nodes": {
        "<id>": {
          "type": "TaskNode" | "BinaryJudgementNode" | ...,
          ... node-specific fields,
          "children": ["<id>", ...]                # for non-VerdictNode
        },
        "<id>": {
          "type": "VerdictNode",
          "verdict": <bool|str>,
          "score": <int>            # XOR with "child"
          | "child": {              # see ChildType for the discriminator
              "type": "node",   "ref": "<id>"
            | "type": "geval",  ...constructor kwargs
            | "type": "metric", "metric_class": "<class name>", "kwargs": {...}
          }
        }
      }
    }
"""

from __future__ import annotations

import importlib
import inspect
import json
import uuid
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from deepeval.metrics.base_metric import BaseConversationalMetric, BaseMetric
from deepeval.metrics.conversational_dag.nodes import (
    ConversationalBaseNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalTaskNode,
    ConversationalVerdictNode,
)
from deepeval.metrics.conversational_g_eval.conversational_g_eval import (
    ConversationalGEval,
)
from deepeval.metrics.dag.nodes import (
    BaseNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case import LLMTestCaseParams, TurnParams

from .registry import CLASS_TO_NODE_TYPE, NODE_CLASSES
from .types import ChildType, NodeType


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def dag_to_dict(dag) -> Dict[str, Any]:
    """Serialize a ``DeepAcyclicGraph`` instance to a JSON-friendly dict."""
    from deepeval.metrics.dag.utils import is_valid_dag_from_roots

    if not is_valid_dag_from_roots(
        root_nodes=dag.root_nodes, multiturn=dag.multiturn
    ):
        raise ValueError("Cycle detected in DAG graph; cannot serialize.")

    ordered_nodes = _walk_nodes(dag.root_nodes)
    id_map = _assign_ids(ordered_nodes)

    nodes_dict: Dict[str, Dict[str, Any]] = {}
    for node in ordered_nodes:
        node_id = id_map[id(node)]
        nodes_dict[node_id] = _serialize_node(node, id_map)

    return {"nodes": nodes_dict}


def dag_to_json(dag, indent: int = 2) -> str:
    return json.dumps(dag_to_dict(dag), indent=indent)


def dag_from_dict(data: Dict[str, Any], multiturn: bool = False):
    """Re-create a ``DeepAcyclicGraph`` from a dict produced by ``dag_to_dict``."""
    from deepeval.metrics.dag.graph import DeepAcyclicGraph

    if not isinstance(data, dict) or "nodes" not in data:
        raise ValueError(
            "Invalid DAG document: expected a top-level object with a 'nodes' key."
        )
    nodes_spec = data["nodes"]
    if not isinstance(nodes_spec, dict) or len(nodes_spec) == 0:
        raise ValueError(
            "Invalid DAG document: 'nodes' must be a non-empty object."
        )

    # Coerce/validate every node 'type' up front for clear errors.
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
    built: Dict[str, Any] = {}

    def build(node_id: str, stack: Set[str]):
        if node_id in built:
            return built[node_id]
        if node_id in stack:
            raise ValueError(
                f"Cycle detected in JSON refs involving node '{node_id}'."
            )
        if node_id not in nodes_spec:
            raise ValueError(f"Reference to unknown node id '{node_id}'.")

        stack.add(node_id)
        spec = nodes_spec[node_id]
        nt = NodeType(spec["type"])
        cls = class_map[nt]
        node: Any

        if nt == NodeType.VERDICT:
            node = _build_verdict(spec, cls, multiturn, build, stack)
        elif nt == NodeType.TASK:
            children = [build(cid, stack) for cid in spec.get("children", [])]
            kwargs = _task_kwargs(spec, multiturn)
            node = cls(children=children, **kwargs)
        elif nt in (NodeType.BINARY_JUDGEMENT, NodeType.NON_BINARY_JUDGEMENT):
            children = [build(cid, stack) for cid in spec.get("children", [])]
            kwargs = _judgement_kwargs(spec, multiturn)
            node = cls(children=children, **kwargs)
        else:
            raise ValueError(f"Unhandled node type '{nt}'.")

        stack.discard(node_id)
        built[node_id] = node
        return node

    root_nodes = [build(rid, set()) for rid in root_ids]
    return DeepAcyclicGraph(root_nodes=root_nodes)


def dag_from_json(s: str, multiturn: bool = False):
    return dag_from_dict(json.loads(s), multiturn=multiturn)


# ----------------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------------


def _walk_nodes(root_nodes: List[Any]) -> List[Any]:
    """BFS from each root, returning every reachable node exactly once,
    in stable BFS order (roots first)."""
    seen: Set[int] = set()
    ordered: List[Any] = []
    queue: deque = deque(root_nodes)
    while queue:
        node = queue.popleft()
        if id(node) in seen:
            continue
        seen.add(id(node))
        ordered.append(node)

        for child in _iter_children(node):
            queue.append(child)
    return ordered


def _iter_children(node: Any):
    if hasattr(node, "children") and node.children:
        for c in node.children:
            yield c
    if isinstance(node, (VerdictNode, ConversationalVerdictNode)):
        if node.child is not None and _is_node(node.child):
            yield node.child


def _is_node(obj: Any) -> bool:
    return isinstance(obj, (BaseNode, ConversationalBaseNode))


def _assign_ids(ordered_nodes: List[Any]) -> Dict[int, str]:
    """Assign a fresh uuid4 string to every node, keyed by id(node)."""
    return {id(node): str(uuid.uuid4()) for node in ordered_nodes}


def _serialize_node(node: Any, id_map: Dict[int, str]) -> Dict[str, Any]:
    cls = type(node)
    if cls not in CLASS_TO_NODE_TYPE:
        raise ValueError(
            f"Unsupported node class '{cls.__name__}'; cannot serialize."
        )
    nt = CLASS_TO_NODE_TYPE[cls]

    if nt == NodeType.TASK:
        out: Dict[str, Any] = {
            "type": nt.value,
            "instructions": node.instructions,
            "output_label": node.output_label,
            "label": node.label,
            "evaluation_params": _serialize_eval_params(node.evaluation_params),
            "children": [id_map[id(c)] for c in node.children],
        }
        if (
            isinstance(node, ConversationalTaskNode)
            and node.turn_window is not None
        ):
            out["turn_window"] = list(node.turn_window)
        return out

    if nt in (NodeType.BINARY_JUDGEMENT, NodeType.NON_BINARY_JUDGEMENT):
        out = {
            "type": nt.value,
            "criteria": node.criteria,
            "label": node.label,
            "evaluation_params": _serialize_eval_params(node.evaluation_params),
            "children": [id_map[id(c)] for c in node.children],
        }
        if (
            isinstance(
                node,
                (
                    ConversationalBinaryJudgementNode,
                    ConversationalNonBinaryJudgementNode,
                ),
            )
            and node.turn_window is not None
        ):
            out["turn_window"] = list(node.turn_window)
        return out

    if nt == NodeType.VERDICT:
        out = {"type": nt.value, "verdict": node.verdict}
        if node.score is not None:
            out["score"] = node.score
        if node.child is not None:
            out["child"] = _serialize_verdict_child(node.child, id_map)
        return out

    raise ValueError(f"Unhandled node type '{nt}'.")  # pragma: no cover


def _serialize_eval_params(params) -> Optional[List[str]]:
    if params is None:
        return None
    return [p.value for p in params]


def _serialize_verdict_child(
    child: Any, id_map: Dict[int, str]
) -> Dict[str, Any]:
    if _is_node(child):
        return {"type": ChildType.NODE.value, "ref": id_map[id(child)]}
    if isinstance(child, (GEval, ConversationalGEval)):
        return _serialize_geval(child)
    if isinstance(child, (BaseMetric, BaseConversationalMetric)):
        return _serialize_metric(child)
    raise ValueError(
        f"VerdictNode.child has unsupported type '{type(child).__name__}'. "
        "Expected a BaseNode, GEval/ConversationalGEval, or a BaseMetric/"
        "BaseConversationalMetric subclass."
    )


def _serialize_geval(geval: Any) -> Dict[str, Any]:
    init_params = _init_param_names(type(geval))
    payload: Dict[str, Any] = {"type": ChildType.GEVAL.value}
    for name in init_params:
        if name == "self":
            continue
        if not hasattr(geval, name):
            continue
        value = getattr(geval, name)
        if name == "evaluation_params":
            payload[name] = _serialize_eval_params(value)
            continue
        json_value = _maybe_jsonify(value)
        if json_value is _SKIP:
            continue
        payload[name] = json_value
    return payload


def _serialize_metric(metric: Any) -> Dict[str, Any]:
    cls = type(metric)
    init_params = _init_param_names(cls)
    kwargs: Dict[str, Any] = {}
    for name in init_params:
        if name == "self":
            continue
        if not hasattr(metric, name):
            continue
        value = getattr(metric, name)
        if name == "evaluation_params":
            serialized = _serialize_eval_params(value)
            if serialized is not None:
                kwargs[name] = serialized
            continue
        json_value = _maybe_jsonify(value)
        if json_value is _SKIP:
            continue
        kwargs[name] = json_value

    return {
        "type": ChildType.METRIC.value,
        "metric_class": cls.__name__,
        "kwargs": kwargs,
    }


def _init_param_names(cls: Type) -> List[str]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    return list(sig.parameters.keys())


_SKIP = object()


def _maybe_jsonify(value: Any) -> Any:
    """Return a JSON-friendly version of ``value`` or ``_SKIP`` if it cannot
    be safely round-tripped."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            jv = _maybe_jsonify(item)
            if jv is _SKIP:
                return _SKIP
            out.append(jv)
        return out
    if isinstance(value, dict):
        out_d: Dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                return _SKIP
            jv = _maybe_jsonify(v)
            if jv is _SKIP:
                return _SKIP
            out_d[k] = jv
        return out_d
    if isinstance(value, Enum):
        return _maybe_jsonify(value.value)
    # Anything else (DeepEvalBaseLLM instances, classes, callables, custom
    # objects, etc.) is skipped.
    return _SKIP


# ----------------------------------------------------------------------------
# Deserialization helpers
# ----------------------------------------------------------------------------


def _collect_referenced_ids(nodes_spec: Dict[str, Any]) -> Set[str]:
    referenced: Set[str] = set()
    for spec in nodes_spec.values():
        for cid in spec.get("children", []) or []:
            referenced.add(cid)
        child = spec.get("child")
        if (
            isinstance(child, dict)
            and child.get("type") == ChildType.NODE.value
        ):
            ref = child.get("ref")
            if isinstance(ref, str):
                referenced.add(ref)
    return referenced


def _eval_params_cls(multiturn: bool):
    return TurnParams if multiturn else LLMTestCaseParams


def _deserialize_eval_params(values, multiturn: bool):
    if values is None:
        return None
    enum_cls = _eval_params_cls(multiturn)
    out = []
    for v in values:
        try:
            out.append(enum_cls(v))
        except ValueError:
            valid = ", ".join(p.value for p in enum_cls)
            raise ValueError(
                f"Unknown evaluation_param '{v}'. Expected one of: {valid}."
            )
    return out


def _task_kwargs(spec: Dict[str, Any], multiturn: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "instructions": spec["instructions"],
        "output_label": spec["output_label"],
    }
    if "label" in spec and spec["label"] is not None:
        kwargs["label"] = spec["label"]
    if "evaluation_params" in spec:
        kwargs["evaluation_params"] = _deserialize_eval_params(
            spec["evaluation_params"], multiturn
        )
    if multiturn and "turn_window" in spec and spec["turn_window"] is not None:
        kwargs["turn_window"] = tuple(spec["turn_window"])
    return kwargs


def _judgement_kwargs(spec: Dict[str, Any], multiturn: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"criteria": spec["criteria"]}
    if "label" in spec and spec["label"] is not None:
        kwargs["label"] = spec["label"]
    if "evaluation_params" in spec:
        kwargs["evaluation_params"] = _deserialize_eval_params(
            spec["evaluation_params"], multiturn
        )
    if multiturn and "turn_window" in spec and spec["turn_window"] is not None:
        kwargs["turn_window"] = tuple(spec["turn_window"])
    return kwargs


def _build_verdict(
    spec: Dict[str, Any],
    cls: Type,
    multiturn: bool,
    build,
    stack: Set[str],
):
    verdict = spec["verdict"]
    if "score" in spec and spec["score"] is not None:
        return cls(verdict=verdict, score=spec["score"])
    if "child" not in spec or spec["child"] is None:
        raise ValueError(
            "VerdictNode spec must have either 'score' or 'child'."
        )
    child_spec = spec["child"]
    if not isinstance(child_spec, dict) or "type" not in child_spec:
        raise ValueError(
            "VerdictNode 'child' must be an object with a 'type' field."
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
            raise ValueError("VerdictNode child of type 'node' requires 'ref'.")
        child_obj = build(ref, stack)
    elif ctype == ChildType.GEVAL:
        child_obj = _build_geval(child_spec, multiturn)
    else:
        child_obj = _build_metric(child_spec)

    return cls(verdict=verdict, child=child_obj)


def _build_geval(child_spec: Dict[str, Any], multiturn: bool):
    cls = ConversationalGEval if multiturn else GEval
    kwargs = {k: v for k, v in child_spec.items() if k != "type"}
    if "evaluation_params" in kwargs:
        kwargs["evaluation_params"] = _deserialize_eval_params(
            kwargs["evaluation_params"], multiturn
        )
    return cls(**kwargs)


def _build_metric(child_spec: Dict[str, Any]):
    metric_class = child_spec.get("metric_class")
    if not isinstance(metric_class, str) or not metric_class:
        raise ValueError(
            "Metric child requires a non-empty 'metric_class' field."
        )
    metrics_module = importlib.import_module("deepeval.metrics")
    cls = getattr(metrics_module, metric_class, None)
    if cls is None:
        raise ValueError(
            f"Unknown metric_class '{metric_class}'. "
            f"It must be importable from 'deepeval.metrics'."
        )
    kwargs = dict(child_spec.get("kwargs") or {})
    # Reconstruct evaluation_params enum list if present.
    if "evaluation_params" in kwargs and isinstance(
        kwargs["evaluation_params"], list
    ):
        # Try LLMTestCaseParams first, then TurnParams for conversational metrics.
        if issubclass(cls, BaseConversationalMetric):
            kwargs["evaluation_params"] = _deserialize_eval_params(
                kwargs["evaluation_params"], multiturn=True
            )
        else:
            kwargs["evaluation_params"] = _deserialize_eval_params(
                kwargs["evaluation_params"], multiturn=False
            )
    return cls(**kwargs)
