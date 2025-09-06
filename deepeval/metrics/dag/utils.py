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
from deepeval.test_case import LLMTestCaseParams, TurnParams


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
        Union[Set[LLMTestCaseParams], Set[TurnParams]]
    ] = None,
) -> Union[Set[LLMTestCaseParams], Set[TurnParams]]:
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
