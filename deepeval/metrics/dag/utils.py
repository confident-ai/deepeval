from deepeval.metrics.dag import (
    BaseNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from deepeval.test_case import LLMTestCaseParams


def is_valid_dag(node: BaseNode, visited=None, stack=None) -> bool:
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
    if (
        isinstance(node, TaskNode)
        or isinstance(node, BinaryJudgementNode)
        or isinstance(node, NonBinaryJudgementNode)
    ):
        for child in node.children:
            if not is_valid_dag(child, visited, stack):
                return False
    elif isinstance(node, VerdictNode) and node.child is not None:
        if not is_valid_dag(node.child, visited, stack):
            return False

    stack.remove(node)
    return True


from typing import Set


def extract_required_params(
    node: BaseNode, required_params: Set[LLMTestCaseParams] = None
) -> Set[LLMTestCaseParams]:
    if required_params is None:
        required_params = set()

    # Traverse children based on the node type
    if (
        isinstance(node, TaskNode)
        or isinstance(node, BinaryJudgementNode)
        or isinstance(node, NonBinaryJudgementNode)
    ):
        if node.evaluation_params is not None:
            required_params.update(node.evaluation_params)
        for child in node.children:
            extract_required_params(child, required_params)
    elif isinstance(node, VerdictNode) and node.child is not None:
        extract_required_params(node.child, required_params)

    return required_params
