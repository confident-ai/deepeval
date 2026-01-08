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

DAG_NODE_MAPPING = {
    "taskNode": TaskNode,
    "binaryJudgementNode": BinaryJudgementNode,
    "nonBinaryJudgementNode": NonBinaryJudgementNode,
    "verdictNode": VerdictNode,
}

CONVERSATIONAL_DAG_NODE_MAPPING = {
    "taskNode": ConversationalTaskNode,
    "binaryJudgementNode": ConversationalBinaryJudgementNode,
    "nonBinaryJudgementNode": ConversationalNonBinaryJudgementNode,
    "verdictNode": ConversationalVerdictNode,
}

PARAMS_MAPPING = {
    "outputLabel": "output_label",
    "evaluationParams": "evaluation_params",
}

DAG_API_PARAMS_MAPPING = {
    "input": LLMTestCaseParams.INPUT,
    "actualOutput": LLMTestCaseParams.ACTUAL_OUTPUT,
    "expectedOutput": LLMTestCaseParams.EXPECTED_OUTPUT,
    "context": LLMTestCaseParams.CONTEXT,
    "retrievalContext": LLMTestCaseParams.RETRIEVAL_CONTEXT,
    "expectedTools": LLMTestCaseParams.EXPECTED_TOOLS,
    "toolsCalled": LLMTestCaseParams.TOOLS_CALLED,
}

CONVERSATIONAL_DAG_API_PARAMS_MAPPING = {
    "role": TurnParams.ROLE,
    "content": TurnParams.CONTENT,
    "scenario": TurnParams.SCENARIO,
    "expectedOutcome": TurnParams.EXPECTED_OUTCOME,
    "retrievalContext": TurnParams.RETRIEVAL_CONTEXT,
    "toolsCalled": TurnParams.TOOLS_CALLED,
}


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

def normalize_params(payload: dict, param_mapping: Dict) -> dict:
    normalized = {}

    for key, value in payload.items():
        if key in {"name", "children", "child"}:
            continue

        python_key = param_mapping.get(key, key)
        normalized[python_key] = value

    return normalized

def deserialize_node(
    node_dict: Dict,
    multi_turn: bool,
):  
    if multi_turn is False:
        nodes_mapping = DAG_NODE_MAPPING
        api_params_mapping = DAG_API_PARAMS_MAPPING
    else:
        nodes_mapping = CONVERSATIONAL_DAG_NODE_MAPPING
        api_params_mapping = CONVERSATIONAL_DAG_API_PARAMS_MAPPING

    node_type = node_dict["name"]

    if node_type not in nodes_mapping:
        raise ValueError(f"Unknown node type: {node_type}")

    cls = nodes_mapping[node_type]

    kwargs = normalize_params(node_dict, PARAMS_MAPPING)

    if "evaluation_params" in kwargs:
        if kwargs["evaluation_params"] is not None:
            new_params = []
            for param in kwargs["evaluation_params"]:
                new_params.append(api_params_mapping[param])
            kwargs["evaluation_params"] = new_params

    if node_dict.get("children") is not None:
        kwargs["children"] = [
            deserialize_node(child, multi_turn=multi_turn)
            for child in node_dict["children"]
        ]
        return cls(**kwargs)

    if node_dict.get("child") is not None:
        kwargs["child"] = deserialize_node(
            node_dict["child"],
            multi_turn=multi_turn,
        )
        return cls(**kwargs)

    return cls(**kwargs)

def parse_dag_metric(dag_dict: Dict):
    from deepeval.metrics import DAGMetric, ConversationalDAGMetric

    if not dag_dict.get("isDag") or "dag" not in dag_dict:
        raise ValueError(f"Invalid dictionary for a 'DAGMetric' {dag_dict}.")

    multi_turn = dag_dict.get("multiTurn", False)

    cls = ConversationalDAGMetric if multi_turn else DAGMetric

    root_nodes = [
        deserialize_node(node, multi_turn=multi_turn)
        for node in dag_dict["dag"].get("rootNodes", [])
    ]

    dag = DeepAcyclicGraph(root_nodes=root_nodes)

    return cls(
        name=dag_dict["name"],
        dag=dag,
    )