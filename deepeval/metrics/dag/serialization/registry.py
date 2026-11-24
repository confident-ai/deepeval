from typing import Dict, Type

from deepeval.metrics.dag.nodes import (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    TaskNode,
    VerdictNode,
)
from deepeval.metrics.conversational_dag.nodes import (
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalTaskNode,
    ConversationalVerdictNode,
)

from .types import NodeType


NODE_CLASSES: Dict[bool, Dict[NodeType, Type]] = {
    False: {
        NodeType.TASK: TaskNode,
        NodeType.BINARY_JUDGEMENT: BinaryJudgementNode,
        NodeType.NON_BINARY_JUDGEMENT: NonBinaryJudgementNode,
        NodeType.VERDICT: VerdictNode,
    },
    True: {
        NodeType.TASK: ConversationalTaskNode,
        NodeType.BINARY_JUDGEMENT: ConversationalBinaryJudgementNode,
        NodeType.NON_BINARY_JUDGEMENT: ConversationalNonBinaryJudgementNode,
        NodeType.VERDICT: ConversationalVerdictNode,
    },
}


CLASS_TO_NODE_TYPE: Dict[Type, NodeType] = {
    cls: nt
    for mode_map in NODE_CLASSES.values()
    for nt, cls in mode_map.items()
}
