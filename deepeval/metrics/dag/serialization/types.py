from enum import Enum


class NodeType(str, Enum):
    TASK = "TaskNode"
    BINARY_JUDGEMENT = "BinaryJudgementNode"
    NON_BINARY_JUDGEMENT = "NonBinaryJudgementNode"
    VERDICT = "VerdictNode"


class ChildType(str, Enum):
    NODE = "node"
    GEVAL = "geval"
    METRIC = "metric"
