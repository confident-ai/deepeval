from enum import Enum
from typing import Literal


class ToolOrchestrationAbuseType(Enum):
    RECURSIVE_TOOL_CALLS = "recursive_tool_calls"
    UNSAFE_TOOL_COMPOSITION = "unsafe_tool_composition"
    TOOL_BUDGET_EXHAUSTION = "tool_budget_exhaustion"
    CROSS_TOOL_STATE_LEAKAGE = "cross_tool_state_leakage"


ToolOrchestrationAbuseTypes = Literal[
    ToolOrchestrationAbuseType.RECURSIVE_TOOL_CALLS.value,
    ToolOrchestrationAbuseType.UNSAFE_TOOL_COMPOSITION.value,
    ToolOrchestrationAbuseType.TOOL_BUDGET_EXHAUSTION.value,
    ToolOrchestrationAbuseType.CROSS_TOOL_STATE_LEAKAGE.value,
]
