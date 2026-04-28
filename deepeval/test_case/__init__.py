import warnings

from .llm_test_case import (
    LLMTestCase,
    SingleTurnParams,
    ToolCall,
    ToolCallParams,
    MLLMImage,
)
from .conversational_test_case import (
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
)
from .arena_test_case import ArenaTestCase, Contestant
from .mcp import (
    MCPServer,
    MCPPromptCall,
    MCPResourceCall,
    MCPToolCall,
)

__all__ = [
    "LLMTestCase",
    "SingleTurnParams",
    "ToolCall",
    "ToolCallParams",
    "ConversationalTestCase",
    "Turn",
    "MultiTurnParams",
    "MCPServer",
    "MCPPromptCall",
    "MCPResourceCall",
    "MCPToolCall",
    "MLLMImage",
    "ArenaTestCase",
    "Contestant",
]


def __getattr__(name: str):
    if name == "LLMTestCaseParams":
        warnings.warn(
            "'LLMTestCaseParams' is deprecated and will be removed in a future "
            "release. Use 'SingleTurnParams' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SingleTurnParams
    if name == "TurnParams":
        warnings.warn(
            "'TurnParams' is deprecated and will be removed in a future "
            "release. Use 'MultiTurnParams' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return MultiTurnParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
