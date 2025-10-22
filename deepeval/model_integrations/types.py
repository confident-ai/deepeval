from typing import Any, Optional, List, Dict
from pydantic import BaseModel

from deepeval.test_case.llm_test_case import ToolCall


class InputParameters(BaseModel):
    model: Optional[str] = None
    input: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    instructions: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tool_descriptions: Optional[Dict[str, str]] = None
    # Anthropic Specific
    system: Optional[str] = None
    max_tokens: Optional[int] = None


class OutputParameters(BaseModel):
    output: Optional[Any] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tools_called: Optional[List[ToolCall]] = None
    # Anthropic Specific
    content: Optional[Any] = None
    role: Optional[str] = None
    type: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
