from typing import Any, Optional, List, Dict
from pydantic import BaseModel

from deepeval.test_case.llm_test_case import ToolCall


class InputParameters(BaseModel):
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tool_descriptions: Optional[Dict[str, str]] = None


class OutputParameters(BaseModel):
    content: Optional[Any] = None
    role: Optional[str] = None
    type: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    tools_called: Optional[List[ToolCall]] = None
