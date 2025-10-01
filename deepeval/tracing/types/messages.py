from typing import Any, Literal
from pydantic import BaseModel
from tools import ToolCall, ToolCallOutput

class BaseMessage(BaseModel):
    role: str
    content: Any
    type: Literal["text", "tool_call", "tool_call_output", "base"] = "base"

class TextMessage(BaseMessage):
    content: str
    type: Literal["text"] = "text"

class ToolCallMessage(BaseMessage):
    content: ToolCall
    type: Literal["tool_call"] = "tool_call"

class ToolCallOutputMessage(BaseMessage):
    content: ToolCallOutput
    type: Literal["tool_call_output"] = "tool_call_output"