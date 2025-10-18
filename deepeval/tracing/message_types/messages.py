from typing import Literal, Dict, Any
from .base import BaseMessage


class TextMessage(BaseMessage):
    type: Literal["text", "thinking"]
    content: str


class ToolCallMessage(BaseMessage):
    """This is a message for tool calls in response.choices[0].message.tool_calls"""

    name: str
    args: Dict[str, Any]
