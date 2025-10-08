from typing import Literal, Dict, Any
from .base import BaseMessage

class TextMessage(BaseMessage):
    type: Literal["text", "thinking"]
    content: str

class ToolCallMessage(BaseMessage):
    name: str
    args: Dict[str, Any]