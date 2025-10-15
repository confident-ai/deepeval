from typing import Literal, Dict, Any, Optional
from .base import BaseMessage


class TextMessage(BaseMessage):
    type: Literal["text", "thinking"]
    content: str


class ToolCallMessage(BaseMessage):
    """This is a message for tool calls in response.choices[0].message.tool_calls"""

    id: str
    name: str
    args: Dict[str, Any]
    
class ToolOutputMessage(BaseMessage):
    """Output of the tool function"""

    id: str
    output: Any
