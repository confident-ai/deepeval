from typing import Any
from pydantic import BaseModel

class BaseTool(BaseModel):
    name: str
    description: str

class ToolCall(BaseTool):
    parameters: Any

class ToolCallOutput(BaseTool):
    args: Any