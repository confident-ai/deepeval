from typing import Any, Optional
from pydantic import BaseModel

class BaseTool(BaseModel):
    name: str
    description: Optional[str] = None

class InputTool(BaseTool):
    parameters: Any

class ToolCallOutput(BaseTool):
    args: Any

class ToolOutput(BaseTool):
    output: Any