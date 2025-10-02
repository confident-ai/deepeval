from typing import Any, Optional, Dict
from pydantic import BaseModel

class BaseTool(BaseModel):
    name: str
    description: Optional[str] = None

class InputTool(BaseTool):
    parameters: Dict[str, Any]

class ToolCallOutput(BaseTool):
    args: Dict[str, Any]

class ToolOutput(BaseTool):
    output: Any