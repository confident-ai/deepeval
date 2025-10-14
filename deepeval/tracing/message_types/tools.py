from typing import Any, Optional, Dict
from pydantic import BaseModel


class BaseTool(BaseModel):
    name: str
    description: Optional[str] = None


class ToolSchema(BaseTool):
    parameters: Dict[str, Any]
    is_called: Optional[bool] = False


class ToolOutput(BaseTool):
    """Output of the tool function"""

    output: Any
