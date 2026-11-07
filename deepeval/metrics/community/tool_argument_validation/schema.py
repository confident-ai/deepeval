from typing import List, Optional
from pydantic import BaseModel, Field


class ToolCallValidationError(BaseModel):
    """Detailed error diagnostics for a single tool call argument violation."""

    tool_name: str
    call_index: int
    message: str
    error_type: str
    field_path: Optional[str] = None


class ToolArgumentValidationResult(BaseModel):
    """Validation result summary for an individual tool call."""

    tool_name: str
    call_index: int
    is_valid: bool
    errors: List[ToolCallValidationError] = Field(default_factory=list)
