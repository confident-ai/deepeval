from pydantic import BaseModel, ConfigDict
from typing import Any, Optional


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    deprecated: Optional[bool] = None
    link: Optional[str] = None


class ConfidentApiError(Exception):
    """Custom exception that preserves API response metadata"""

    def __init__(self, message: str, link: Optional[str] = None):
        super().__init__(message)
        self.link = link
