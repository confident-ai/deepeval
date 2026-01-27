from pydantic import BaseModel
from typing import Any, Optional

from deepeval.utils import make_model_config


class ApiResponse(BaseModel):
    model_config = make_model_config(extra="ignore")

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
