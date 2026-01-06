from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Type
from deepeval.prompt.api import (
    ModelSettings,
    OutputType,
    PromptInterpolationType,
    PromptMessage,
    PromptType,
)


class PromptBase(BaseModel):
    alias: Optional[str] = None
    text_template: Optional[str] = None
    messages_template: Optional[List[PromptMessage]] = None
    model_settings: Optional[ModelSettings] = None
    output_type: Optional[OutputType] = None
    output_schema: Optional[Type[BaseModel]] = None
    interpolation_type: Optional[PromptInterpolationType] = None
    label: Optional[str] = None
    type: Optional[PromptType] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
