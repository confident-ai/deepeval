from pydantic import BaseModel


class EnhancedConfusion(BaseModel):
    confusion_strategy: str
    input: str


class IsConfusionValid(BaseModel):
    is_valid_confusion: bool
