from pydantic import BaseModel


class ClassificationResult(BaseModel):
    label: str
    reason: str
