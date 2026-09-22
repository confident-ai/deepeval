from pydantic import BaseModel


class ClassificationResult(BaseModel):
    label: str
    reason: str


# Experimental (DEEPEVAL_MODE=experimental); see EXPERIMENTAL.md.
class Reason(BaseModel):
    reason: str
