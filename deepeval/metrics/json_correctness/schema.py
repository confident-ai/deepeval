from pydantic import BaseModel


class JsonCorrectnessScoreReason(BaseModel):
    reason: str
