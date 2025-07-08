from pydantic import BaseModel


class JsonCorrectnessReason(BaseModel):
    reason: str
