from pydantic import BaseModel


class VoiceStyleVerdict(BaseModel):
    score: int
    reason: str
