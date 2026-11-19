from pydantic import BaseModel


class AudioGlitchReason(BaseModel):
    reason: str
