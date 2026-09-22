from pydantic import BaseModel


class Reason(BaseModel):
    reason: str
