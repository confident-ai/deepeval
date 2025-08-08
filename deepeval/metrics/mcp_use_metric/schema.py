from pydantic import BaseModel


class MCPPrimitivesScore(BaseModel):
    score: float
    reason: str


class MCPArgsScore(BaseModel):
    score: float
    reason: str
