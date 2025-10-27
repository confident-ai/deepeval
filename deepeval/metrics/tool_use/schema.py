from pydantic import BaseModel


class UserInputAndTools(BaseModel):
    user_messages: str
    assistant_messages: str
    tools_called: str
    available_tools: str
    tools_used: bool


class ToolSelectionScore(BaseModel):
    score: float
    reason: str


class ArgumentCorrectnessScore(BaseModel):
    score: float
    reason: str
