from pydantic import BaseModel

class UserInputAndTools(BaseModel):
    user_messages: str
    assistant_messages: str
    tools_called: str
    available_tools: str

class ToolUseScore(BaseModel):
    score: float
    reason: str
