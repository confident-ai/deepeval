from pydantic import BaseModel


class EnhancedRedirection(BaseModel):
    redirection_reasoning: str
    input: str


class IsGoalRedirected(BaseModel):
    is_goal_redirected: bool
