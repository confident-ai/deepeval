from pydantic import BaseModel, Field
from typing import List, Optional
from deepeval.test_case import MCPToolCall, MCPResourceCall, MCPPromptCall


class Task(BaseModel):
    task: str
    steps_taken: List[str]


class TaskScore(BaseModel):
    score: float
    reason: str


class ToolScore(BaseModel):
    score: float
    reason: str


class ArgsScore(BaseModel):
    score: float
    reason: str


# class MCPArgs(BaseModel):
#     tools_called: Optional[List[MCPToolCall]] = None
#     resources_called: Optional[List[MCPResourceCall]] = None
#     prompts_called: Optional[List[MCPPromptCall]] = None
