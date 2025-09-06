from pydantic import BaseModel
from typing import List


class HumanLikenessWinner(BaseModel):
    winner: str
    reason: str


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ModelConversation(BaseModel):
    model_name: str
    turns: List[ConversationTurn]
    conversation_starter: str
