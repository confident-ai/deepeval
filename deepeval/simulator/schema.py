from pydantic import BaseModel


class ConversationCompletion(BaseModel):
    is_complete: bool
    reason: str


class SimulatedInput(BaseModel):
    simulated_input: str
