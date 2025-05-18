from pydantic import BaseModel


class ConversationCompletion(BaseModel):
    is_complete: bool
    reason: str


class SimulatedInput(BaseModel):
    simulated_input: str


class UserProfile(BaseModel):
    user_profile: str


class Scenario(BaseModel):
    scenario: str
