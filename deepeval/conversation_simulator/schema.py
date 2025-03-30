from pydantic import BaseModel


class SimulatedInput(BaseModel):
    simulated_input: str


class UserProfile(BaseModel):
    user_profile: str


class Scenario(BaseModel):
    scenario: str
