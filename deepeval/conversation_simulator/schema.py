from pydantic import BaseModel


class FirstInput(BaseModel):
    first_input: str


class NextInput(BaseModel):
    next_input: str


class UserProfile(BaseModel):
    user_profile: str


class Scenario(BaseModel):
    scenario: str
