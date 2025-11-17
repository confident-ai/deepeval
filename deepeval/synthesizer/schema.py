from typing import List, Literal
from pydantic import BaseModel


class RewrittenInput(BaseModel):
    rewritten_input: str


class InputFeedback(BaseModel):
    score: float
    feedback: str


class SyntheticData(BaseModel):
    input: str


class SyntheticDataList(BaseModel):
    data: List[SyntheticData]


class SQLData(BaseModel):
    sql: str


class ComplianceData(BaseModel):
    non_compliant: bool


class Response(BaseModel):
    response: str


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    response: bool


class Rating(BaseModel):
    number: int


class TreeScore(BaseModel):
    answer_1: int
    answer_2: int
    answer_3: int


class NonRefusal(BaseModel):
    classification: Literal["Non-refusal", "Refusal"]


class PromptStyling(BaseModel):
    scenario: str
    task: str
    input_format: str


class ConversationalScenario(BaseModel):
    scenario: str


class ConversationalScenarioList(BaseModel):
    data: List[ConversationalScenario]


class RewrittenScenario(BaseModel):
    rewritten_scenario: str


class ScenarioFeedback(BaseModel):
    score: float
    feedback: str


class ConversationalPromptStyling(BaseModel):
    scenario_context: str
    conversational_task: str
    participant_roles: str
