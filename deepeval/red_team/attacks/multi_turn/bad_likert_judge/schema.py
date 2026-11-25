from pydantic import BaseModel


class LikertScore(BaseModel):
    score: int
    reasoning: str


class LikertExamples(BaseModel):
    score_1_example: str
    score_3_example: str


class RefinedExample(BaseModel):
    refined_example: str


class BLJJudge(BaseModel):
    harmful: bool
    reasoning: str


class NewAttackPrompt(BaseModel):
    new_attack_prompt: str
