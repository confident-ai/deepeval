from typing import Literal
from pydantic import BaseModel

# Literal types for better autocomplete and docs
SequentialJailbreakTypeLiteral = Literal[
    "dialogue", "question_bank", "game_environment"
]

DialogueTypeLiteral = Literal["prisoner", "student", "researcher", "generic"]


class RewrittenDialogue(BaseModel):
    paraphrased_version: str


class DialogueJudge(BaseModel):
    harmful: bool
    reasoning: str


class QuestionBankJudge(BaseModel):
    harmful: bool
    reasoning: str


class GameEnvironmentJudge(BaseModel):
    harmful: bool
    reasoning: str


class ImprovedAttack(BaseModel):
    improved_attack: str


class AdaptiveDialogueTemplate(BaseModel):
    template: str


class AdaptiveQuestionBankTemplate(BaseModel):
    template: str


class AdaptiveGameEnvironmentTemplate(BaseModel):
    template: str
