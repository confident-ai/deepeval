from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class NoulQuestion(BaseModel):
    instructions: Any
    true: Optional[Any] = None
    false: Optional[Any] = None


class ChoiceQuestion(BaseModel):
    instructions: Any
    options: Dict[str, Any]


class ScoreQuestion(BaseModel):
    instructions: Any
    levels: List[Any]


SystemOneQuestion = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class NoulAnswer(BaseModel):
    probability: float


class ChoiceAnswer(BaseModel):
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    score: float
    probabilities: Dict[int, float]
    confidence: float

    @property
    def normalized(self) -> float:
        top = max(self.probabilities) if self.probabilities else 0
        return self.score / top if top > 0 else 0.0


class SystemOneAnswers(BaseModel):
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict)
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict)
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict)
